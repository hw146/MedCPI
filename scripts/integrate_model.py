#!/usr/bin/env python3
"""Model components for the Integrate stage."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from path_roots import INTEGRATE_ROOT
from integrate_dataset import (
    IntegratePreviewDataset,
    NODE_KIND_VOCAB,
    SOURCE_TYPE_VOCAB,
    build_concept_vocab,
    collate_integrate_batch,
    iter_jsonl,
)
TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]
FUSION_STRATEGIES = ["cross_attention", "concat_mlp", "gated", "film"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-htt-layers", type=int, default=1)
    parser.add_argument("--num-rgcn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-strategy", choices=FUSION_STRATEGIES, default="cross_attention")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--emit-spec", action="store_true")
    parser.add_argument("--run-forward", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def artifact_name(base: str, suffix: str) -> str:
    if not suffix:
        return base
    stem, ext = base.rsplit(".", 1)
    return "{0}_{1}.{2}".format(stem, suffix, ext)


def output_suffix(args: argparse.Namespace) -> str:
    if args.output_tag:
        return args.output_tag
    if args.fusion_strategy != "cross_attention":
        return args.fusion_strategy
    return ""


def build_spec(dataset: str, task: str, split: str, suffix: str = "") -> dict:
    task_root = INTEGRATE_ROOT / dataset / task
    return {
        "stage": "integrate_forward",
        "dataset": dataset,
        "task": task,
        "split": split,
        "inputs": {
            "sample_preview": str(task_root / f"{split}_integrate_samples.preview.jsonl"),
            "relation_vocab": str(task_root / "relation_vocab.json"),
        },
        "outputs": {
            "task_root": str(task_root),
            "spec_path": str(task_root / artifact_name(f"{split}_forward_spec.json", suffix)),
            "summary_path": str(task_root / artifact_name(f"{split}_forward_smoke_summary.json", suffix)),
        },
    }


def ensure_inputs_exist(spec: dict) -> None:
    missing = [path for path in spec["inputs"].values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def task_granularity(task: str) -> str:
    if task in {"mortality", "readmission_30d"}:
        return "visit"
    return "patient"


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(values.dtype)
    total = (values * weight).sum(dim=dim)
    denom = weight.sum(dim=dim).clamp(min=1.0)
    return total / denom


def masked_node_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(values, mask.unsqueeze(-1), dim=1)


class TimeAwareEHREncoder(nn.Module):
    def __init__(
        self,
        concept_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.concept_embedding = nn.Embedding(concept_vocab_size, embedding_dim, padding_idx=0)
        self.source_type_embedding = nn.Embedding(len(SOURCE_TYPE_VOCAB), embedding_dim, padding_idx=0)
        self.visit_projection = nn.Linear(embedding_dim, hidden_dim)
        self.position_embedding = nn.Embedding(256, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        ehr_concept_ids: torch.Tensor,
        ehr_source_type_ids: torch.Tensor,
        ehr_concept_mask: torch.Tensor,
        visit_mask: torch.Tensor,
        ehr_time_delta_days: torch.Tensor,
        granularity: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concept_embed = self.concept_embedding(ehr_concept_ids)
        source_embed = self.source_type_embedding(ehr_source_type_ids)
        visit_concepts = concept_embed + source_embed
        visit_repr = masked_mean(visit_concepts, ehr_concept_mask.unsqueeze(-1), dim=2)
        visit_repr = self.visit_projection(visit_repr)

        batch_size, num_visits = visit_repr.shape[:2]
        visit_positions = torch.arange(num_visits, device=visit_repr.device).unsqueeze(0).expand(batch_size, -1)
        visit_repr = visit_repr + self.position_embedding(visit_positions)
        visit_repr = visit_repr + self.time_mlp(ehr_time_delta_days.unsqueeze(-1))
        visit_repr = self.input_dropout(visit_repr)

        visit_hidden = self.transformer(visit_repr, src_key_padding_mask=~visit_mask)
        if granularity == "visit":
            last_visit_idx = visit_mask.sum(dim=1) - 1
            query = visit_hidden[torch.arange(batch_size, device=visit_hidden.device), last_visit_idx]
        else:
            scores = self.pool_score(visit_hidden).squeeze(-1)
            scores = scores.masked_fill(~visit_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            query = torch.sum(visit_hidden * weights.unsqueeze(-1), dim=1)
        return query, visit_hidden


class SimpleRGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.rel_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(node_states)
        deg = torch.zeros((node_states.size(0), 1), device=node_states.device, dtype=node_states.dtype)
        if edge_type.numel() > 0:
            for rel_id in torch.unique(edge_type).tolist():
                rel_mask = edge_type == rel_id
                src = edge_index[0, rel_mask]
                dst = edge_index[1, rel_mask]
                messages = self.rel_linears[rel_id](node_states[src])
                agg.index_add_(0, dst, messages)
                deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=node_states.device, dtype=node_states.dtype))
        agg = agg / deg.clamp(min=1.0)
        return self.dropout(torch.relu(self.self_linear(node_states) + agg))


class PKGEncoder(nn.Module):
    def __init__(
        self,
        concept_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_relations: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.concept_embedding = nn.Embedding(concept_vocab_size, embedding_dim, padding_idx=0)
        self.node_kind_embedding = nn.Embedding(len(NODE_KIND_VOCAB), embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([SimpleRGCNLayer(hidden_dim, num_relations, dropout) for _ in range(num_layers)])

    def forward(
        self,
        pkg_node_concept_ids: torch.Tensor,
        pkg_node_kind_ids: torch.Tensor,
        pkg_node_mask: torch.Tensor,
        pkg_edge_index: torch.Tensor,
        pkg_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        node_embed = self.concept_embedding(pkg_node_concept_ids) + self.node_kind_embedding(pkg_node_kind_ids)
        node_embed = self.input_dropout(self.input_projection(node_embed))
        encoded = torch.zeros_like(node_embed)
        node_counts = pkg_node_mask.sum(dim=1).tolist()

        offset = 0
        for batch_idx, count_value in enumerate(node_counts):
            node_count = int(count_value)
            if node_count == 0:
                continue
            local_states = node_embed[batch_idx, :node_count]
            local_start = offset
            local_end = offset + node_count
            edge_mask = (
                (pkg_edge_index[0] >= local_start)
                & (pkg_edge_index[0] < local_end)
                & (pkg_edge_index[1] >= local_start)
                & (pkg_edge_index[1] < local_end)
            )
            local_edge_index = pkg_edge_index[:, edge_mask] - local_start
            local_edge_type = pkg_edge_type[edge_mask]
            for layer in self.layers:
                local_states = layer(local_states, local_edge_index, local_edge_type)
            encoded[batch_idx, :node_count] = local_states
            offset = local_end
        return encoded


class MedCPIIntegrateModel(nn.Module):
    def __init__(
        self,
        concept_vocab_size: int,
        relation_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_htt_layers: int,
        num_rgcn_layers: int,
        dropout: float,
        granularity: str,
        fusion_strategy: str = "cross_attention",
    ):
        super().__init__()
        self.granularity = granularity
        self.fusion_strategy = fusion_strategy
        self.ehr_encoder = TimeAwareEHREncoder(
            concept_vocab_size=concept_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_htt_layers,
            dropout=dropout,
        )
        self.pkg_encoder = PKGEncoder(
            concept_vocab_size=concept_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_relations=relation_vocab_size,
            num_layers=num_rgcn_layers,
            dropout=dropout,
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.concat_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gated_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.film_linear = nn.Linear(hidden_dim, hidden_dim * 2)
        self.film_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        query, visit_hidden = self.ehr_encoder(
            ehr_concept_ids=batch["ehr_concept_ids"],
            ehr_source_type_ids=batch["ehr_source_type_ids"],
            ehr_concept_mask=batch["ehr_concept_mask"],
            visit_mask=batch["visit_mask"],
            ehr_time_delta_days=batch["ehr_time_delta_days"],
            granularity=self.granularity,
        )
        pkg_hidden = self.pkg_encoder(
            pkg_node_concept_ids=batch["pkg_node_concept_ids"],
            pkg_node_kind_ids=batch["pkg_node_kind_ids"],
            pkg_node_mask=batch["pkg_node_mask"],
            pkg_edge_index=batch["pkg_edge_index"],
            pkg_edge_type=batch["pkg_edge_type"],
        )
        pkg_pooled = masked_node_mean(pkg_hidden, batch["pkg_node_mask"])
        attn_weights = None

        if self.fusion_strategy == "cross_attention":
            cross_context, attn_weights = self.cross_attention(
                query=query.unsqueeze(1),
                key=pkg_hidden,
                value=pkg_hidden,
                key_padding_mask=~batch["pkg_node_mask"],
                need_weights=True,
            )
            fused = self.concat_fusion(torch.cat([query, cross_context.squeeze(1)], dim=-1))
        elif self.fusion_strategy == "concat_mlp":
            fused = self.concat_fusion(torch.cat([query, pkg_pooled], dim=-1))
        elif self.fusion_strategy == "gated":
            gate = torch.sigmoid(self.gate_linear(torch.cat([query, pkg_pooled], dim=-1)))
            fused = self.gated_fusion(gate * query + (1.0 - gate) * pkg_pooled)
        elif self.fusion_strategy == "film":
            gamma_beta = self.film_linear(pkg_pooled)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            modulated = (1.0 + gamma) * query + beta
            fused = self.film_fusion(modulated)
        else:
            raise ValueError("Unsupported fusion_strategy: {0}".format(self.fusion_strategy))

        logits = self.prediction_head(fused).squeeze(-1)
        return {
            "logits": logits,
            "visit_hidden": visit_hidden,
            "pkg_hidden": pkg_hidden,
            "attn_weights": attn_weights,
        }


def main() -> None:
    args = parse_args()
    suffix = output_suffix(args)
    spec = build_spec(args.dataset, args.task, args.split, suffix)
    ensure_inputs_exist(spec)

    samples = list(iter_jsonl(Path(spec["inputs"]["sample_preview"])))[: args.limit]
    relation_vocab = json.loads(Path(spec["inputs"]["relation_vocab"]).read_text())
    concept_vocab = build_concept_vocab(samples)
    granularity = task_granularity(args.task)

    if args.emit_spec:
        write_json(Path(spec["outputs"]["spec_path"]), spec)

    if args.run_forward:
        dataset = IntegratePreviewDataset(samples=samples, concept_vocab=concept_vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=collate_integrate_batch,
        )
        batch = next(iter(dataloader))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        model = MedCPIIntegrateModel(
            concept_vocab_size=len(concept_vocab["concept_to_id"]),
            relation_vocab_size=len(relation_vocab["relation_to_id"]),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_htt_layers=args.num_htt_layers,
            num_rgcn_layers=args.num_rgcn_layers,
            dropout=args.dropout,
            granularity=granularity,
            fusion_strategy=args.fusion_strategy,
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch)

        summary = {
            "dataset": args.dataset,
            "task": args.task,
            "split": args.split,
            "granularity": granularity,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "fusion_strategy": args.fusion_strategy,
            "batch_size": len(batch["instance_ids"]),
            "instance_ids": batch["instance_ids"],
            "embedding_dim": args.embedding_dim,
            "concept_vocab_size": len(concept_vocab["concept_to_id"]),
            "relation_vocab_size": len(relation_vocab["relation_to_id"]),
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_htt_layers": args.num_htt_layers,
            "num_rgcn_layers": args.num_rgcn_layers,
            "dropout": args.dropout,
            "shape_logits": list(outputs["logits"].shape),
            "shape_visit_hidden": list(outputs["visit_hidden"].shape),
            "shape_pkg_hidden": list(outputs["pkg_hidden"].shape),
            "shape_attn_weights": list(outputs["attn_weights"].shape) if outputs["attn_weights"] is not None else None,
            "shape_ehr_time_delta_days": list(batch["ehr_time_delta_days"].shape),
            "logits_preview": [float(x) for x in outputs["logits"].detach().cpu().tolist()],
        }
        write_json(Path(spec["outputs"]["summary_path"]), summary)

    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
