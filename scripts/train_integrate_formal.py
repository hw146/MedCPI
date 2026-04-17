#!/usr/bin/env python3
"""Run formal-scale Integrate training with full manifests, checkpointing, and evaluation."""

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from path_roots import ALIGNED_ROOT, CONSTRUCT_ROOT, INTEGRATE_ROOT, PERSONALIZE_ROOT, SPLITS_ROOT
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from integrate_data import (
    build_spec as build_integrate_data_spec,
    build_relation_vocab,
    iter_jsonl,
    load_patient_row,
    load_segment_rows,
    make_instance_id,
    materialize_sample,
    selected_visit_ids,
)
from integrate_dataset import collate_integrate_batch, tensorize_sample
from integrate_model import FUSION_STRATEGIES, MedCPIIntegrateModel, task_granularity

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-htt-layers", type=int, default=1)
    parser.add_argument("--num-rgcn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-strategy", choices=FUSION_STRATEGIES, default="cross_attention")
    parser.add_argument(
        "--selection-metric",
        choices=["auroc", "auprc", "loss"],
        default="auroc",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--emit-summary", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


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


def combine_suffix(*parts: str) -> str:
    cleaned = [part for part in parts if part]
    return "_".join(cleaned)


def add_concept(concept_to_id: dict, concept_id: str | None) -> None:
    if concept_id and concept_id not in concept_to_id:
        concept_to_id[concept_id] = len(concept_to_id)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def resolve_selection_metric_value(metric_name: str, valid_metrics: dict) -> float:
    if metric_name == "auprc":
        if valid_metrics["auprc"] is not None:
            return valid_metrics["auprc"]
        return -valid_metrics["loss"]
    if metric_name == "auroc":
        if valid_metrics["auroc"] is not None:
            return valid_metrics["auroc"]
        return -valid_metrics["loss"]
    if metric_name == "loss":
        return -valid_metrics["loss"]
    raise ValueError("Unsupported selection metric: {0}".format(metric_name))


def parse_seed_values(args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [args.seed]
    seeds = []
    for token in args.seeds.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def summarize_numeric(values: list[float | None]) -> dict | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    if len(numeric) == 1:
        return {"mean": numeric[0], "std": 0.0, "min": numeric[0], "max": numeric[0]}
    return {
        "mean": statistics.mean(numeric),
        "std": statistics.stdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def write_progress(
    path: Path,
    *,
    seed: int,
    args: argparse.Namespace,
    suffix: str,
    device: torch.device,
    relation_vocab: dict,
    concept_vocab: dict,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    checkpoint_path: Path,
    total_params: int,
    trainable_params: int,
    epoch_summaries: list[dict],
    best_epoch: int,
    best_valid_metric: float | None,
    best_valid_loss: float | None,
    status: str,
    test_metrics: dict | None = None,
) -> None:
    write_json(
        path,
        {
            "dataset": args.dataset,
            "task": args.task,
            "seed": seed,
            "granularity": task_granularity(args.task),
            "status": status,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "epochs_ran": len(epoch_summaries),
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_htt_layers": args.num_htt_layers,
            "num_rgcn_layers": args.num_rgcn_layers,
            "dropout": args.dropout,
            "fusion_strategy": args.fusion_strategy,
            "selection_metric_name": args.selection_metric,
            "output_suffix": suffix or None,
            "num_parameters": total_params,
            "num_trainable_parameters": trainable_params,
            "concept_vocab_size": len(concept_vocab["concept_to_id"]),
            "relation_vocab_size": len(relation_vocab["relation_to_id"]),
            "num_train_examples": len(train_dataset),
            "num_valid_examples": len(valid_dataset),
            "num_test_examples": len(test_dataset),
            "best_epoch": best_epoch,
            "best_valid_metric": best_valid_metric,
            "best_valid_loss": best_valid_loss,
            "best_checkpoint_path": str(checkpoint_path),
            "epoch_summaries": epoch_summaries,
            "test_metrics": test_metrics,
        },
    )


def build_train_concept_vocab(dataset: str, task: str, task_root: Path) -> dict:
    split_instances_path = SPLITS_ROOT / dataset / task / "train.jsonl"
    aligned_ehr_path = ALIGNED_ROOT / dataset / "patients_visits_aligned.jsonl"
    pkg_nodes_path = PERSONALIZE_ROOT / dataset / task / "train_pkg_nodes.jsonl"
    concept_vocab_path = task_root / "formal_concept_vocab.json"

    concept_to_id = {"<pad>": 0, "<unk>": 1}
    visible_visits_by_patient: dict[str, set[str]] = {}
    for row in iter_jsonl(split_instances_path):
        patient_id = str(row["patient_id"])
        visible_visits_by_patient.setdefault(patient_id, set()).update(selected_visit_ids(row))

    for patient_row in iter_jsonl(aligned_ehr_path):
        patient_id = str(patient_row["patient_id"])
        visible_visits = visible_visits_by_patient.get(patient_id)
        if not visible_visits:
            continue
        for visit in patient_row["visits"]:
            if str(visit["visit_id"]) not in visible_visits:
                continue
            for concept in visit.get("aligned_concepts", []):
                add_concept(concept_to_id, concept.get("ehr_concept_id"))

    for row in iter_jsonl(pkg_nodes_path):
        add_concept(concept_to_id, row.get("concept_id"))

    payload = {"concept_to_id": concept_to_id}
    write_json(concept_vocab_path, payload)
    return payload


class LazyIntegrateDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        task: str,
        split: str,
        concept_vocab: dict,
        relation_vocab: dict,
    ):
        self.dataset = dataset
        self.task = task
        self.split = split
        self.concept_vocab = concept_vocab
        self.relation_vocab = relation_vocab

        self.integrate_spec = build_integrate_data_spec(dataset, task, split)
        self.manifest_rows = list(iter_jsonl(Path(self.integrate_spec["inputs"]["integrate_manifest"])))
        self.split_instance_map = {
            make_instance_id(row): row
            for row in iter_jsonl(Path(self.integrate_spec["inputs"]["split_instances"]))
        }
        self.patient_index = load_json(Path(self.integrate_spec["outputs"]["patient_index"]))
        self.pkg_nodes_index = load_json(Path(self.integrate_spec["outputs"]["pkg_nodes_index"]))
        self.pkg_edges_index = load_json(Path(self.integrate_spec["outputs"]["pkg_edges_index"]))

        self.aligned_ehr_path = Path(self.integrate_spec["inputs"]["aligned_ehr"])
        self.pkg_nodes_path = Path(self.integrate_spec["inputs"]["pkg_nodes"])
        self.pkg_edges_path = Path(self.integrate_spec["inputs"]["pkg_edges"])

    def __len__(self) -> int:
        return len(self.manifest_rows)

    def __getitem__(self, idx: int) -> dict:
        manifest_row = self.manifest_rows[idx]
        patient_row = load_patient_row(
            self.aligned_ehr_path,
            self.patient_index,
            manifest_row["patient_id"],
        )
        split_instance = self.split_instance_map[manifest_row["instance_id"]]
        node_segment = self.pkg_nodes_index[manifest_row["pkg_id"]]
        edge_segment = self.pkg_edges_index[manifest_row["pkg_id"]]
        pkg_nodes = load_segment_rows(self.pkg_nodes_path, node_segment["start"], node_segment["end"])
        pkg_edges = load_segment_rows(self.pkg_edges_path, edge_segment["start"], edge_segment["end"])
        sample = materialize_sample(
            manifest_row=manifest_row,
            split_instance=split_instance,
            patient_row=patient_row,
            pkg_nodes=pkg_nodes,
            pkg_edges=pkg_edges,
            relation_vocab=self.relation_vocab,
        )
        return tensorize_sample(sample, self.concept_vocab)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def build_loader_for_seed(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_integrate_batch,
        generator=generator,
    )


def collect_eval_outputs(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    labels = []
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            loss = criterion(logits, batch["labels"])
            total_loss += float(loss.item())
            total_batches += 1
            labels.extend(batch["labels"].detach().cpu().tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "num_examples": len(labels),
        "positive_rate": float(sum(labels) / max(len(labels), 1)),
    }
    unique_labels = {int(x) for x in labels}
    if len(unique_labels) >= 2:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    return metrics


def run_single_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    task_root: Path,
    relation_vocab: dict,
    concept_vocab: dict,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    base_suffix: str,
) -> dict:
    set_global_seed(seed)
    seed_suffix = combine_suffix(base_suffix, f"seed{seed}")
    progress_path = task_root / artifact_name("formal_train_progress.json", seed_suffix)
    checkpoint_path = task_root / artifact_name("best_valid_formal.pt", seed_suffix)

    train_loader = build_loader_for_seed(train_dataset, args.batch_size, shuffle=True, seed=seed)
    valid_loader = build_loader_for_seed(valid_dataset, args.batch_size, shuffle=False, seed=seed)
    test_loader = build_loader_for_seed(test_dataset, args.batch_size, shuffle=False, seed=seed)

    model = MedCPIIntegrateModel(
        concept_vocab_size=len(concept_vocab["concept_to_id"]),
        relation_vocab_size=len(relation_vocab["relation_to_id"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_htt_layers=args.num_htt_layers,
        num_rgcn_layers=args.num_rgcn_layers,
        dropout=args.dropout,
        granularity=task_granularity(args.task),
        fusion_strategy=args.fusion_strategy,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_metric = float("-inf")
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_summaries = []
    selection_metric_name = args.selection_metric

    write_progress(
        progress_path,
        seed=seed,
        args=args,
        suffix=seed_suffix,
        device=device,
        relation_vocab=relation_vocab,
        concept_vocab=concept_vocab,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        checkpoint_path=checkpoint_path,
        total_params=total_params,
        trainable_params=trainable_params,
        epoch_summaries=epoch_summaries,
        best_epoch=best_epoch,
        best_valid_metric=None,
        best_valid_loss=None,
        status="running",
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs["logits"], batch["labels"])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)
        valid_metrics = collect_eval_outputs(model, valid_loader, criterion, device)
        selection_metric = resolve_selection_metric_value(selection_metric_name, valid_metrics)
        epoch_summary = {
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_auroc": valid_metrics["auroc"],
            "valid_auprc": valid_metrics["auprc"],
            "selection_metric_name": selection_metric_name,
            "selection_metric": selection_metric,
        }
        epoch_summaries.append(epoch_summary)

        if selection_metric > best_valid_metric:
            best_valid_metric = selection_metric
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "concept_vocab": concept_vocab,
                    "relation_vocab_size": len(relation_vocab["relation_to_id"]),
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_heads": args.num_heads,
                    "num_htt_layers": args.num_htt_layers,
                    "num_rgcn_layers": args.num_rgcn_layers,
                    "dropout": args.dropout,
                    "epoch": epoch + 1,
                    "best_valid_metric": best_valid_metric,
                    "best_valid_loss": best_valid_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        write_progress(
            progress_path,
            seed=seed,
            args=args,
            suffix=seed_suffix,
            device=device,
            relation_vocab=relation_vocab,
            concept_vocab=concept_vocab,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            checkpoint_path=checkpoint_path,
            total_params=total_params,
            trainable_params=trainable_params,
            epoch_summaries=epoch_summaries,
            best_epoch=best_epoch,
            best_valid_metric=best_valid_metric,
            best_valid_loss=best_valid_loss,
            status="running",
        )
        print(json.dumps(epoch_summary, sort_keys=True), flush=True)

        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = collect_eval_outputs(model, test_loader, criterion, device)

    write_progress(
        progress_path,
        seed=seed,
        args=args,
        suffix=seed_suffix,
        device=device,
        relation_vocab=relation_vocab,
        concept_vocab=concept_vocab,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        checkpoint_path=checkpoint_path,
        total_params=total_params,
        trainable_params=trainable_params,
        epoch_summaries=epoch_summaries,
        best_epoch=best_epoch,
        best_valid_metric=best_valid_metric,
        best_valid_loss=best_valid_loss,
        status="completed",
        test_metrics=test_metrics,
    )

    seed_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "seed": seed,
        "granularity": task_granularity(args.task),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "fusion_strategy": args.fusion_strategy,
        "selection_metric_name": selection_metric_name,
        "output_suffix": seed_suffix or None,
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_ran": len(epoch_summaries),
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_htt_layers": args.num_htt_layers,
        "num_rgcn_layers": args.num_rgcn_layers,
        "dropout": args.dropout,
        "num_parameters": seed_results[0]["num_parameters"],
        "num_trainable_parameters": seed_results[0]["num_trainable_parameters"],
        "concept_vocab_size": len(concept_vocab["concept_to_id"]),
        "relation_vocab_size": len(relation_vocab["relation_to_id"]),
        "num_train_examples": len(train_dataset),
        "num_valid_examples": len(valid_dataset),
        "num_test_examples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_valid_metric": best_valid_metric,
        "best_valid_loss": best_valid_loss,
        "best_checkpoint_path": str(checkpoint_path),
        "epoch_summaries": epoch_summaries,
        "test_metrics": test_metrics,
        "progress_path": str(progress_path),
    }
    return seed_summary


def main() -> None:
    args = parse_args()
    task_root = INTEGRATE_ROOT / args.dataset / args.task
    base_suffix = output_suffix(args)
    seed_values = parse_seed_values(args)

    for split in SPLITS:
        spec = build_integrate_data_spec(args.dataset, args.task, split)
        required = [
            Path(spec["inputs"]["integrate_manifest"]),
            Path(spec["outputs"]["patient_index"]),
            Path(spec["outputs"]["pkg_nodes_index"]),
            Path(spec["outputs"]["pkg_edges_index"]),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required full-scale integrate artifacts for split {split}:\n- " + "\n- ".join(missing))

    relation_vocab = build_relation_vocab(CONSTRUCT_ROOT / "shared" / args.task / "relation_schema.json")
    concept_vocab = build_train_concept_vocab(args.dataset, args.task, task_root)

    train_dataset = LazyIntegrateDataset(args.dataset, args.task, "train", concept_vocab, relation_vocab)
    valid_dataset = LazyIntegrateDataset(args.dataset, args.task, "valid", concept_vocab, relation_vocab)
    test_dataset = LazyIntegrateDataset(args.dataset, args.task, "test", concept_vocab, relation_vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = [
        run_single_seed(
            args=args,
            seed=seed,
            device=device,
            task_root=task_root,
            relation_vocab=relation_vocab,
            concept_vocab=concept_vocab,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            base_suffix=base_suffix,
        )
        for seed in seed_values
    ]

    aggregate_summary = {
        "dataset": args.dataset,
        "task": args.task,
        "granularity": task_granularity(args.task),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "fusion_strategy": args.fusion_strategy,
        "selection_metric_name": args.selection_metric,
        "output_suffix": base_suffix or None,
        "seeds": seed_values,
        "num_seed_runs": len(seed_values),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_htt_layers": args.num_htt_layers,
        "num_rgcn_layers": args.num_rgcn_layers,
        "dropout": args.dropout,
        "concept_vocab_size": len(concept_vocab["concept_to_id"]),
        "relation_vocab_size": len(relation_vocab["relation_to_id"]),
        "num_train_examples": len(train_dataset),
        "num_valid_examples": len(valid_dataset),
        "num_test_examples": len(test_dataset),
        "aggregate_test_metrics": {
            "auroc": summarize_numeric([item["test_metrics"]["auroc"] for item in seed_results]),
            "auprc": summarize_numeric([item["test_metrics"]["auprc"] for item in seed_results]),
            "loss": summarize_numeric([item["test_metrics"]["loss"] for item in seed_results]),
            "positive_rate": summarize_numeric([item["test_metrics"]["positive_rate"] for item in seed_results]),
        },
        "aggregate_best_valid_metric": summarize_numeric([item["best_valid_metric"] for item in seed_results]),
        "aggregate_best_epoch": summarize_numeric([item["best_epoch"] for item in seed_results]),
        "seed_results": seed_results,
    }

    if args.emit_summary:
        write_json(task_root / artifact_name("formal_train_eval_summary.json", base_suffix), aggregate_summary)

    print(
        json.dumps(
            aggregate_summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
