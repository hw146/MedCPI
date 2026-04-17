#!/usr/bin/env python3
"""Tensorize Integrate samples into batchable EHR and PKG inputs."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from path_roots import INTEGRATE_ROOT

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]
NODE_KIND_VOCAB = {"patient": 0, "visit": 1, "concept": 2}
SOURCE_TYPE_VOCAB = {"<pad>": 0, "diagnosis": 1, "procedure": 2, "medication": 3}


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--emit-spec", action="store_true")
    parser.add_argument("--tensorize-preview", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_spec(dataset: str, task: str, split: str) -> dict:
    task_root = INTEGRATE_ROOT / dataset / task
    return {
        "stage": "integrate_dataset",
        "dataset": dataset,
        "task": task,
        "split": split,
        "inputs": {
            "sample_preview": str(task_root / f"{split}_integrate_samples.preview.jsonl"),
            "relation_vocab": str(task_root / "relation_vocab.json"),
        },
        "outputs": {
            "task_root": str(task_root),
            "spec_path": str(task_root / f"{split}_tensor_data_spec.json"),
            "stats_path": str(task_root / f"{split}_tensor_data_stats.json"),
            "batch_preview": str(task_root / f"{split}_tensor_batch.preview.json"),
            "concept_vocab": str(task_root / f"{split}_concept_vocab.json"),
        },
    }


def ensure_inputs_exist(spec: dict) -> None:
    missing = [path for path in spec["inputs"].values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def build_concept_vocab(samples: List[dict]) -> dict:
    concept_to_id = {"<pad>": 0, "<unk>": 1}
    for sample in samples:
        for visit in sample["ehr_sequence"]:
            for concept in visit["concepts"]:
                concept_id = concept["concept_id"]
                if concept_id not in concept_to_id:
                    concept_to_id[concept_id] = len(concept_to_id)
        for node in sample["pkg_nodes"]:
            concept_id = node.get("concept_id")
            if concept_id and concept_id not in concept_to_id:
                concept_to_id[concept_id] = len(concept_to_id)
    return {"concept_to_id": concept_to_id}


def tensorize_sample(sample: dict, concept_vocab: dict) -> dict:
    concept_to_id = concept_vocab["concept_to_id"]

    ehr_concept_ids = []
    ehr_source_type_ids = []
    ehr_time_delta_days = []
    prev_admittime = None
    for visit in sample["ehr_sequence"]:
        visit_concept_ids = []
        visit_source_ids = []
        for concept in visit["concepts"]:
            visit_concept_ids.append(concept_to_id.get(concept["concept_id"], concept_to_id["<unk>"]))
            visit_source_ids.append(SOURCE_TYPE_VOCAB.get(concept["source_type"], 0))
        ehr_concept_ids.append(visit_concept_ids)
        ehr_source_type_ids.append(visit_source_ids)
        current_admittime = parse_timestamp(visit["admittime"])
        if prev_admittime is None:
            ehr_time_delta_days.append(0.0)
        else:
            ehr_time_delta_days.append((current_admittime - prev_admittime).total_seconds() / 86400.0)
        prev_admittime = current_admittime

    pkg_node_concept_ids = []
    pkg_node_kind_ids = []
    for node in sample["pkg_nodes"]:
        concept_id = node.get("concept_id")
        if concept_id:
            pkg_node_concept_ids.append(concept_to_id.get(concept_id, concept_to_id["<unk>"]))
        else:
            pkg_node_concept_ids.append(concept_to_id["<pad>"])
        pkg_node_kind_ids.append(NODE_KIND_VOCAB[node["node_kind"]])

    edge_index = [[], []]
    edge_type = []
    for edge in sample["pkg_edges"]:
        edge_index[0].append(edge["src_index"])
        edge_index[1].append(edge["dst_index"])
        edge_type.append(edge["relation_id"])

    return {
        "instance_id": sample["instance_id"],
        "label": float(sample["label"]),
        "ehr_concept_ids": ehr_concept_ids,
        "ehr_source_type_ids": ehr_source_type_ids,
        "ehr_time_delta_days": ehr_time_delta_days,
        "pkg_node_concept_ids": pkg_node_concept_ids,
        "pkg_node_kind_ids": pkg_node_kind_ids,
        "pkg_edge_index": edge_index,
        "pkg_edge_type": edge_type,
        "num_visits": len(sample["ehr_sequence"]),
        "num_nodes": len(sample["pkg_nodes"]),
        "num_edges": len(sample["pkg_edges"]),
    }


class IntegratePreviewDataset(Dataset):
    def __init__(self, samples: List[dict], concept_vocab: dict):
        self.samples = samples
        self.concept_vocab = concept_vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return tensorize_sample(self.samples[idx], self.concept_vocab)


def collate_integrate_batch(batch: List[dict]) -> dict:
    batch_size = len(batch)
    max_visits = max(item["num_visits"] for item in batch)
    max_visit_concepts = max(
        max((len(visit) for visit in item["ehr_concept_ids"]), default=0)
        for item in batch
    )
    max_nodes = max(item["num_nodes"] for item in batch)

    ehr_concept_ids = torch.zeros((batch_size, max_visits, max_visit_concepts), dtype=torch.long)
    ehr_source_type_ids = torch.zeros((batch_size, max_visits, max_visit_concepts), dtype=torch.long)
    ehr_concept_mask = torch.zeros((batch_size, max_visits, max_visit_concepts), dtype=torch.bool)
    visit_mask = torch.zeros((batch_size, max_visits), dtype=torch.bool)
    ehr_time_delta_days = torch.zeros((batch_size, max_visits), dtype=torch.float32)

    pkg_node_concept_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    pkg_node_kind_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    pkg_node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)

    labels = torch.zeros((batch_size,), dtype=torch.float32)
    instance_ids = []
    edge_index_pieces = []
    edge_type_pieces = []
    node_batch_index = []
    node_offset = 0

    for batch_idx, item in enumerate(batch):
        instance_ids.append(item["instance_id"])
        labels[batch_idx] = item["label"]

        for visit_idx, visit_concepts in enumerate(item["ehr_concept_ids"]):
            visit_mask[batch_idx, visit_idx] = True
            ehr_time_delta_days[batch_idx, visit_idx] = item["ehr_time_delta_days"][visit_idx]
            for concept_idx, concept_id in enumerate(visit_concepts):
                ehr_concept_ids[batch_idx, visit_idx, concept_idx] = concept_id
                ehr_source_type_ids[batch_idx, visit_idx, concept_idx] = item["ehr_source_type_ids"][visit_idx][
                    concept_idx
                ]
                ehr_concept_mask[batch_idx, visit_idx, concept_idx] = True

        for node_idx, concept_id in enumerate(item["pkg_node_concept_ids"]):
            pkg_node_concept_ids[batch_idx, node_idx] = concept_id
            pkg_node_kind_ids[batch_idx, node_idx] = item["pkg_node_kind_ids"][node_idx]
            pkg_node_mask[batch_idx, node_idx] = True
            node_batch_index.append(batch_idx)

        if item["pkg_edge_type"]:
            edge_index = torch.tensor(item["pkg_edge_index"], dtype=torch.long)
            edge_index_pieces.append(edge_index + node_offset)
            edge_type_pieces.append(torch.tensor(item["pkg_edge_type"], dtype=torch.long))
        node_offset += item["num_nodes"]

    if edge_index_pieces:
        batch_edge_index = torch.cat(edge_index_pieces, dim=1)
        batch_edge_type = torch.cat(edge_type_pieces, dim=0)
    else:
        batch_edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch_edge_type = torch.zeros((0,), dtype=torch.long)

    return {
        "instance_ids": instance_ids,
        "labels": labels,
        "ehr_concept_ids": ehr_concept_ids,
        "ehr_source_type_ids": ehr_source_type_ids,
        "ehr_concept_mask": ehr_concept_mask,
        "visit_mask": visit_mask,
        "ehr_time_delta_days": ehr_time_delta_days,
        "pkg_node_concept_ids": pkg_node_concept_ids,
        "pkg_node_kind_ids": pkg_node_kind_ids,
        "pkg_node_mask": pkg_node_mask,
        "pkg_node_batch_index": torch.tensor(node_batch_index, dtype=torch.long),
        "pkg_edge_index": batch_edge_index,
        "pkg_edge_type": batch_edge_type,
    }


def batch_summary(batch: dict, concept_vocab: dict, relation_vocab: dict) -> dict:
    return {
        "batch_size": len(batch["instance_ids"]),
        "concept_vocab_size": len(concept_vocab["concept_to_id"]),
        "relation_vocab_size": len(relation_vocab["relation_to_id"]),
        "instance_ids": batch["instance_ids"],
        "shape_ehr_concept_ids": list(batch["ehr_concept_ids"].shape),
        "shape_ehr_source_type_ids": list(batch["ehr_source_type_ids"].shape),
        "shape_ehr_concept_mask": list(batch["ehr_concept_mask"].shape),
        "shape_visit_mask": list(batch["visit_mask"].shape),
        "shape_ehr_time_delta_days": list(batch["ehr_time_delta_days"].shape),
        "shape_pkg_node_concept_ids": list(batch["pkg_node_concept_ids"].shape),
        "shape_pkg_node_kind_ids": list(batch["pkg_node_kind_ids"].shape),
        "shape_pkg_node_mask": list(batch["pkg_node_mask"].shape),
        "shape_pkg_edge_index": list(batch["pkg_edge_index"].shape),
        "shape_pkg_edge_type": list(batch["pkg_edge_type"].shape),
        "num_true_visit_mask": int(batch["visit_mask"].sum().item()),
        "num_true_node_mask": int(batch["pkg_node_mask"].sum().item()),
    }


def main() -> None:
    args = parse_args()
    spec = build_spec(args.dataset, args.task, args.split)
    ensure_inputs_exist(spec)

    samples = list(iter_jsonl(Path(spec["inputs"]["sample_preview"])))[: args.limit]
    relation_vocab = json.loads(Path(spec["inputs"]["relation_vocab"]).read_text())
    concept_vocab = build_concept_vocab(samples)

    if args.emit_spec:
        write_json(Path(spec["outputs"]["spec_path"]), spec)

    if args.tensorize_preview:
        dataset = IntegratePreviewDataset(samples=samples, concept_vocab=concept_vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=min(args.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=collate_integrate_batch,
        )
        batch = next(iter(dataloader))
        write_json(Path(spec["outputs"]["concept_vocab"]), concept_vocab)
        write_json(
            Path(spec["outputs"]["batch_preview"]),
            batch_summary(batch=batch, concept_vocab=concept_vocab, relation_vocab=relation_vocab),
        )
        write_json(
            Path(spec["outputs"]["stats_path"]),
            {
                "num_samples": len(samples),
                "concept_vocab_size": len(concept_vocab["concept_to_id"]),
                "relation_vocab_size": len(relation_vocab["relation_to_id"]),
            },
        )

    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
