#!/usr/bin/env python3
"""Verify Construct outputs against the expected split-first protocol."""

import argparse
import json
from pathlib import Path
from typing import List

from path_roots import CONSTRUCT_ROOT

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "both"], default="both")
    return parser.parse_args()


def selected_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "both":
        return ["mimic3", "mimic4"]
    return [dataset_arg]


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def count_jsonl_rows(path: Path) -> int:
    with path.open("r") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args.dataset)

    summary = {"shared_schemas": {}, "datasets": {}}
    for task in TASKS:
        schema_path = CONSTRUCT_ROOT / "shared" / task / "relation_schema.json"
        payload = load_json(schema_path)
        if payload.get("schema_scope") not in {
            "task_specific",
            "task_specific_shared_across_datasets",
            "task_shared",
        }:
            raise ValueError("Unexpected schema scope in {0}".format(schema_path))
        kept = [row for row in payload.get("relations", []) if row.get("keep") is True]
        if not kept:
            raise ValueError("Shared schema keeps no relations: {0}".format(schema_path))
        summary["shared_schemas"][task] = len(kept)

    for dataset in datasets:
        dataset_summary = {}
        for task in TASKS:
            task_dir = CONSTRUCT_ROOT / dataset / task
            stats_path = task_dir / "construct_stats.json"
            nodes_path = task_dir / "concept_mkg_nodes.jsonl"
            edges_path = task_dir / "concept_mkg_edges.jsonl"
            anchor_path = task_dir / "train_anchor_concepts.json"

            stats = load_json(stats_path)
            anchor_meta = load_json(anchor_path)
            num_nodes = count_jsonl_rows(nodes_path)
            num_edges = count_jsonl_rows(edges_path)

            if stats.get("num_nodes") != num_nodes:
                raise ValueError("Node count mismatch for {0}/{1}".format(dataset, task))
            if stats.get("num_edges") != num_edges:
                raise ValueError("Edge count mismatch for {0}/{1}".format(dataset, task))
            if stats.get("schema_resolution") != "shared_task":
                raise ValueError("Expected shared_task schema for {0}/{1}".format(dataset, task))
            if stats.get("construct_protocol") != "split_then_construct_then_evaluate":
                raise ValueError("Unexpected construct protocol for {0}/{1}".format(dataset, task))
            if anchor_meta.get("train_anchor_scope") != "training_split_only":
                raise ValueError("Unexpected anchor scope for {0}/{1}".format(dataset, task))
            if stats.get("num_edges", 0) <= 0:
                raise ValueError("Construct graph is empty for {0}/{1}".format(dataset, task))

            dataset_summary[task] = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_kept_raw_relations": stats.get("num_kept_raw_relations"),
            }
        summary["datasets"][dataset] = dataset_summary

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
