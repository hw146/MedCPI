#!/usr/bin/env python3
"""Prepare Integrate-stage input manifests from task splits and PKG outputs."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from path_roots import INTEGRATE_ROOT, PERSONALIZE_ROOT, SPLITS_ROOT
from pkg_utils import iter_jsonl, make_instance_id, write_json, write_jsonl

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]
FUSION_STRATEGIES = ["cross_attention", "concat_mlp", "gated", "film"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS + ["all"], default="all")
    parser.add_argument("--emit-spec", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fusion-strategy", choices=FUSION_STRATEGIES, default="cross_attention")
    parser.add_argument("--integrate-root", type=Path, default=INTEGRATE_ROOT)
    parser.add_argument("--personalize-root", type=Path, default=PERSONALIZE_ROOT)
    return parser.parse_args()


def selected_splits(split_arg: str) -> List[str]:
    if split_arg == "all":
        return list(SPLITS)
    return [split_arg]


def required_input_paths(personalize_root: Path, dataset: str, task: str, split_names: List[str]) -> Dict[str, Path]:
    personalize_dir = personalize_root / dataset / task
    paths = {
        "pkg_spec": personalize_dir / "pkg_output_spec.json",
        "pkg_stats": personalize_dir / "pkg_stats.json",
    }
    for split_name in split_names:
        paths[f"split_instances_{split_name}"] = SPLITS_ROOT / dataset / task / f"{split_name}.jsonl"
        paths[f"pkg_metadata_{split_name}"] = personalize_dir / f"{split_name}_pkg_metadata.jsonl"
        paths[f"pkg_nodes_{split_name}"] = personalize_dir / f"{split_name}_pkg_nodes.jsonl"
        paths[f"pkg_edges_{split_name}"] = personalize_dir / f"{split_name}_pkg_edges.jsonl"
    return paths


def ensure_inputs_exist(paths: Dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def output_layout(integrate_root: Path, dataset: str, task: str, split_names: List[str]) -> Dict[str, object]:
    task_root = integrate_root / dataset / task
    split_outputs = {}
    for split_name in split_names:
        split_outputs[split_name] = {
            "manifest_jsonl": str(task_root / f"{split_name}_integrate_manifest.jsonl"),
        }
    return {
        "task_root": str(task_root),
        "spec_path": str(task_root / "integrate_input_spec.json"),
        "stats_path": str(task_root / "integrate_stats.json"),
        "splits": split_outputs,
    }


def build_spec(args: argparse.Namespace) -> dict:
    split_names = selected_splits(args.split)
    inputs = required_input_paths(args.personalize_root, args.dataset, args.task, split_names)
    ensure_inputs_exist(inputs)
    sample_instance = next(iter_jsonl(inputs[f"split_instances_{split_names[0]}"]))
    return {
        "stage": "integrate",
        "dataset": args.dataset,
        "task": args.task,
        "granularity": sample_instance["granularity"],
        "protocol": {
            "paper_stage": "Integrate",
            "ehr_context_source": "task_split_instances",
            "pkg_source": "personalize_outputs",
            "fusion": "cross_attention_over_pkg_nodes"
            if args.fusion_strategy == "cross_attention"
            else args.fusion_strategy,
            "fusion_strategy": args.fusion_strategy,
            "prediction_time_leakage": "disallowed",
        },
        "inputs": {key: str(value) for key, value in inputs.items()},
        "outputs": output_layout(args.integrate_root, args.dataset, args.task, split_names),
        "field_schema": {
            "manifest_jsonl": [
                "instance_id",
                "pkg_id",
                "dataset",
                "task",
                "split",
                "granularity",
                "patient_id",
                "prediction_time",
                "label",
                "pkg_num_nodes",
                "pkg_num_edges",
                "pkg_num_ehr_edges",
                "pkg_num_concept_1hop_edges",
                "pkg_num_bridge_path_edges",
                "pkg_construction_state",
            ]
        },
    }


def maybe_write_spec(spec: dict, integrate_root: Path, dataset: str, task: str) -> None:
    task_root = integrate_root / dataset / task
    task_root.mkdir(parents=True, exist_ok=True)
    write_json(task_root / "integrate_input_spec.json", spec)


def build_split_manifest(split_name: str, spec: dict, limit: int = None) -> dict:
    instance_path = Path(spec["inputs"][f"split_instances_{split_name}"])
    metadata_path = Path(spec["inputs"][f"pkg_metadata_{split_name}"])
    instances = list(iter_jsonl(instance_path))
    metadata_rows = list(iter_jsonl(metadata_path))
    if limit is not None:
        instances = instances[:limit]
        metadata_rows = metadata_rows[:limit]

    metadata_by_instance = {row["instance_id"]: row for row in metadata_rows}
    manifest_rows = []
    stats = Counter()
    for instance_row in instances:
        instance_id = make_instance_id(instance_row)
        pkg_meta = metadata_by_instance.get(instance_id)
        if pkg_meta is None:
            raise ValueError(
                "Missing PKG metadata for instance_id {0} in split {1}".format(instance_id, split_name)
            )
        manifest_rows.append(
            {
                "instance_id": instance_id,
                "pkg_id": pkg_meta["pkg_id"],
                "dataset": instance_row["dataset"],
                "task": instance_row["task"],
                "split": split_name,
                "granularity": instance_row["granularity"],
                "patient_id": str(instance_row["patient_id"]),
                "prediction_time": instance_row["prediction_time"],
                "label": instance_row.get("label"),
                "pkg_num_nodes": pkg_meta["num_nodes"],
                "pkg_num_edges": pkg_meta["num_edges"],
                "pkg_num_ehr_edges": pkg_meta["num_ehr_edges"],
                "pkg_num_concept_1hop_edges": pkg_meta["num_concept_1hop_edges"],
                "pkg_num_bridge_path_edges": pkg_meta["num_bridge_path_edges"],
                "pkg_construction_state": pkg_meta["construction_state"],
            }
        )
        stats["instances"] += 1
        stats["pkg_nodes"] += pkg_meta["num_nodes"]
        stats["pkg_edges"] += pkg_meta["num_edges"]
        stats["pkg_bridge_edges"] += pkg_meta["num_bridge_path_edges"]

    output_path = Path(spec["outputs"]["splits"][split_name]["manifest_jsonl"])
    write_jsonl(output_path, manifest_rows)
    return {
        "split": split_name,
        "num_instances": stats["instances"],
        "pkg_node_rows_summed": stats["pkg_nodes"],
        "pkg_edge_rows_summed": stats["pkg_edges"],
        "pkg_bridge_edge_rows_summed": stats["pkg_bridge_edges"],
    }


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    if args.emit_spec:
        maybe_write_spec(spec, args.integrate_root, args.dataset, args.task)
    if args.build_manifest:
        maybe_write_spec(spec, args.integrate_root, args.dataset, args.task)
        split_names = list(spec["outputs"]["splits"].keys())
        split_stats = [build_split_manifest(split_name, spec, args.limit) for split_name in split_names]
        write_json(
            Path(spec["outputs"]["stats_path"]),
            {
                "stage": "integrate",
                "dataset": args.dataset,
                "task": args.task,
                "granularity": spec["granularity"],
                "split_stats": split_stats,
            },
        )
    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
