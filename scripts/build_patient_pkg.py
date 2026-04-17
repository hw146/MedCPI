#!/usr/bin/env python3
"""Build the Personalize-stage EHR subgraph skeleton for patient PKGs."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from path_roots import ALIGNED_ROOT, CONSTRUCT_ROOT, PERSONALIZE_ROOT, SPLITS_ROOT, UMLS_ROOT
from pkg_utils import (
    attach_bridge_paths,
    attach_concept_1hop,
    anchor_scope_for_granularity,
    build_patient_index,
    collect_ehr_subgraph,
    collect_instance_anchor_concepts,
    iter_jsonl,
    load_anchor_bridge_index,
    load_concept_mkg_graph,
    load_patient_row,
    write_json,
)
MRREL_PATH = UMLS_ROOT / "MRREL.RRF"
TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS + ["all"], default="all")
    parser.add_argument(
        "--personalize-root",
        type=Path,
        default=PERSONALIZE_ROOT,
        help="Root directory for Personalize outputs.",
    )
    parser.add_argument(
        "--emit-spec",
        action="store_true",
        help="Write the resolved PKG output spec to disk.",
    )
    parser.add_argument(
        "--build-ehr-skeleton",
        action="store_true",
        help="Build the patient EHR-subgraph skeleton for the selected split(s).",
    )
    parser.add_argument(
        "--attach-concept-1hop",
        action="store_true",
        help="Attach 1-hop Concept MKG neighborhoods on top of the EHR skeleton.",
    )
    parser.add_argument(
        "--attach-bridge-paths",
        action="store_true",
        help="Attach 2-hop bridge paths from the task-normalized MKG.",
    )
    parser.add_argument(
        "--mrrel-path",
        type=Path,
        default=MRREL_PATH,
        help="MRREL path used to materialize the task-normalized MKG for path search.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of instances to process per run.",
    )
    return parser.parse_args()


def selected_splits(split_arg: str) -> List[str]:
    if split_arg == "all":
        return list(SPLITS)
    return [split_arg]


def iter_limited_jsonl(path: Path, limit: Optional[int]):
    for idx, row in enumerate(iter_jsonl(path)):
        if limit is not None and idx >= limit:
            break
        yield row


def required_input_paths(dataset: str, task: str) -> Dict[str, Path]:
    task_dir = CONSTRUCT_ROOT / dataset / task
    return {
        "aligned_ehr": ALIGNED_ROOT / dataset / "patients_visits_aligned.jsonl",
        "construct_stats": task_dir / "construct_stats.json",
        "concept_mkg_nodes": task_dir / "concept_mkg_nodes.jsonl",
        "concept_mkg_edges": task_dir / "concept_mkg_edges.jsonl",
        "shared_schema": CONSTRUCT_ROOT / "shared" / task / "relation_schema.json",
    }


def split_input_paths(dataset: str, task: str, split_names: List[str]) -> Dict[str, Path]:
    task_root = SPLITS_ROOT / dataset / task
    return {split_name: task_root / "{0}.jsonl".format(split_name) for split_name in split_names}


def ensure_inputs_exist(paths: Dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def load_first_jsonl_row(path: Path) -> dict:
    for row in iter_jsonl(path):
        return row
    raise ValueError("JSONL file is empty: {0}".format(path))


def output_layout(personalize_root: Path, dataset: str, task: str, split_names: List[str]) -> Dict[str, object]:
    task_root = personalize_root / dataset / task
    split_outputs = {}
    for split_name in split_names:
        split_outputs[split_name] = {
            "pkg_metadata_jsonl": str(task_root / "{0}_pkg_metadata.jsonl".format(split_name)),
            "pkg_nodes_jsonl": str(task_root / "{0}_pkg_nodes.jsonl".format(split_name)),
            "pkg_edges_jsonl": str(task_root / "{0}_pkg_edges.jsonl".format(split_name)),
        }
    return {
        "task_root": str(task_root),
        "spec_path": str(task_root / "pkg_output_spec.json"),
        "stats_path": str(task_root / "pkg_stats.json"),
        "splits": split_outputs,
    }


def pkg_field_schema(granularity: str) -> Dict[str, List[str]]:
    metadata_fields = [
        "pkg_id",
        "instance_id",
        "dataset",
        "task",
        "split",
        "granularity",
        "patient_id",
        "prediction_time",
        "anchor_concepts",
        "num_anchor_concepts",
        "num_nodes",
        "num_edges",
        "num_ehr_edges",
        "num_concept_1hop_edges",
        "num_bridge_path_edges",
        "num_bridge_paths",
        "max_path_length",
        "k_nbr",
        "k_path",
        "anchor_scope",
        "construction_state",
    ]
    if granularity == "visit":
        metadata_fields.extend(["target_visit_id", "target_visit_index", "context_visit_ids"])
    elif granularity == "patient":
        metadata_fields.extend(["observation_start", "observation_end", "observation_visit_ids"])

    return {
        "pkg_metadata_jsonl": metadata_fields,
        "pkg_nodes_jsonl": [
            "pkg_id",
            "instance_id",
            "patient_id",
            "node_id",
            "node_kind",
            "concept_id",
            "visit_id",
            "source",
            "construction_state",
        ],
        "pkg_edges_jsonl": [
            "pkg_id",
            "instance_id",
            "patient_id",
            "source_id",
            "relation_type",
            "target_id",
            "edge_source",
            "path_role",
            "construction_state",
        ],
    }


def build_spec(args: argparse.Namespace) -> dict:
    split_names = selected_splits(args.split)
    base_inputs = required_input_paths(args.dataset, args.task)
    split_inputs = split_input_paths(args.dataset, args.task, split_names)
    ensure_inputs_exist({**base_inputs, **split_inputs})

    sample_row = load_first_jsonl_row(split_inputs[split_names[0]])
    granularity = sample_row["granularity"]

    return {
        "stage": "personalize",
        "dataset": args.dataset,
        "task": args.task,
        "granularity": granularity,
        "protocol": {
            "paper_stage": "Personalize",
            "pkg_source": "ehr_subgraph_plus_concept_mkg_plus_task_normalized_mkg_paths",
            "max_path_length": 2,
            "k_nbr": 30,
            "k_path": 2,
            "path_search_scope": "cooccurring_anchor_pairs_only",
            "deduplication": "enabled",
            "prediction_time_leakage": "disallowed",
            "anchor_scope": anchor_scope_for_granularity(granularity),
        },
        "inputs": {
            "aligned_ehr": str(base_inputs["aligned_ehr"]),
            "shared_schema": str(base_inputs["shared_schema"]),
            "construct_stats": str(base_inputs["construct_stats"]),
            "concept_mkg_nodes": str(base_inputs["concept_mkg_nodes"]),
            "concept_mkg_edges": str(base_inputs["concept_mkg_edges"]),
            "split_instances": {key: str(value) for key, value in split_inputs.items()},
        },
        "outputs": output_layout(args.personalize_root, args.dataset, args.task, split_names),
        "field_schema": pkg_field_schema(granularity),
    }


def maybe_write_spec(spec: dict, personalize_root: Path, dataset: str, task: str) -> None:
    task_root = personalize_root / dataset / task
    task_root.mkdir(parents=True, exist_ok=True)
    spec_path = task_root / "pkg_output_spec.json"
    write_json(spec_path, spec)


def process_split(
    spec: dict,
    split_name: str,
    split_path: Path,
    aligned_path: Path,
    concept_mkg_graph: dict,
    mrrel_path: Path,
    schema_path: Path,
    attach_concept_1hop_flag: bool,
    attach_bridge_paths_flag: bool,
    limit: int,
) -> dict:
    patient_index = build_patient_index(aligned_path)

    current_patient_id = None
    current_patient_row = None

    def resolve_patient_row(patient_id: str) -> dict:
        nonlocal current_patient_id, current_patient_row
        if patient_id != current_patient_id:
            current_patient_id = patient_id
            current_patient_row = load_patient_row(aligned_path, patient_index, patient_id)
        return current_patient_row

    bridge_index = None
    if attach_bridge_paths_flag:
        union_anchor_concepts = sorted(
            {
                concept_id
                for instance_row in iter_limited_jsonl(split_path, limit)
                for concept_id in collect_instance_anchor_concepts(
                    instance_row, resolve_patient_row(str(instance_row["patient_id"]))
                )
            }
        )
        bridge_index = load_anchor_bridge_index(
            mrrel_path=mrrel_path,
            schema_path=schema_path,
            anchor_concepts=union_anchor_concepts,
        )

    stats = Counter()
    split_outputs = spec["outputs"]["splits"][split_name]
    metadata_path = Path(split_outputs["pkg_metadata_jsonl"])
    nodes_path = Path(split_outputs["pkg_nodes_jsonl"])
    edges_path = Path(split_outputs["pkg_edges_jsonl"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w") as metadata_handle, nodes_path.open("w") as nodes_handle, edges_path.open("w") as edges_handle:
        for instance_row in iter_limited_jsonl(split_path, limit):
            patient_row = resolve_patient_row(str(instance_row["patient_id"]))
            metadata, nodes, edges = collect_ehr_subgraph(instance_row, patient_row, split_name)
            if attach_concept_1hop_flag:
                metadata, nodes, edges = attach_concept_1hop(
                    metadata=metadata,
                    node_rows=nodes,
                    edge_rows=edges,
                    concept_mkg_graph=concept_mkg_graph,
                    k_nbr=spec["protocol"]["k_nbr"],
                )
            if attach_bridge_paths_flag:
                metadata, nodes, edges = attach_bridge_paths(
                    metadata=metadata,
                    node_rows=nodes,
                    edge_rows=edges,
                    instance_row=instance_row,
                    patient_row=patient_row,
                    bridge_index=bridge_index,
                    k_path=spec["protocol"]["k_path"],
                )

            metadata_handle.write(json.dumps(metadata) + "\n")
            for row in nodes:
                nodes_handle.write(json.dumps(row) + "\n")
            for row in edges:
                edges_handle.write(json.dumps(row) + "\n")

            stats["instances"] += 1
            stats["nodes"] += metadata["num_nodes"]
            stats["edges"] += metadata["num_edges"]
            stats["anchor_concepts"] += metadata["num_anchor_concepts"]
    return {
        "split": split_name,
        "num_instances": stats["instances"],
        "num_pkg_nodes": stats["nodes"],
        "num_pkg_edges": stats["edges"],
        "num_anchor_concepts_total": stats["anchor_concepts"],
        "construction_state": (
            "ehr_plus_concept_1hop_plus_bridge_paths"
            if attach_bridge_paths_flag
            else "ehr_plus_concept_1hop"
            if attach_concept_1hop_flag
            else "ehr_subgraph_only"
        ),
    }


def build_ehr_skeleton_outputs(args: argparse.Namespace, spec: dict) -> None:
    split_names = list(spec["inputs"]["split_instances"].keys())
    concept_mkg_graph = load_concept_mkg_graph(
        nodes_path=Path(spec["inputs"]["concept_mkg_nodes"]),
        edges_path=Path(spec["inputs"]["concept_mkg_edges"]),
    )
    split_stats = []
    for split_name in split_names:
        split_stats.append(
            process_split(
                spec=spec,
                split_name=split_name,
                split_path=Path(spec["inputs"]["split_instances"][split_name]),
                aligned_path=Path(spec["inputs"]["aligned_ehr"]),
                concept_mkg_graph=concept_mkg_graph,
                mrrel_path=args.mrrel_path,
                schema_path=Path(spec["inputs"]["shared_schema"]),
                attach_concept_1hop_flag=args.attach_concept_1hop or args.attach_bridge_paths,
                attach_bridge_paths_flag=args.attach_bridge_paths,
                limit=args.limit,
            )
        )
    write_json(
        Path(spec["outputs"]["stats_path"]),
        {
            "stage": "personalize",
            "dataset": args.dataset,
            "task": args.task,
            "granularity": spec["granularity"],
            "construction_state": (
                "ehr_plus_concept_1hop_plus_bridge_paths"
                if args.attach_bridge_paths
                else "ehr_plus_concept_1hop"
                if args.attach_concept_1hop
                else "ehr_subgraph_only"
            ),
            "split_stats": split_stats,
        },
    )


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    if args.emit_spec:
        maybe_write_spec(spec, args.personalize_root, args.dataset, args.task)
    should_build = args.build_ehr_skeleton or args.attach_concept_1hop or args.attach_bridge_paths
    if should_build:
        maybe_write_spec(spec, args.personalize_root, args.dataset, args.task)
        build_ehr_skeleton_outputs(args, spec)
    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
