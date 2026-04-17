#!/usr/bin/env python3
"""Build Integrate-stage data indexes and materialize sample-ready graph inputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from path_roots import (
    ALIGNED_ROOT,
    CONSTRUCT_ROOT,
    INTEGRATE_ROOT,
    PERSONALIZE_ROOT,
    PROJECT_ROOT,
    SPLITS_ROOT,
)

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]
SPLITS = ["train", "valid", "test"]
EHR_RELATIONS = ["has_admission", "has_diagnosis", "has_procedure", "has_medication"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--emit-spec", action="store_true")
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument("--materialize-samples", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
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


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def selected_visit_ids(instance_row: dict) -> List[str]:
    if instance_row["granularity"] == "visit":
        return [str(v) for v in instance_row["context_visit_ids"]]
    return [str(v) for v in instance_row["observation_visit_ids"]]


def make_instance_id(instance_row: dict) -> str:
    dataset = instance_row["dataset"]
    task = instance_row["task"]
    patient_id = str(instance_row["patient_id"])
    if instance_row["granularity"] == "visit":
        return "{0}:{1}:{2}:visit:{3}:{4}".format(
            dataset,
            task,
            patient_id,
            instance_row["target_visit_index"],
            instance_row["target_visit_id"],
        )
    return "{0}:{1}:{2}:patient".format(dataset, task, patient_id)


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def build_spec(dataset: str, task: str, split: str) -> dict:
    task_root = INTEGRATE_ROOT / dataset / task
    return {
        "stage": "integrate_data",
        "dataset": dataset,
        "task": task,
        "split": split,
        "inputs": {
            "aligned_ehr": str(ALIGNED_ROOT / dataset / "patients_visits_aligned.jsonl"),
            "split_instances": str(SPLITS_ROOT / dataset / task / f"{split}.jsonl"),
            "integrate_manifest": str(INTEGRATE_ROOT / dataset / task / f"{split}_integrate_manifest.jsonl"),
            "pkg_nodes": str(PERSONALIZE_ROOT / dataset / task / f"{split}_pkg_nodes.jsonl"),
            "pkg_edges": str(PERSONALIZE_ROOT / dataset / task / f"{split}_pkg_edges.jsonl"),
            "shared_schema": str(CONSTRUCT_ROOT / "shared" / task / "relation_schema.json"),
        },
        "outputs": {
            "task_root": str(task_root),
            "spec_path": str(task_root / f"{split}_integrate_data_spec.json"),
            "patient_index": str(task_root / f"{split}_aligned_patient_index.json"),
            "pkg_nodes_index": str(task_root / f"{split}_pkg_nodes_index.json"),
            "pkg_edges_index": str(task_root / f"{split}_pkg_edges_index.json"),
            "relation_vocab": str(task_root / "relation_vocab.json"),
            "sample_preview": str(task_root / f"{split}_integrate_samples.preview.jsonl"),
            "stats_path": str(task_root / f"{split}_integrate_data_stats.json"),
        },
    }


def ensure_inputs_exist(spec: dict) -> None:
    missing = [path for path in spec["inputs"].values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def build_patient_index(path: Path) -> Dict[str, dict]:
    index = {}
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            index[str(row["patient_id"])] = {
                "start": start,
                "end": handle.tell(),
            }
    return index


def build_group_index(path: Path, group_field: str) -> Dict[str, dict]:
    index = {}
    current_key = None
    start = None
    count = 0
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            key = row[group_field]
            if key != current_key:
                if current_key is not None:
                    index[current_key]["end"] = offset
                    index[current_key]["num_rows"] = count
                current_key = key
                start = offset
                count = 0
                index[current_key] = {"start": start}
            count += 1
        if current_key is not None:
            index[current_key]["end"] = handle.tell()
            index[current_key]["num_rows"] = count
    return index


def load_segment_rows(path: Path, start: int, end: int) -> List[dict]:
    rows = []
    with path.open("rb") as handle:
        handle.seek(start)
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                break
            rows.append(json.loads(line))
    return rows


def load_patient_row(aligned_path: Path, patient_index: Dict[str, dict], patient_id: str) -> dict:
    segment = patient_index[str(patient_id)]
    rows = load_segment_rows(aligned_path, segment["start"], segment["end"])
    if len(rows) != 1:
        raise ValueError("Expected one aligned patient row for patient_id {0}".format(patient_id))
    return rows[0]


def build_relation_vocab(schema_path: Path) -> dict:
    payload = load_json(schema_path)
    canonical_relations = sorted(
        {
            row["canonical_relation"]
            for row in payload.get("relations", [])
            if row.get("keep") is True and row.get("canonical_relation")
        }
    )
    relation_to_id = {"<pad>": 0}
    for relation in EHR_RELATIONS + canonical_relations:
        if relation not in relation_to_id:
            relation_to_id[relation] = len(relation_to_id)
    return {
        "ehr_relations": EHR_RELATIONS,
        "canonical_relations": canonical_relations,
        "relation_to_id": relation_to_id,
    }


def materialize_sample(
    manifest_row: dict,
    split_instance: dict,
    patient_row: dict,
    pkg_nodes: List[dict],
    pkg_edges: List[dict],
    relation_vocab: dict,
) -> dict:
    selected_ids = set(selected_visit_ids(split_instance))
    selected_visits = [visit for visit in patient_row["visits"] if str(visit["visit_id"]) in selected_ids]

    pkg_node_list = []
    node_id_to_idx = {}
    for row in pkg_nodes:
        node_id_to_idx[row["node_id"]] = len(pkg_node_list)
        pkg_node_list.append(
            {
                "node_index": len(pkg_node_list),
                "node_id": row["node_id"],
                "node_kind": row["node_kind"],
                "concept_id": row.get("concept_id"),
                "visit_id": row.get("visit_id"),
                "source": row["source"],
            }
        )

    edge_list = []
    for row in pkg_edges:
        edge_list.append(
            {
                "src_index": node_id_to_idx[row["source_id"]],
                "dst_index": node_id_to_idx[row["target_id"]],
                "relation_type": row["relation_type"],
                "relation_id": relation_vocab["relation_to_id"][row["relation_type"]],
                "edge_source": row["edge_source"],
                "path_role": row["path_role"],
            }
        )

    ehr_visits = []
    for visit in selected_visits:
        visit_concepts = []
        for concept in visit.get("aligned_concepts", []):
            visit_concepts.append(
                {
                    "concept_id": concept["ehr_concept_id"],
                    "source_type": concept["source_type"],
                }
            )
        ehr_visits.append(
            {
                "visit_id": str(visit["visit_id"]),
                "admittime": visit["admittime"],
                "dischtime": visit["dischtime"],
                "concepts": visit_concepts,
            }
        )

    return {
        "instance_id": manifest_row["instance_id"],
        "pkg_id": manifest_row["pkg_id"],
        "label": manifest_row["label"],
        "granularity": manifest_row["granularity"],
        "prediction_time": manifest_row["prediction_time"],
        "ehr_sequence": ehr_visits,
        "pkg_nodes": pkg_node_list,
        "pkg_edges": edge_list,
        "pkg_stats": {
            "num_nodes": manifest_row["pkg_num_nodes"],
            "num_edges": manifest_row["pkg_num_edges"],
            "num_ehr_edges": manifest_row["pkg_num_ehr_edges"],
            "num_concept_1hop_edges": manifest_row["pkg_num_concept_1hop_edges"],
            "num_bridge_path_edges": manifest_row["pkg_num_bridge_path_edges"],
        },
    }


def main() -> None:
    args = parse_args()
    spec = build_spec(args.dataset, args.task, args.split)
    ensure_inputs_exist(spec)
    if args.emit_spec:
        write_json(Path(spec["outputs"]["spec_path"]), spec)

    if args.build_indexes or args.materialize_samples:
        patient_index = build_patient_index(Path(spec["inputs"]["aligned_ehr"]))
        pkg_nodes_index = build_group_index(Path(spec["inputs"]["pkg_nodes"]), "pkg_id")
        pkg_edges_index = build_group_index(Path(spec["inputs"]["pkg_edges"]), "pkg_id")
        relation_vocab = build_relation_vocab(Path(spec["inputs"]["shared_schema"]))

        write_json(Path(spec["outputs"]["patient_index"]), patient_index)
        write_json(Path(spec["outputs"]["pkg_nodes_index"]), pkg_nodes_index)
        write_json(Path(spec["outputs"]["pkg_edges_index"]), pkg_edges_index)
        write_json(Path(spec["outputs"]["relation_vocab"]), relation_vocab)

        stats = {
            "num_patient_index_rows": len(patient_index),
            "num_pkg_node_index_rows": len(pkg_nodes_index),
            "num_pkg_edge_index_rows": len(pkg_edges_index),
            "num_relations": len(relation_vocab["relation_to_id"]),
        }

        if args.materialize_samples:
            manifests = list(iter_jsonl(Path(spec["inputs"]["integrate_manifest"])))[: args.limit]
            split_instances = {
                make_instance_id(row): row
                for row in iter_jsonl(Path(spec["inputs"]["split_instances"]))
            }
            preview_rows = []
            for manifest_row in manifests:
                pkg_id = manifest_row["pkg_id"]
                patient_id = manifest_row["patient_id"]
                patient_row = load_patient_row(
                    Path(spec["inputs"]["aligned_ehr"]),
                    patient_index,
                    patient_id,
                )
                node_segment = pkg_nodes_index[pkg_id]
                edge_segment = pkg_edges_index[pkg_id]
                pkg_nodes = load_segment_rows(
                    Path(spec["inputs"]["pkg_nodes"]), node_segment["start"], node_segment["end"]
                )
                pkg_edges = load_segment_rows(
                    Path(spec["inputs"]["pkg_edges"]), edge_segment["start"], edge_segment["end"]
                )
                preview_rows.append(
                    materialize_sample(
                        manifest_row=manifest_row,
                        split_instance=split_instances[manifest_row["instance_id"]],
                        patient_row=patient_row,
                        pkg_nodes=pkg_nodes,
                        pkg_edges=pkg_edges,
                        relation_vocab=relation_vocab,
                    )
                )
            write_jsonl(Path(spec["outputs"]["sample_preview"]), preview_rows)
            stats["num_preview_samples"] = len(preview_rows)

        write_json(Path(spec["outputs"]["stats_path"]), stats)

    print(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
