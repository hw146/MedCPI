#!/usr/bin/env python3
"""Build a task-specific Concept MKG from train anchor concepts and kept relations."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from path_roots import CONSTRUCT_ROOT as DEFAULT_CONSTRUCT_ROOT, UMLS_ROOT

MRREL_PATH = UMLS_ROOT / "MRREL.RRF"
MRCONSO_PATH = UMLS_ROOT / "MRCONSO.RRF"
MRSTY_PATH = UMLS_ROOT / "MRSTY.RRF"
SEMGROUPS_PATH = UMLS_ROOT / "SemGroups.txt"
# Paper text describes four coarse categories:
# Disorders, Findings, Procedures, and Chemicals & Drugs.
# In the local UMLS SemGroups resource, "Finding" (T033) and
# "Sign or Symptom" (T184) are grouped under DISO, so the clinical
# filter is implemented via CHEM / DISO / PROC.
ALLOWED_SEMANTIC_GROUP_IDS = {"CHEM", "DISO", "PROC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=["mortality", "readmission_30d", "t2dm_onset", "cad_onset"], required=True)
    parser.add_argument("--construct-root", type=Path, default=DEFAULT_CONSTRUCT_ROOT)
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument("--mrrel-path", type=Path, default=MRREL_PATH)
    parser.add_argument("--mrconso-path", type=Path, default=MRCONSO_PATH)
    parser.add_argument("--mrsty-path", type=Path, default=MRSTY_PATH)
    parser.add_argument("--semgroups-path", type=Path, default=SEMGROUPS_PATH)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_mrrel_row(line: str) -> Optional[dict]:
    parts = line.rstrip("\n").split("|")
    if len(parts) < 16:
        return None
    return {
        "cui1": parts[0],
        "aui1": parts[1],
        "stype1": parts[2],
        "rel": parts[3],
        "cui2": parts[4],
        "aui2": parts[5],
        "stype2": parts[6],
        "rela": parts[7],
        "rui": parts[8],
        "srui": parts[9],
        "sab": parts[10],
        "sl": parts[11],
        "rg": parts[12],
        "dir": parts[13],
        "suppress": parts[14],
        "cvf": parts[15],
    }


def raw_relation_name(row: dict) -> str:
    value = (row.get("rela") or "").strip()
    if value:
        return value
    return (row.get("rel") or "").strip()


def load_anchor_records(path: Path) -> Dict[str, dict]:
    return {row["concept_id"]: row for row in iter_jsonl(path)}


def load_schema_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def load_schema_decisions(payload: dict) -> Dict[str, str]:
    keep_map = {}
    for row in payload.get("relations", []):
        if row.get("keep") is True:
            keep_map[row["raw_relation"]] = row.get("canonical_relation") or row["raw_relation"]
    return keep_map


def normalized_schema_scope(payload: dict) -> str:
    scope = payload.get("schema_scope")
    if scope in {"task_specific", "task_specific_shared_across_datasets"}:
        return scope
    if scope in {"task_shared", "dataset_task"}:
        if schema_shared_across_datasets(payload, "shared_task"):
            return "task_specific_shared_across_datasets"
        return "task_specific"
    if payload.get("dataset") == "shared":
        return "task_specific_shared_across_datasets"
    return "task_specific"


def schema_shared_across_datasets(payload: dict, schema_resolution: str) -> bool:
    if "shared_across_datasets" in payload:
        return bool(payload["shared_across_datasets"])
    return schema_resolution == "shared_task"


def load_preferred_terms(mrconso_path: Path, cuis: Set[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    scores: Dict[str, tuple] = {}
    with mrconso_path.open("r", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue
            cui = parts[0]
            if cui not in cuis:
                continue
            lat = parts[1]
            ts = parts[2]
            ispref = parts[6]
            term = parts[14].strip()
            if not term:
                continue
            score = (lat == "ENG", ispref == "Y", ts == "P", -len(term))
            if cui not in scores or score > scores[cui]:
                scores[cui] = score
                labels[cui] = term
    return labels


def load_semgroups(semgroups_path: Path) -> Dict[str, dict]:
    groups = {}
    with semgroups_path.open("r", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            group_id, group_name, tui, sty = parts[:4]
            groups[tui] = {
                "group_id": group_id,
                "group_name": group_name,
                "semantic_type": sty,
            }
    return groups


def load_semantic_profiles(mrsty_path: Path, cuis: Set[str], tui_to_group: Dict[str, dict]) -> Dict[str, dict]:
    semantic_types: Dict[str, Set[str]] = defaultdict(set)
    semantic_group_ids: Dict[str, Set[str]] = defaultdict(set)
    semantic_group_names: Dict[str, Set[str]] = defaultdict(set)
    with mrsty_path.open("r", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            if cui not in cuis:
                continue
            tui = parts[1]
            sty = parts[3]
            semantic_types[cui].add(sty)
            group = tui_to_group.get(tui)
            if group is not None:
                semantic_group_ids[cui].add(group["group_id"])
                semantic_group_names[cui].add(group["group_name"])
    return {
        cui: {
            "semantic_types": sorted(semantic_types.get(cui, set())),
            "semantic_group_ids": sorted(semantic_group_ids.get(cui, set())),
            "semantic_group_names": sorted(semantic_group_names.get(cui, set())),
        }
        for cui in cuis
    }


def concept_in_smed(profile: Optional[dict]) -> bool:
    if profile is None:
        return False
    return bool(ALLOWED_SEMANTIC_GROUP_IDS.intersection(profile.get("semantic_group_ids", [])))


def resolve_schema_path(construct_root: Path, dataset: str, task: str, schema_path_arg: Optional[Path]) -> Tuple[Path, str]:
    if schema_path_arg is not None:
        return schema_path_arg, "explicit"

    shared_schema_path = construct_root / "shared" / task / "relation_schema.json"
    if shared_schema_path.exists():
        return shared_schema_path, "shared_task"

    task_dir = construct_root / dataset / task
    dataset_schema_path = task_dir / "relation_schema.json"
    if dataset_schema_path.exists():
        return dataset_schema_path, "dataset_task"

    return dataset_schema_path, "missing"


def main() -> None:
    args = parse_args()
    task_dir = args.construct_root / args.dataset / args.task
    anchor_path = task_dir / "train_anchor_concepts.jsonl"
    schema_path, schema_resolution = resolve_schema_path(
        construct_root=args.construct_root,
        dataset=args.dataset,
        task=args.task,
        schema_path_arg=args.schema_path,
    )
    if not schema_path.exists():
        raise FileNotFoundError("Missing schema file: {0}".format(schema_path))

    if not anchor_path.exists():
        raise FileNotFoundError("Missing train anchor concepts: {0}".format(anchor_path))

    anchors = load_anchor_records(anchor_path)
    anchor_ids = set(anchors)
    schema_payload = load_schema_payload(schema_path)
    if schema_payload.get("task") not in {None, args.task}:
        raise ValueError(
            "Schema task mismatch: expected {0}, found {1}".format(
                args.task, schema_payload.get("task")
            )
        )
    kept_relations = load_schema_decisions(schema_payload)
    tui_to_group = load_semgroups(args.semgroups_path)

    stats = Counter()
    candidate_edges = []
    candidate_nodes: Set[str] = set(anchor_ids)
    if kept_relations:
        with args.mrrel_path.open("r", errors="replace") as handle:
            for line in handle:
                row = parse_mrrel_row(line)
                if row is None:
                    continue
                if row["suppress"] not in {"", "N"}:
                    continue

                raw_relation = raw_relation_name(row)
                canonical_relation = kept_relations.get(raw_relation)
                if canonical_relation is None:
                    continue
                if row["cui1"] not in anchor_ids and row["cui2"] not in anchor_ids:
                    continue

                candidate_edges.append(
                    {
                        "src": row["cui1"],
                        "dst": row["cui2"],
                        "raw_relation": raw_relation,
                        "canonical_relation": canonical_relation,
                        "rel": row["rel"],
                        "rela": row["rela"],
                        "sab": row["sab"],
                        "sl": row["sl"],
                        "rg": row["rg"],
                        "dir": row["dir"],
                        "rui": row["rui"],
                        "srui": row["srui"],
                    }
                )
                candidate_nodes.add(row["cui1"])
                candidate_nodes.add(row["cui2"])
                stats["num_anchor_incident_kept_edges_before_semantic_filter"] += 1

    labels = load_preferred_terms(args.mrconso_path, candidate_nodes)
    semantic_profiles = load_semantic_profiles(args.mrsty_path, candidate_nodes, tui_to_group)

    edge_rows = []
    seen_edges = set()
    for edge in candidate_edges:
        src_profile = semantic_profiles.get(edge["src"])
        dst_profile = semantic_profiles.get(edge["dst"])
        if not concept_in_smed(src_profile) or not concept_in_smed(dst_profile):
            stats["num_edges_dropped_by_semantic_group_filter"] += 1
            continue

        edge_key = (edge["src"], edge["dst"], edge["canonical_relation"])
        if edge_key in seen_edges:
            stats["num_edges_dropped_as_canonical_duplicates"] += 1
            continue
        seen_edges.add(edge_key)

        edge_rows.append(edge)
        stats["num_edges"] += 1
        stats["num_induced_kept_edges"] += 1
        stats["num_anchor_incident_kept_edges_after_semantic_filter"] += 1
        stats["canonical_relation_{0}".format(edge["canonical_relation"])] += 1

    expanded_nodes: Set[str] = set()
    for edge in edge_rows:
        expanded_nodes.add(edge["src"])
        expanded_nodes.add(edge["dst"])

    anchor_outside_smed = 0
    anchor_nodes_in_graph = 0
    for cui in anchor_ids:
        if not concept_in_smed(semantic_profiles.get(cui)):
            anchor_outside_smed += 1
        if cui in expanded_nodes:
            anchor_nodes_in_graph += 1
    stats["num_anchor_nodes_outside_smed"] = anchor_outside_smed
    stats["num_anchor_nodes_in_graph"] = anchor_nodes_in_graph
    stats["num_non_anchor_nodes_in_graph"] = max(0, len(expanded_nodes) - anchor_nodes_in_graph)

    node_rows = []
    for cui in sorted(expanded_nodes):
        anchor_record = anchors.get(cui)
        profile = semantic_profiles.get(cui, {})
        node_rows.append(
            {
                "concept_id": cui,
                "preferred_term": labels.get(cui) or (anchor_record or {}).get("preferred_term"),
                "semantic_types": profile.get("semantic_types", []),
                "semantic_group_ids": profile.get("semantic_group_ids", []),
                "semantic_group_names": profile.get("semantic_group_names", []),
                "is_train_anchor": cui in anchor_ids,
                "anchor_source_types": [] if anchor_record is None else anchor_record.get("source_types", []),
                "anchor_event_count": 0 if anchor_record is None else anchor_record.get("event_count", 0),
                "concept_space": "umls" if not anchor_record else anchor_record.get("concept_space"),
            }
        )

    write_jsonl(task_dir / "concept_mkg_nodes.jsonl", node_rows)
    write_jsonl(task_dir / "concept_mkg_edges.jsonl", edge_rows)
    write_json(
        task_dir / "construct_stats.json",
        {
            "dataset": args.dataset,
            "task": args.task,
            "anchor_path": str(anchor_path),
            "schema_path": str(schema_path),
            "schema_resolution": schema_resolution,
            "schema_scope": normalized_schema_scope(schema_payload),
            "schema_shared_across_datasets": schema_shared_across_datasets(schema_payload, schema_resolution),
            "schema_decision_source": schema_payload.get("decision_source"),
            "schema_dataset": schema_payload.get("dataset"),
            "schema_task": schema_payload.get("task"),
            "construct_protocol": "split_then_construct_then_evaluate",
            "train_anchor_scope": "training_split_only",
            "global_expansion_hops": 1,
            "mrrel_path": str(args.mrrel_path),
            "mrconso_path": str(args.mrconso_path),
            "mrsty_path": str(args.mrsty_path),
            "semgroups_path": str(args.semgroups_path),
            "allowed_semantic_group_ids": sorted(ALLOWED_SEMANTIC_GROUP_IDS),
            "num_train_anchor_concepts": len(anchor_ids),
            "num_nodes": len(node_rows),
            "num_edges": len(edge_rows),
            "num_kept_raw_relations": len(kept_relations),
            "stats": dict(sorted(stats.items())),
        },
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "task": args.task,
                "num_nodes": len(node_rows),
                "num_edges": len(edge_rows),
                "num_kept_raw_relations": len(kept_relations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
