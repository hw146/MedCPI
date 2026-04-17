#!/usr/bin/env python3
"""Utilities for Personalize-stage PKG construction."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


EHR_RELATIONS = {
    "diagnosis": "has_diagnosis",
    "procedure": "has_procedure",
    "medication": "has_medication",
}


def parse_mrrel_row(line: str) -> dict:
    parts = line.rstrip("\n").split("|")
    if len(parts) < 16:
        return {}
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


def selected_visit_ids(instance_row: dict) -> List[str]:
    granularity = instance_row["granularity"]
    if granularity == "visit":
        return [str(value) for value in instance_row.get("context_visit_ids", [])]
    if granularity == "patient":
        return [str(value) for value in instance_row.get("observation_visit_ids", [])]
    raise ValueError("Unsupported task granularity: {0}".format(granularity))


def anchor_scope_for_granularity(granularity: str) -> str:
    if granularity == "visit":
        return "concepts_observed_in_prior_visits"
    if granularity == "patient":
        return "concepts_observed_within_observation_window"
    raise ValueError("Unsupported task granularity: {0}".format(granularity))


def make_instance_id(instance_row: dict) -> str:
    dataset = instance_row["dataset"]
    task = instance_row["task"]
    patient_id = str(instance_row["patient_id"])
    granularity = instance_row["granularity"]
    if granularity == "visit":
        return "{0}:{1}:{2}:visit:{3}:{4}".format(
            dataset,
            task,
            patient_id,
            instance_row["target_visit_index"],
            instance_row["target_visit_id"],
        )
    if granularity == "patient":
        return "{0}:{1}:{2}:patient".format(dataset, task, patient_id)
    raise ValueError("Unsupported task granularity: {0}".format(granularity))


def make_pkg_id(instance_row: dict) -> str:
    return "pkg::{0}".format(make_instance_id(instance_row))


def patient_node_id(patient_id: str) -> str:
    return "patient:{0}".format(patient_id)


def visit_node_id(visit_id: str) -> str:
    return "visit:{0}".format(visit_id)


def concept_node_id(concept_id: str) -> str:
    return "concept:{0}".format(concept_id)


def load_patient_rows_for_instances(aligned_path: Path, patient_ids: Sequence[str]) -> Dict[str, dict]:
    wanted = set(patient_ids)
    rows: Dict[str, dict] = {}
    for row in iter_jsonl(aligned_path):
        patient_id = str(row["patient_id"])
        if patient_id in wanted:
            rows[patient_id] = row
            if len(rows) == len(wanted):
                break
    missing = sorted(wanted.difference(rows))
    if missing:
        raise ValueError(
            "Missing aligned EHR rows for patient ids: {0}".format(", ".join(missing[:10]))
        )
    return rows


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


def load_patient_row(aligned_path: Path, patient_index: Dict[str, dict], patient_id: str) -> dict:
    segment = patient_index[str(patient_id)]
    with aligned_path.open("rb") as handle:
        handle.seek(segment["start"])
        line = handle.readline()
    return json.loads(line)


def load_concept_mkg_graph(nodes_path: Path, edges_path: Path) -> dict:
    node_lookup = {}
    for row in iter_jsonl(nodes_path):
        node_lookup[row["concept_id"]] = row

    incident_edges = defaultdict(list)
    for row in iter_jsonl(edges_path):
        src = row["src"]
        dst = row["dst"]
        incident_edges[src].append((dst, row))
        incident_edges[dst].append((src, row))

    return {
        "nodes": node_lookup,
        "incident_edges": incident_edges,
    }


def load_schema_keep_map(schema_path: Path) -> Dict[str, str]:
    payload = json.loads(schema_path.read_text())
    keep_map = {}
    for row in payload.get("relations", []):
        if row.get("keep") is True:
            keep_map[row["raw_relation"]] = row.get("canonical_relation") or row["raw_relation"]
    return keep_map


def load_anchor_bridge_index(mrrel_path: Path, schema_path: Path, anchor_concepts: Sequence[str]) -> dict:
    keep_map = load_schema_keep_map(schema_path)
    anchor_set = set(anchor_concepts)
    out_from_anchor = defaultdict(list)
    in_to_anchor = defaultdict(list)
    seen_edges = set()
    with mrrel_path.open("r", errors="replace") as handle:
        for line in handle:
            row = parse_mrrel_row(line)
            if not row:
                continue
            if row["suppress"] not in {"", "N"}:
                continue
            raw_relation = raw_relation_name(row)
            canonical_relation = keep_map.get(raw_relation)
            if canonical_relation is None:
                continue
            edge_key = (row["cui1"], canonical_relation, row["cui2"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge = {
                "src": row["cui1"],
                "dst": row["cui2"],
                "canonical_relation": canonical_relation,
            }
            if row["cui1"] in anchor_set:
                out_from_anchor[row["cui1"]].append(edge)
            if row["cui2"] in anchor_set:
                in_to_anchor[row["cui2"]].append(edge)
    for src in list(out_from_anchor):
        out_from_anchor[src].sort(key=lambda item: (item["dst"], item["canonical_relation"]))
    for dst in list(in_to_anchor):
        in_to_anchor[dst].sort(key=lambda item: (item["src"], item["canonical_relation"]))
    return {
        "out_from_anchor": out_from_anchor,
        "in_to_anchor": in_to_anchor,
    }


def collect_ehr_subgraph(instance_row: dict, patient_row: dict, split_name: str) -> Tuple[dict, List[dict], List[dict]]:
    patient_id = str(instance_row["patient_id"])
    pkg_id = make_pkg_id(instance_row)
    instance_id = make_instance_id(instance_row)
    selected_ids = selected_visit_ids(instance_row)
    selected_id_set = set(selected_ids)
    visits = patient_row.get("visits", [])
    visit_lookup = {str(visit["visit_id"]): visit for visit in visits}
    missing_visits = [visit_id for visit_id in selected_ids if visit_id not in visit_lookup]
    if missing_visits:
        raise ValueError(
            "Instance references unknown visits for patient {0}: {1}".format(
                patient_id, ", ".join(missing_visits)
            )
        )

    ordered_visits = [visit for visit in visits if str(visit["visit_id"]) in selected_id_set]
    anchor_concepts = set()
    node_rows = []
    edge_rows = []
    seen_nodes = set()
    seen_edges = set()

    def add_node(row: dict) -> None:
        key = row["node_id"]
        if key not in seen_nodes:
            seen_nodes.add(key)
            node_rows.append(row)

    def add_edge(row: dict) -> None:
        key = (row["source_id"], row["relation_type"], row["target_id"])
        if key not in seen_edges:
            seen_edges.add(key)
            edge_rows.append(row)

    add_node(
        {
            "pkg_id": pkg_id,
            "instance_id": instance_id,
            "patient_id": patient_id,
            "node_id": patient_node_id(patient_id),
            "node_kind": "patient",
            "concept_id": None,
            "visit_id": None,
            "source": "ehr",
            "construction_state": "ehr_subgraph_only",
        }
    )

    ehr_concept_edge_count = 0
    for visit in ordered_visits:
        visit_id = str(visit["visit_id"])
        visit_node = visit_node_id(visit_id)
        add_node(
            {
                "pkg_id": pkg_id,
                "instance_id": instance_id,
                "patient_id": patient_id,
                "node_id": visit_node,
                "node_kind": "visit",
                "concept_id": None,
                "visit_id": visit_id,
                "source": "ehr",
                "construction_state": "ehr_subgraph_only",
            }
        )
        add_edge(
            {
                "pkg_id": pkg_id,
                "instance_id": instance_id,
                "patient_id": patient_id,
                "source_id": patient_node_id(patient_id),
                "relation_type": "has_admission",
                "target_id": visit_node,
                "edge_source": "ehr",
                "path_role": "ehr",
                "construction_state": "ehr_subgraph_only",
            }
        )

        for concept in visit.get("kg_anchor_concepts", []):
            concept_id = concept["ehr_concept_id"]
            source_type = concept["source_type"]
            relation_type = EHR_RELATIONS[source_type]
            anchor_concepts.add(concept_id)
            add_node(
                {
                    "pkg_id": pkg_id,
                    "instance_id": instance_id,
                    "patient_id": patient_id,
                    "node_id": concept_node_id(concept_id),
                    "node_kind": "concept",
                    "concept_id": concept_id,
                    "visit_id": None,
                    "source": "ehr",
                    "construction_state": "ehr_subgraph_only",
                }
            )
            add_edge(
                {
                    "pkg_id": pkg_id,
                    "instance_id": instance_id,
                    "patient_id": patient_id,
                    "source_id": visit_node,
                    "relation_type": relation_type,
                    "target_id": concept_node_id(concept_id),
                    "edge_source": "ehr",
                    "path_role": "ehr",
                    "construction_state": "ehr_subgraph_only",
                }
            )
            ehr_concept_edge_count += 1

    metadata = {
        "pkg_id": pkg_id,
        "instance_id": instance_id,
        "dataset": instance_row["dataset"],
        "task": instance_row["task"],
        "split": split_name,
        "granularity": instance_row["granularity"],
        "patient_id": patient_id,
        "prediction_time": instance_row["prediction_time"],
        "anchor_concepts": sorted(anchor_concepts),
        "num_anchor_concepts": len(anchor_concepts),
        "num_nodes": len(node_rows),
        "num_edges": len(edge_rows),
        "num_ehr_edges": len(edge_rows),
        "num_concept_1hop_edges": 0,
        "num_bridge_path_edges": 0,
        "num_bridge_paths": 0,
        "max_path_length": 2,
        "k_nbr": 30,
        "k_path": 2,
        "anchor_scope": anchor_scope_for_granularity(instance_row["granularity"]),
        "construction_state": "ehr_subgraph_only",
    }
    if instance_row["granularity"] == "visit":
        metadata.update(
            {
                "target_visit_id": instance_row["target_visit_id"],
                "target_visit_index": instance_row["target_visit_index"],
                "context_visit_ids": selected_ids,
            }
        )
    else:
        metadata.update(
            {
                "observation_start": instance_row["observation_start"],
                "observation_end": instance_row["observation_end"],
                "observation_visit_ids": selected_ids,
            }
        )
    return metadata, node_rows, edge_rows


def collect_instance_anchor_concepts(instance_row: dict, patient_row: dict) -> List[str]:
    selected_ids = set(selected_visit_ids(instance_row))
    anchors = set()
    for visit in patient_row.get("visits", []):
        if str(visit["visit_id"]) not in selected_ids:
            continue
        for concept in visit.get("kg_anchor_concepts", []):
            anchors.add(concept["ehr_concept_id"])
    return sorted(anchors)


def cooccurring_anchor_pairs(instance_row: dict, patient_row: dict) -> List[Tuple[str, str]]:
    selected_ids = set(selected_visit_ids(instance_row))
    pairs = set()
    for visit in patient_row.get("visits", []):
        if str(visit["visit_id"]) not in selected_ids:
            continue
        anchors = sorted({row["ehr_concept_id"] for row in visit.get("kg_anchor_concepts", [])})
        for idx, left in enumerate(anchors):
            for right in anchors[idx + 1 :]:
                pairs.add((left, right))
    return sorted(pairs)


def attach_concept_1hop(
    metadata: dict,
    node_rows: List[dict],
    edge_rows: List[dict],
    concept_mkg_graph: dict,
    k_nbr: int,
) -> Tuple[dict, List[dict], List[dict]]:
    updated_metadata = dict(metadata)
    updated_metadata["construction_state"] = "ehr_plus_concept_1hop"

    updated_nodes = []
    seen_node_ids = set()
    for row in node_rows:
        new_row = dict(row)
        new_row["construction_state"] = "ehr_plus_concept_1hop"
        updated_nodes.append(new_row)
        seen_node_ids.add(new_row["node_id"])

    updated_edges = []
    seen_edge_keys = set()
    for row in edge_rows:
        new_row = dict(row)
        new_row["construction_state"] = "ehr_plus_concept_1hop"
        updated_edges.append(new_row)
        seen_edge_keys.add((new_row["source_id"], new_row["relation_type"], new_row["target_id"]))

    concept_nodes = concept_mkg_graph["nodes"]
    incident_edges = concept_mkg_graph["incident_edges"]
    num_added_1hop_edges = 0

    for anchor_concept in updated_metadata["anchor_concepts"]:
        # When an anchor has more than K_nbr candidates, keep a deterministic
        # prefix by neighbor id / relation order so the paper-fixed cap is stable.
        candidate_edges = sorted(
            incident_edges.get(anchor_concept, []),
            key=lambda item: (
                item[0],
                item[1]["canonical_relation"],
                item[1]["src"],
                item[1]["dst"],
            ),
        )

        selected_neighbors = []
        selected_neighbor_set = set()
        for neighbor_concept, _edge in candidate_edges:
            if neighbor_concept == anchor_concept:
                continue
            if neighbor_concept in selected_neighbor_set:
                continue
            selected_neighbor_set.add(neighbor_concept)
            selected_neighbors.append(neighbor_concept)
            if len(selected_neighbors) >= k_nbr:
                break

        if not selected_neighbor_set:
            continue

        for neighbor_concept, edge in candidate_edges:
            if neighbor_concept not in selected_neighbor_set:
                continue
            for concept_id in (edge["src"], edge["dst"]):
                node_id = concept_node_id(concept_id)
                if node_id in seen_node_ids:
                    continue
                updated_nodes.append(
                    {
                        "pkg_id": updated_metadata["pkg_id"],
                        "instance_id": updated_metadata["instance_id"],
                        "patient_id": updated_metadata["patient_id"],
                        "node_id": node_id,
                        "node_kind": "concept",
                        "concept_id": concept_id,
                        "visit_id": None,
                        "source": "concept_mkg" if concept_id not in updated_metadata["anchor_concepts"] else "ehr",
                        "construction_state": "ehr_plus_concept_1hop",
                    }
                )
                seen_node_ids.add(node_id)

            edge_row = {
                "pkg_id": updated_metadata["pkg_id"],
                "instance_id": updated_metadata["instance_id"],
                "patient_id": updated_metadata["patient_id"],
                "source_id": concept_node_id(edge["src"]),
                "relation_type": edge["canonical_relation"],
                "target_id": concept_node_id(edge["dst"]),
                "edge_source": "concept_mkg",
                "path_role": "concept_1hop",
                "construction_state": "ehr_plus_concept_1hop",
            }
            edge_key = (edge_row["source_id"], edge_row["relation_type"], edge_row["target_id"])
            if edge_key in seen_edge_keys:
                continue
            updated_edges.append(edge_row)
            seen_edge_keys.add(edge_key)
            num_added_1hop_edges += 1

    updated_metadata["num_nodes"] = len(updated_nodes)
    updated_metadata["num_edges"] = len(updated_edges)
    updated_metadata["num_ehr_edges"] = sum(1 for row in updated_edges if row["edge_source"] == "ehr")
    updated_metadata["num_concept_1hop_edges"] = num_added_1hop_edges
    updated_metadata["num_bridge_path_edges"] = 0
    return updated_metadata, updated_nodes, updated_edges


def attach_bridge_paths(
    metadata: dict,
    node_rows: List[dict],
    edge_rows: List[dict],
    instance_row: dict,
    patient_row: dict,
    bridge_index: dict,
    k_path: int,
) -> Tuple[dict, List[dict], List[dict]]:
    updated_metadata = dict(metadata)
    updated_metadata["construction_state"] = "ehr_plus_concept_1hop_plus_bridge_paths"

    updated_nodes = []
    seen_node_ids = set()
    for row in node_rows:
        new_row = dict(row)
        new_row["construction_state"] = "ehr_plus_concept_1hop_plus_bridge_paths"
        updated_nodes.append(new_row)
        seen_node_ids.add(new_row["node_id"])

    updated_edges = []
    seen_edge_keys = set()
    for row in edge_rows:
        new_row = dict(row)
        new_row["construction_state"] = "ehr_plus_concept_1hop_plus_bridge_paths"
        updated_edges.append(new_row)
        seen_edge_keys.add((new_row["source_id"], new_row["relation_type"], new_row["target_id"]))

    out_from_anchor = bridge_index["out_from_anchor"]
    in_to_anchor = bridge_index["in_to_anchor"]
    num_added_bridge_edges = 0
    num_added_paths = 0

    for left_anchor, right_anchor in cooccurring_anchor_pairs(instance_row, patient_row):
        candidate_paths = []
        for start_anchor, end_anchor in [(left_anchor, right_anchor), (right_anchor, left_anchor)]:
            right_by_mid = defaultdict(list)
            for edge2 in in_to_anchor.get(end_anchor, []):
                right_by_mid[edge2["src"]].append(edge2)

            for edge1 in out_from_anchor.get(start_anchor, []):
                mid = edge1["dst"]
                for edge2 in right_by_mid.get(mid, []):
                    candidate_paths.append(
                        (
                            mid,
                            edge1["canonical_relation"],
                            edge2["canonical_relation"],
                            start_anchor,
                            end_anchor,
                            edge1,
                            edge2,
                        )
                    )

        candidate_paths.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        selected_paths = candidate_paths[:k_path]
        for mid, _rel1, _rel2, _start_anchor, _end_anchor, edge1, edge2 in selected_paths:
            path_added_new_content = False
            mid_node_id = concept_node_id(mid)
            if mid_node_id not in seen_node_ids:
                updated_nodes.append(
                    {
                        "pkg_id": updated_metadata["pkg_id"],
                        "instance_id": updated_metadata["instance_id"],
                        "patient_id": updated_metadata["patient_id"],
                        "node_id": mid_node_id,
                        "node_kind": "concept",
                        "concept_id": mid,
                        "visit_id": None,
                        "source": "task_normalized_mkg",
                        "construction_state": "ehr_plus_concept_1hop_plus_bridge_paths",
                    }
                )
                seen_node_ids.add(mid_node_id)
                path_added_new_content = True

            for edge in (edge1, edge2):
                edge_row = {
                    "pkg_id": updated_metadata["pkg_id"],
                    "instance_id": updated_metadata["instance_id"],
                    "patient_id": updated_metadata["patient_id"],
                    "source_id": concept_node_id(edge["src"]),
                    "relation_type": edge["canonical_relation"],
                    "target_id": concept_node_id(edge["dst"]),
                    "edge_source": "task_normalized_mkg",
                    "path_role": "bridge_path",
                    "construction_state": "ehr_plus_concept_1hop_plus_bridge_paths",
                }
                edge_key = (edge_row["source_id"], edge_row["relation_type"], edge_row["target_id"])
                if edge_key in seen_edge_keys:
                    continue
                updated_edges.append(edge_row)
                seen_edge_keys.add(edge_key)
                num_added_bridge_edges += 1
                path_added_new_content = True
            if path_added_new_content:
                num_added_paths += 1

    updated_metadata["num_nodes"] = len(updated_nodes)
    updated_metadata["num_edges"] = len(updated_edges)
    updated_metadata["num_ehr_edges"] = sum(1 for row in updated_edges if row["edge_source"] == "ehr")
    updated_metadata["num_concept_1hop_edges"] = sum(
        1 for row in updated_edges if row["edge_source"] == "concept_mkg"
    )
    updated_metadata["num_bridge_path_edges"] = num_added_bridge_edges
    updated_metadata["num_bridge_paths"] = num_added_paths
    return updated_metadata, updated_nodes, updated_edges
