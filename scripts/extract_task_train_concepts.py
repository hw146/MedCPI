#!/usr/bin/env python3
"""Extract task-specific train anchor concepts for Construct."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from path_roots import ALIGNED_ROOT as DEFAULT_ALIGNED_ROOT, CONSTRUCT_ROOT as DEFAULT_OUTPUT_ROOT, SPLITS_ROOT as DEFAULT_SPLITS_ROOT

TASKS = {"mortality", "readmission_30d", "t2dm_onset", "cad_onset"}
VISIT_LEVEL_TASKS = {"mortality", "readmission_30d"}
PATIENT_LEVEL_TASKS = {"t2dm_onset", "cad_onset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--include-rxnorm-only",
        action="store_true",
        help="Include RxNorm-only medication concepts in the exported anchor space.",
    )
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


def load_train_scope(split_path: Path, task: str) -> Dict[str, Set[str]]:
    patient_to_visit_ids: Dict[str, Set[str]] = defaultdict(set)
    for row in iter_jsonl(split_path):
        patient_id = row["patient_id"]
        if task in VISIT_LEVEL_TASKS:
            patient_to_visit_ids[patient_id].update(row.get("context_visit_ids", []))
        elif task in PATIENT_LEVEL_TASKS:
            patient_to_visit_ids[patient_id].update(row.get("observation_visit_ids", []))
        else:
            raise ValueError(f"Unsupported task: {task}")
    return patient_to_visit_ids


def aligned_concepts_for_visit(visit: dict) -> List[dict]:
    if "aligned_concepts" in visit:
        return visit["aligned_concepts"]
    return visit.get("kg_anchor_concepts", [])


def concept_allowed(concept: dict, include_rxnorm_only: bool) -> bool:
    if concept.get("kg_anchor") is True:
        return True
    if include_rxnorm_only and concept.get("concept_space") == "rxnorm":
        return True
    return False


def build_anchor_exports(
    aligned_path: Path,
    patient_to_visit_ids: Dict[str, Set[str]],
    include_rxnorm_only: bool,
) -> Tuple[List[dict], dict]:
    concept_index: Dict[str, dict] = {}
    stats = Counter()
    excluded_rxnorm_only: Set[str] = set()

    for patient in iter_jsonl(aligned_path):
        patient_id = patient["patient_id"]
        target_visit_ids = patient_to_visit_ids.get(patient_id)
        if not target_visit_ids:
            continue

        stats["patients_in_train_scope"] += 1
        for visit in patient["visits"]:
            visit_id = visit["visit_id"]
            if visit_id not in target_visit_ids:
                continue

            stats["visits_in_train_scope"] += 1
            for concept in aligned_concepts_for_visit(visit):
                concept_id = concept["ehr_concept_id"]
                if concept.get("concept_space") == "rxnorm" and concept.get("umls_cui") is None:
                    excluded_rxnorm_only.add(concept_id)
                if not concept_allowed(concept, include_rxnorm_only):
                    continue

                if concept_id not in concept_index:
                    concept_index[concept_id] = {
                        "concept_id": concept_id,
                        "umls_cui": concept.get("umls_cui"),
                        "rxnorm_rxcui": concept.get("rxnorm_rxcui"),
                        "concept_space": concept.get("concept_space"),
                        "kg_anchor": bool(concept.get("kg_anchor")),
                        "preferred_term": concept.get("preferred_term"),
                        "matched_sab": concept.get("matched_sab"),
                        "matched_tty": concept.get("matched_tty"),
                        "source_types": set(),
                        "patient_ids": set(),
                        "visit_ids": set(),
                        "mention_ids": set(),
                        "mapping_modes": Counter(),
                        "event_count": 0,
                    }

                record = concept_index[concept_id]
                record["source_types"].add(concept["source_type"])
                record["patient_ids"].add(patient_id)
                record["visit_ids"].add(visit_id)
                record["mention_ids"].update(concept.get("mention_ids", []))
                for mode in concept.get("mapping_modes", []):
                    record["mapping_modes"][mode] += 1
                record["event_count"] += int(concept.get("event_count", 0))

                stats["included_concept_occurrences"] += int(concept.get("event_count", 0))
                stats[f"source_type_{concept['source_type']}"] += int(concept.get("event_count", 0))
                stats[f"concept_space_{concept.get('concept_space')}"] += int(concept.get("event_count", 0))

    exported_rows = []
    for concept_id, record in sorted(concept_index.items()):
        exported_rows.append(
            {
                "concept_id": concept_id,
                "umls_cui": record["umls_cui"],
                "rxnorm_rxcui": record["rxnorm_rxcui"],
                "concept_space": record["concept_space"],
                "kg_anchor": record["kg_anchor"],
                "preferred_term": record["preferred_term"],
                "matched_sab": record["matched_sab"],
                "matched_tty": record["matched_tty"],
                "source_types": sorted(record["source_types"]),
                "num_patients": len(record["patient_ids"]),
                "num_visits": len(record["visit_ids"]),
                "num_mentions": len(record["mention_ids"]),
                "event_count": record["event_count"],
                "mapping_modes": dict(sorted(record["mapping_modes"].items())),
            }
        )

    stats["num_unique_anchor_concepts"] = len(exported_rows)
    stats["num_unique_rxnorm_only_excluded"] = len(excluded_rxnorm_only)
    return exported_rows, {"stats": dict(sorted(stats.items())), "excluded_rxnorm_only_concepts": sorted(excluded_rxnorm_only)}


def main() -> None:
    args = parse_args()
    split_path = args.splits_root / args.dataset / args.task / "train.jsonl"
    aligned_path = args.aligned_root / args.dataset / "patients_visits_aligned.jsonl"
    output_dir = args.output_root / args.dataset / args.task

    patient_to_visit_ids = load_train_scope(split_path, args.task)
    anchor_rows, metadata = build_anchor_exports(
        aligned_path=aligned_path,
        patient_to_visit_ids=patient_to_visit_ids,
        include_rxnorm_only=args.include_rxnorm_only,
    )

    write_jsonl(output_dir / "train_anchor_concepts.jsonl", anchor_rows)
    write_json(
        output_dir / "train_anchor_concepts.json",
        {
            "dataset": args.dataset,
            "task": args.task,
            "split_path": str(split_path),
            "aligned_path": str(aligned_path),
            "train_anchor_scope": "training_split_only",
            "construct_protocol": "split_then_construct_then_evaluate",
            "include_rxnorm_only": args.include_rxnorm_only,
            "num_train_patients": len(patient_to_visit_ids),
            "num_train_visits_in_scope": sum(len(visit_ids) for visit_ids in patient_to_visit_ids.values()),
            **metadata,
        },
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "task": args.task,
                "num_anchor_concepts": len(anchor_rows),
                "stats": metadata["stats"],
                "include_rxnorm_only": args.include_rxnorm_only,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
