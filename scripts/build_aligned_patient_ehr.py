#!/usr/bin/env python3
"""Build aligned patient EHR views from raw visits and mapping outputs."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from path_roots import ALIGNED_ROOT, MAPPING_ROOT, PREPROCESSED_ROOT
from prepare_mapping_inputs import build_medication_aliases, normalize_code, stable_medication_id

DEFAULT_INPUT_ROOT = PREPROCESSED_ROOT
DEFAULT_MAPPING_ROOT = MAPPING_ROOT / "sapbert"
DEFAULT_OUTPUT_ROOT = ALIGNED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "both"], default="both")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=DEFAULT_MAPPING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
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


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def patient_id_of(patient_row: dict) -> str:
    return str(patient_row["patient_id"])


def update_stats_from_patient(stats: Counter, patient_row: dict) -> None:
    stats["patients"] += 1
    visits = patient_row.get("visits", [])
    stats["visits"] += len(visits)
    for visit in visits:
        for event in visit.get("aligned_events", []):
            stats["{0}_{1}".format(event["source_type"], event["alignment_status"])] += 1


def load_processed_patient_ids(path: Path) -> Set[str]:
    processed = set()
    if not path.exists():
        return processed
    for row in iter_jsonl(path):
        processed.add(patient_id_of(row))
    return processed


def rebuild_partial_stats(path: Path, dataset: str, input_path: Path, output_path: Path) -> dict:
    stats = Counter()
    for row in iter_jsonl(path):
        update_stats_from_patient(stats, row)
    return {
        "dataset": dataset,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_patients": stats["patients"],
        "num_visits": stats["visits"],
        "counts": dict(sorted(stats.items())),
    }


def aligned_output_paths(
    output_root: Path,
    dataset: str,
    num_shards: int,
    shard_index: int,
) -> Dict[str, Path]:
    output_dir = output_root / dataset
    if num_shards == 1:
        final_output_path = output_dir / "patients_visits_aligned.jsonl"
        partial_output_path = output_dir / "patients_visits_aligned.partial.jsonl"
        final_stats_path = output_dir / "aligned_ehr_stats.json"
        partial_stats_path = output_dir / "aligned_ehr_stats.partial.json"
        return {
            "output_dir": output_dir,
            "final_output_path": final_output_path,
            "partial_output_path": partial_output_path,
            "final_stats_path": final_stats_path,
            "partial_stats_path": partial_stats_path,
        }

    shard_dir = output_dir / "shards"
    shard_stem = "patients_visits_aligned.shard-{0:05d}-of-{1:05d}".format(shard_index, num_shards)
    final_output_path = shard_dir / "{0}.jsonl".format(shard_stem)
    partial_output_path = shard_dir / "{0}.partial.jsonl".format(shard_stem)
    final_stats_path = shard_dir / "{0}.stats.json".format(shard_stem)
    partial_stats_path = shard_dir / "{0}.stats.partial.json".format(shard_stem)
    return {
        "output_dir": output_dir,
        "final_output_path": final_output_path,
        "partial_output_path": partial_output_path,
        "final_stats_path": final_stats_path,
        "partial_stats_path": partial_stats_path,
    }


def write_progress_summary(
    dataset: str,
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    stats: Counter,
) -> dict:
    summary = {
        "dataset": dataset,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_patients": stats["patients"],
        "num_visits": stats["visits"],
        "counts": dict(sorted(stats.items())),
    }
    write_json(stats_path, summary)
    return summary


def cleanup_stale_partial_outputs(paths: Dict[str, Path]) -> None:
    if paths["final_output_path"].exists():
        remove_if_exists(paths["partial_output_path"])
    if paths["final_stats_path"].exists():
        remove_if_exists(paths["partial_stats_path"])


def merge_dataset_shards(dataset: str, output_root: Path, num_shards: int, overwrite: bool) -> dict:
    if num_shards <= 1:
        raise ValueError("--merge-shards requires --num-shards > 1")

    output_dir = output_root / dataset
    shard_dir = output_dir / "shards"
    final_output_path = output_dir / "patients_visits_aligned.jsonl"
    partial_output_path = output_dir / "patients_visits_aligned.partial.jsonl"
    final_stats_path = output_dir / "aligned_ehr_stats.json"
    partial_stats_path = output_dir / "aligned_ehr_stats.partial.json"

    if overwrite:
        remove_if_exists(final_output_path)
        remove_if_exists(partial_output_path)
        remove_if_exists(final_stats_path)
        remove_if_exists(partial_stats_path)

    if final_output_path.exists() and final_stats_path.exists() and not overwrite:
        cleanup_stale_partial_outputs(
            {
                "final_output_path": final_output_path,
                "partial_output_path": partial_output_path,
                "final_stats_path": final_stats_path,
                "partial_stats_path": partial_stats_path,
            }
        )
        return load_json(final_stats_path)

    remove_if_exists(partial_output_path)
    remove_if_exists(partial_stats_path)

    stats = Counter()
    shard_dir.mkdir(parents=True, exist_ok=True)

    with partial_output_path.open("w") as output_handle:
        for shard_index in range(num_shards):
            shard_paths = aligned_output_paths(output_root, dataset, num_shards, shard_index)
            shard_output_path = shard_paths["final_output_path"]
            shard_stats_path = shard_paths["final_stats_path"]
            if not shard_output_path.exists():
                raise FileNotFoundError("Missing shard output: {0}".format(shard_output_path))
            if not shard_stats_path.exists():
                raise FileNotFoundError("Missing shard stats: {0}".format(shard_stats_path))

            shard_summary = load_json(shard_stats_path)
            for key, value in shard_summary.get("counts", {}).items():
                stats[key] += value

            with shard_output_path.open("r") as shard_handle:
                for line in shard_handle:
                    output_handle.write(line)

    summary = {
        "dataset": dataset,
        "input_path": "merged_shards",
        "output_path": str(final_output_path),
        "num_patients": stats["patients"],
        "num_visits": stats["visits"],
        "counts": dict(sorted(stats.items())),
        "num_shards": num_shards,
    }
    write_json(partial_stats_path, summary)
    partial_output_path.replace(final_output_path)
    partial_stats_path.replace(final_stats_path)
    return summary


def diagnosis_mention_id(dataset: str, version: Optional[int], raw_code: str) -> str:
    return f"{dataset}_diag_icd{version}_{normalize_code(raw_code)}"


def procedure_mention_id(dataset: str, version: Optional[int], raw_code: str) -> str:
    return f"{dataset}_proc_icd{version}_{normalize_code(raw_code)}"


def load_mapping_rows(dataset: str, mapping_root: Path) -> Dict[str, Dict[str, dict]]:
    source_to_path = {
        "diagnosis": mapping_root / dataset / "diagnosis" / "diagnosis_mention_to_umls.jsonl",
        "procedure": mapping_root / dataset / "procedure" / "procedure_mention_to_umls.jsonl",
        "medication": mapping_root / dataset / "medication" / "medication_mention_to_umls.jsonl",
    }
    mappings: Dict[str, Dict[str, dict]] = {}
    for source_type, path in source_to_path.items():
        rows = {}
        for row in iter_jsonl(path):
            rows[row["mention_id"]] = row
        mappings[source_type] = rows
    return mappings


def event_mapping_status(mapping_row: Optional[dict], source_type: str) -> str:
    if mapping_row is None:
        return "missing_mapping"
    if source_type == "medication" and mapping_row.get("retrieval_mode") == "non_drug_supply":
        return "non_drug_supply"
    if mapping_row.get("matched_cui") is None:
        return "unmapped"
    return "mapped"


def aligned_event_record(
    source_type: str,
    mention_id: str,
    event_index: int,
    raw_code: Optional[str],
    event_text: Optional[str],
    mapping_row: Optional[dict],
) -> dict:
    status = event_mapping_status(mapping_row, source_type)
    if mapping_row is None:
        return {
            "source_type": source_type,
            "mention_id": mention_id,
            "event_index": event_index,
            "event_code": raw_code,
            "event_text": event_text,
            "alignment_status": status,
            "ehr_concept_id": None,
            "umls_cui": None,
            "rxnorm_rxcui": None,
            "concept_space": None,
            "kg_anchor": False,
            "mapping_mode": None,
            "mapping_score": None,
            "matched_term_text": None,
            "matched_code": None,
            "matched_sab": None,
            "matched_tty": None,
        }

    matched_cui = mapping_row.get("matched_cui")
    matched_umls_cui = mapping_row.get("matched_umls_cui")
    if source_type in {"diagnosis", "procedure"}:
        matched_umls_cui = matched_cui

    ehr_concept_id = matched_umls_cui or matched_cui
    concept_space = None
    if ehr_concept_id:
        concept_space = "umls" if matched_umls_cui else "rxnorm"

    return {
        "source_type": source_type,
        "mention_id": mention_id,
        "event_index": event_index,
        "event_code": raw_code,
        "event_text": event_text or mapping_row.get("mention_text"),
        "alignment_status": status,
        "ehr_concept_id": ehr_concept_id,
        "umls_cui": matched_umls_cui,
        "rxnorm_rxcui": mapping_row.get("matched_rxcui"),
        "concept_space": concept_space,
        "kg_anchor": bool(matched_umls_cui),
        "mapping_mode": mapping_row.get("retrieval_mode"),
        "mapping_score": mapping_row.get("score"),
        "matched_term_text": mapping_row.get("matched_term_text"),
        "matched_code": mapping_row.get("matched_code"),
        "matched_sab": mapping_row.get("matched_sab"),
        "matched_tty": mapping_row.get("matched_tty"),
    }


def update_concept_summary(concept_map: Dict[Tuple[str, str], dict], event_record: dict) -> None:
    ehr_concept_id = event_record.get("ehr_concept_id")
    if not ehr_concept_id:
        return
    key = (event_record["source_type"], ehr_concept_id)
    if key not in concept_map:
        concept_map[key] = {
            "source_type": event_record["source_type"],
            "ehr_concept_id": ehr_concept_id,
            "umls_cui": event_record.get("umls_cui"),
            "rxnorm_rxcui": event_record.get("rxnorm_rxcui"),
            "concept_space": event_record.get("concept_space"),
            "kg_anchor": bool(event_record.get("kg_anchor")),
            "preferred_term": event_record.get("matched_term_text"),
            "matched_sab": event_record.get("matched_sab"),
            "matched_tty": event_record.get("matched_tty"),
            "event_count": 0,
            "mention_ids": [],
            "event_codes": [],
            "mapping_modes": [],
        }

    summary = concept_map[key]
    summary["event_count"] += 1
    mention_id = event_record["mention_id"]
    if mention_id not in summary["mention_ids"]:
        summary["mention_ids"].append(mention_id)
    event_code = event_record.get("event_code")
    if event_code is not None and event_code not in summary["event_codes"]:
        summary["event_codes"].append(event_code)
    mapping_mode = event_record.get("mapping_mode")
    if mapping_mode is not None and mapping_mode not in summary["mapping_modes"]:
        summary["mapping_modes"].append(mapping_mode)


def build_visit_alignment(dataset: str, visit: dict, mappings: Dict[str, Dict[str, dict]], stats: Counter) -> dict:
    aligned_events: List[dict] = []
    concept_map: Dict[Tuple[str, str], dict] = {}

    for event_index, diagnosis in enumerate(visit.get("diagnoses", [])):
        mention_id = diagnosis_mention_id(dataset, diagnosis.get("version"), diagnosis["code"])
        mapping_row = mappings["diagnosis"].get(mention_id)
        event_record = aligned_event_record(
            source_type="diagnosis",
            mention_id=mention_id,
            event_index=event_index,
            raw_code=diagnosis.get("code"),
            event_text=None,
            mapping_row=mapping_row,
        )
        aligned_events.append(event_record)
        update_concept_summary(concept_map, event_record)
        stats[f"diagnosis_{event_record['alignment_status']}"] += 1

    for event_index, procedure in enumerate(visit.get("procedures", [])):
        mention_id = procedure_mention_id(dataset, procedure.get("version"), procedure["code"])
        mapping_row = mappings["procedure"].get(mention_id)
        event_record = aligned_event_record(
            source_type="procedure",
            mention_id=mention_id,
            event_index=event_index,
            raw_code=procedure.get("code"),
            event_text=None,
            mapping_row=mapping_row,
        )
        aligned_events.append(event_record)
        update_concept_summary(concept_map, event_record)
        stats[f"procedure_{event_record['alignment_status']}"] += 1

    for event_index, medication in enumerate(visit.get("medications", [])):
        medication_text, aliases, medication_metadata = build_medication_aliases(medication)
        if medication_text is None:
            stats["medication_missing_text"] += 1
            continue
        medication_payload = {
            "mention_text": medication_text,
            "text_aliases": aliases,
            **medication_metadata,
        }
        mention_id = stable_medication_id(dataset, medication_payload)
        mapping_row = mappings["medication"].get(mention_id)
        event_record = aligned_event_record(
            source_type="medication",
            mention_id=mention_id,
            event_index=event_index,
            raw_code=None,
            event_text=medication_text,
            mapping_row=mapping_row,
        )
        aligned_events.append(event_record)
        if event_record["alignment_status"] == "mapped":
            update_concept_summary(concept_map, event_record)
        stats[f"medication_{event_record['alignment_status']}"] += 1

    aligned_concepts = sorted(
        concept_map.values(),
        key=lambda row: (row["source_type"], row["ehr_concept_id"]),
    )
    kg_anchor_concepts = [row for row in aligned_concepts if row["kg_anchor"]]
    return {
        **visit,
        "aligned_events": aligned_events,
        "aligned_concepts": aligned_concepts,
        "kg_anchor_concepts": kg_anchor_concepts,
    }


def process_dataset(
    dataset: str,
    input_root: Path,
    mapping_root: Path,
    output_root: Path,
    overwrite: bool,
    resume: bool,
    flush_every: int,
    num_shards: int,
    shard_index: int,
    merge_shards: bool,
) -> dict:
    if merge_shards:
        return merge_dataset_shards(dataset, output_root, num_shards, overwrite)

    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if flush_every < 1:
        raise ValueError("--flush-every must be >= 1")

    mappings = load_mapping_rows(dataset, mapping_root)
    input_path = input_root / dataset / "patients_visits.jsonl"
    paths = aligned_output_paths(output_root, dataset, num_shards, shard_index)
    final_output_path = paths["final_output_path"]
    partial_output_path = paths["partial_output_path"]
    final_stats_path = paths["final_stats_path"]
    partial_stats_path = paths["partial_stats_path"]

    if overwrite:
        remove_if_exists(final_output_path)
        remove_if_exists(partial_output_path)
        remove_if_exists(final_stats_path)
        remove_if_exists(partial_stats_path)

    cleanup_stale_partial_outputs(paths)

    if final_output_path.exists() and final_stats_path.exists() and not overwrite:
        return load_json(final_stats_path)

    if not resume and partial_output_path.exists():
        remove_if_exists(partial_output_path)
        remove_if_exists(partial_stats_path)

    processed_patient_ids: Set[str] = set()
    if partial_output_path.exists():
        processed_patient_ids = load_processed_patient_ids(partial_output_path)
        if partial_stats_path.exists():
            partial_summary = load_json(partial_stats_path)
        else:
            partial_summary = rebuild_partial_stats(
                partial_output_path,
                dataset=dataset,
                input_path=input_path,
                output_path=final_output_path,
            )
            write_json(partial_stats_path, partial_summary)
        stats = Counter(partial_summary.get("counts", {}))
    else:
        stats = Counter()

    buffered_rows: List[dict] = []
    for patient_index, patient in enumerate(iter_jsonl(input_path)):
        if patient_index % num_shards != shard_index:
            continue
        patient_id = patient_id_of(patient)
        if patient_id in processed_patient_ids:
            continue

        aligned_visits = []
        patient_anchor_ids: Set[str] = set()
        patient_concept_ids: Set[str] = set()
        for visit in patient["visits"]:
            aligned_visit = build_visit_alignment(dataset, visit, mappings, stats)
            aligned_visits.append(aligned_visit)
            for concept in aligned_visit["aligned_concepts"]:
                patient_concept_ids.add(concept["ehr_concept_id"])
            for concept in aligned_visit["kg_anchor_concepts"]:
                patient_anchor_ids.add(concept["ehr_concept_id"])

        buffered_rows.append(
            {
                **patient,
                "visits": aligned_visits,
                "num_aligned_concepts": len(patient_concept_ids),
                "num_kg_anchor_concepts": len(patient_anchor_ids),
            }
        )
        stats["patients"] += 1
        stats["visits"] += len(aligned_visits)
        processed_patient_ids.add(patient_id)

        if len(buffered_rows) >= flush_every:
            append_jsonl(partial_output_path, buffered_rows)
            buffered_rows = []
            write_progress_summary(
                dataset=dataset,
                input_path=input_path,
                output_path=final_output_path,
                stats_path=partial_stats_path,
                stats=stats,
            )
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        "patients_written": stats["patients"],
                        "visits_written": stats["visits"],
                        "partial_output_path": str(partial_output_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if buffered_rows:
        append_jsonl(partial_output_path, buffered_rows)

    summary = write_progress_summary(
        dataset=dataset,
        input_path=input_path,
        output_path=final_output_path,
        stats_path=partial_stats_path,
        stats=stats,
    )

    partial_output_path.parent.mkdir(parents=True, exist_ok=True)
    if final_output_path.exists():
        final_output_path.unlink()
    partial_output_path.replace(final_output_path)
    if final_stats_path.exists():
        final_stats_path.unlink()
    partial_stats_path.replace(final_stats_path)
    return summary


def main() -> None:
    args = parse_args()
    summary_path = args.output_root / "aligned_ehr_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    run_summary = {}
    resume = not args.no_resume
    if args.dataset in {"mimic3", "both"}:
        mimic3_summary = process_dataset(
            "mimic3",
            args.input_root,
            args.mapping_root,
            args.output_root,
            overwrite=args.overwrite,
            resume=resume,
            flush_every=args.flush_every,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            merge_shards=args.merge_shards,
        )
        run_summary["mimic3"] = mimic3_summary
        if args.num_shards == 1 or args.merge_shards:
            summary["mimic3"] = mimic3_summary
    if args.dataset in {"mimic4", "both"}:
        mimic4_summary = process_dataset(
            "mimic4",
            args.input_root,
            args.mapping_root,
            args.output_root,
            overwrite=args.overwrite,
            resume=resume,
            flush_every=args.flush_every,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            merge_shards=args.merge_shards,
        )
        run_summary["mimic4"] = mimic4_summary
        if args.num_shards == 1 or args.merge_shards:
            summary["mimic4"] = mimic4_summary
    if args.num_shards == 1 or args.merge_shards:
        write_json(summary_path, summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
