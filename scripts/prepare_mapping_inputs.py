#!/usr/bin/env python3
"""Parse codebooks and extract unique mentions for later SapBERT mapping."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from path_roots import ASSETS_ROOT, MAPPING_ROOT, PREPROCESSED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mimic3", "mimic4", "both"],
        default="both",
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--patients-root",
        type=Path,
        default=PREPROCESSED_ROOT,
        help="Root directory containing patients_visits.jsonl files.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=ASSETS_ROOT,
        help="Root directory containing UMLS/codebook assets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MAPPING_ROOT,
        help="Root directory for parsed codebooks and unique mentions.",
    )
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = " ".join(value.strip().split())
    return value or None


def normalize_code(value: str) -> str:
    return value.replace(".", "").strip().upper()


def normalize_ndc(value: Optional[str]) -> Optional[str]:
    value = normalize_text(value)
    if value is None:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits or None


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


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_simple_codebook(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            code, description = parts
            code = normalize_code(code)
            description = normalize_text(description)
            if code and description:
                mapping[code] = description
    return mapping


def resolve_existing_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Missing required file. Checked:\n- " + "\n- ".join(str(path) for path in candidates))


def parse_icd10_pcs_order_file(path: Path) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    with path.open("r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue

            sequence = line[:5].strip()
            code = normalize_code(line[6:13])
            billable_flag = line[14:15].strip()
            titles = line[16:]
            short_title = normalize_text(titles[:60])
            long_title = normalize_text(titles[60:])

            if not code:
                continue

            mapping[code] = {
                "sequence": sequence,
                "billable_flag": billable_flag,
                "short_title": short_title,
                "long_title": long_title or short_title,
            }
    return mapping


def parse_codebooks(assets_root: Path, output_root: Path) -> Dict[str, dict]:
    icd10_diag_dir = assets_root / "codebooks" / "icd10" / "diagnosis"
    icd10_proc_dir = assets_root / "codebooks" / "icd10" / "procedure"
    icd10_diag_path = resolve_existing_path(
        [
            icd10_diag_dir / "icd10cm_codes.txt",
            icd10_diag_dir / "icd10cm_codes_2025.txt",
        ]
    )
    icd10_proc_path = resolve_existing_path(
        [
            icd10_proc_dir / "icd10pcs_order.txt",
            icd10_proc_dir / "icd10pcs_order_2025.txt",
        ]
    )

    icd9_diag_long = parse_simple_codebook(assets_root / "codebooks" / "icd9" / "CMS32_DESC_LONG_DX.txt")
    icd9_diag_short = parse_simple_codebook(assets_root / "codebooks" / "icd9" / "CMS32_DESC_SHORT_DX.txt")
    icd9_proc_long = parse_simple_codebook(assets_root / "codebooks" / "icd9" / "CMS32_DESC_LONG_SG.txt")
    icd9_proc_short = parse_simple_codebook(assets_root / "codebooks" / "icd9" / "CMS32_DESC_SHORT_SG.txt")
    icd10_diag = parse_simple_codebook(icd10_diag_path)
    icd10_proc = parse_icd10_pcs_order_file(icd10_proc_path)

    parsed = {
        "icd9_diagnosis": {
            code: {"short": icd9_diag_short.get(code), "long": icd9_diag_long.get(code) or icd9_diag_short.get(code)}
            for code in sorted(set(icd9_diag_long) | set(icd9_diag_short))
        },
        "icd9_procedure": {
            code: {"short": icd9_proc_short.get(code), "long": icd9_proc_long.get(code) or icd9_proc_short.get(code)}
            for code in sorted(set(icd9_proc_long) | set(icd9_proc_short))
        },
        "icd10_diagnosis": {
            code: {"short": description, "long": description}
            for code, description in sorted(icd10_diag.items())
        },
        "icd10_procedure": icd10_proc,
    }

    codebook_dir = output_root / "codebooks"
    write_json(codebook_dir / "icd9_diagnosis.json", parsed["icd9_diagnosis"])
    write_json(codebook_dir / "icd9_procedure.json", parsed["icd9_procedure"])
    write_json(codebook_dir / "icd10_diagnosis.json", parsed["icd10_diagnosis"])
    write_json(codebook_dir / "icd10_procedure.json", parsed["icd10_procedure"])
    write_json(
        codebook_dir / "codebook_stats.json",
        {
            "icd9_diagnosis": len(parsed["icd9_diagnosis"]),
            "icd9_procedure": len(parsed["icd9_procedure"]),
            "icd10_diagnosis": len(parsed["icd10_diagnosis"]),
            "icd10_procedure": len(parsed["icd10_procedure"]),
        },
    )
    return parsed


def lookup_description(parsed_codebooks: Dict[str, dict], source_type: str, version: Optional[int], raw_code: str) -> Tuple[Optional[str], str]:
    code = normalize_code(raw_code)
    if source_type == "diagnosis":
        if version == 9:
            entry = parsed_codebooks["icd9_diagnosis"].get(code)
            return (entry.get("long") if entry else None), "icd9_long"
        if version == 10:
            entry = parsed_codebooks["icd10_diagnosis"].get(code)
            return (entry.get("long") if entry else None), "icd10_description"
    elif source_type == "procedure":
        if version == 9:
            entry = parsed_codebooks["icd9_procedure"].get(code)
            return (entry.get("long") if entry else None), "icd9_long"
        if version == 10:
            entry = parsed_codebooks["icd10_procedure"].get(code)
            return (entry.get("long_title") if entry else None), "icd10_pcs_long_title"
    return None, "missing_codebook"


def code_system_name(source_type: str, version: Optional[int]) -> str:
    if source_type == "medication":
        return "RAW_DRUG_NAME"
    if version == 9:
        return "ICD9"
    if version == 10:
        return "ICD10"
    return "UNKNOWN"


def stable_medication_id(dataset: str, medication_payload: dict) -> str:
    digest = hashlib.sha1(
        json.dumps(medication_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset}_med_{digest}"


def build_medication_aliases(medication: object) -> Tuple[Optional[str], List[dict], dict]:
    aliases: List[dict] = []
    seen: Set[str] = set()

    def add_alias(source: str, value: Optional[str]) -> None:
        normalized = normalize_text(value)
        if normalized is None or normalized in seen:
            return
        seen.add(normalized)
        aliases.append({"source": source, "text": normalized})

    metadata = {
        "ndc": None,
        "gsn": None,
        "formulary_drug_cd": None,
        "drug_name_generic": None,
        "drug_name_poe": None,
    }

    if isinstance(medication, str):
        add_alias("ehr_medication_name", medication)
        return aliases[0]["text"] if aliases else None, aliases, metadata

    if not isinstance(medication, dict):
        return None, aliases, metadata

    metadata["ndc"] = normalize_ndc(medication.get("ndc"))
    metadata["gsn"] = normalize_text(medication.get("gsn"))
    metadata["formulary_drug_cd"] = normalize_text(medication.get("formulary_drug_cd"))
    metadata["drug_name_generic"] = normalize_text(medication.get("drug_name_generic"))
    metadata["drug_name_poe"] = normalize_text(medication.get("drug_name_poe"))

    add_alias("ehr_medication_name", medication.get("drug"))
    add_alias("ehr_medication_generic_name", medication.get("drug_name_generic"))
    add_alias("ehr_medication_poe_name", medication.get("drug_name_poe"))

    primary_text = aliases[0]["text"] if aliases else None
    return primary_text, aliases, metadata


def extract_mentions_for_dataset(dataset: str, patients_path: Path, parsed_codebooks: Dict[str, dict], output_root: Path) -> dict:
    mention_records: Dict[str, dict] = {}
    counts = Counter()
    missing_examples: List[dict] = []

    for patient in iter_jsonl(patients_path):
        for visit in patient["visits"]:
            for diagnosis in visit.get("diagnoses", []):
                raw_code = diagnosis["code"]
                version = diagnosis.get("version")
                normalized_code = normalize_code(raw_code)
                description, text_source = lookup_description(parsed_codebooks, "diagnosis", version, raw_code)
                mention_id = f"{dataset}_diag_{code_system_name('diagnosis', version).lower()}_{normalized_code}"
                record = {
                    "mention_id": mention_id,
                    "dataset": dataset,
                    "source_type": "diagnosis",
                    "code_system": code_system_name("diagnosis", version),
                    "raw_code": raw_code,
                    "normalized_code": normalized_code,
                    "mention_text": description,
                    "text_source": text_source,
                }
                mention_records[mention_id] = record
                counts["diagnosis_total"] += 1
                if description is None:
                    counts["diagnosis_missing_text"] += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(record)

            for procedure in visit.get("procedures", []):
                raw_code = procedure["code"]
                version = procedure.get("version")
                normalized_code = normalize_code(raw_code)
                description, text_source = lookup_description(parsed_codebooks, "procedure", version, raw_code)
                mention_id = f"{dataset}_proc_{code_system_name('procedure', version).lower()}_{normalized_code}"
                record = {
                    "mention_id": mention_id,
                    "dataset": dataset,
                    "source_type": "procedure",
                    "code_system": code_system_name("procedure", version),
                    "raw_code": raw_code,
                    "normalized_code": normalized_code,
                    "mention_text": description,
                    "text_source": text_source,
                }
                mention_records[mention_id] = record
                counts["procedure_total"] += 1
                if description is None:
                    counts["procedure_missing_text"] += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(record)

            for medication in visit.get("medications", []):
                medication_text, aliases, medication_metadata = build_medication_aliases(medication)
                if medication_text is None:
                    continue
                medication_payload = {
                    "mention_text": medication_text,
                    "text_aliases": aliases,
                    **medication_metadata,
                }
                mention_id = stable_medication_id(dataset, medication_payload)
                mention_records[mention_id] = {
                    "mention_id": mention_id,
                    "dataset": dataset,
                    "source_type": "medication",
                    "code_system": "RAW_DRUG_NAME",
                    "raw_code": None,
                    "normalized_code": None,
                    "mention_text": medication_text,
                    "text_source": aliases[0]["source"] if aliases else "ehr_medication_name",
                    "text_aliases": aliases,
                    **medication_metadata,
                }
                counts["medication_total"] += 1
                if medication_metadata["ndc"] is not None:
                    counts["medication_with_ndc"] += 1

    rows = sorted(mention_records.values(), key=lambda row: (row["source_type"], row["mention_id"]))
    mentions_dir = output_root / "mentions"
    write_jsonl(mentions_dir / f"{dataset}_unique_mentions.jsonl", rows)

    type_counter = Counter(row["source_type"] for row in rows)
    stats = {
        "dataset": dataset,
        "num_unique_mentions": len(rows),
        "num_unique_by_type": dict(sorted(type_counter.items())),
        "raw_occurrence_counts": dict(sorted(counts.items())),
        "num_unique_missing_text": sum(1 for row in rows if row["mention_text"] is None),
        "sample_missing_text_mentions": missing_examples,
    }
    write_json(mentions_dir / f"{dataset}_mention_stats.json", stats)
    return stats


def main() -> None:
    args = parse_args()
    parsed_codebooks = parse_codebooks(args.assets_root, args.output_root)

    summary = {
        "codebooks": {
            "icd9_diagnosis": len(parsed_codebooks["icd9_diagnosis"]),
            "icd9_procedure": len(parsed_codebooks["icd9_procedure"]),
            "icd10_diagnosis": len(parsed_codebooks["icd10_diagnosis"]),
            "icd10_procedure": len(parsed_codebooks["icd10_procedure"]),
        }
    }

    if args.dataset in {"mimic3", "both"}:
        summary["mimic3"] = extract_mentions_for_dataset(
            "mimic3",
            args.patients_root / "mimic3" / "patients_visits.jsonl",
            parsed_codebooks,
            args.output_root,
        )
    if args.dataset in {"mimic4", "both"}:
        summary["mimic4"] = extract_mentions_for_dataset(
            "mimic4",
            args.patients_root / "mimic4" / "patients_visits.jsonl",
            parsed_codebooks,
            args.output_root,
        )

    write_json(args.output_root / "mapping_input_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
