#!/usr/bin/env python3
"""Build patient-level ordered-visit JSONL files from raw MIMIC tables.

This script is intentionally stdlib-only so it can run in minimal environments.
It reads the raw MIMIC CSVs without modifying them and writes processed outputs
under a separate directory.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from path_roots import MIMIC3_RAW_ROOT, MIMIC4_RAW_ROOT, PREPROCESSED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mimic3", "mimic4", "both"],
        default="both",
        help="Which dataset(s) to preprocess.",
    )
    parser.add_argument(
        "--mimic3-dir",
        type=Path,
        default=MIMIC3_RAW_ROOT,
        help="Path to the raw MIMIC-III directory.",
    )
    parser.add_argument(
        "--mimic4-dir",
        type=Path,
        default=MIMIC4_RAW_ROOT,
        help="Path to the raw MIMIC-IV directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PREPROCESSED_ROOT,
        help="Directory for generated JSONL/stat files.",
    )
    return parser.parse_args()


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def normalize_medication(value: Optional[str]) -> Optional[str]:
    value = normalize_text(value)
    if value is None:
        return None
    return " ".join(value.upper().split())


def normalize_medication_code(value: Optional[str]) -> Optional[str]:
    value = normalize_text(value)
    if value is None:
        return None
    return value.upper()


def normalize_ndc(value: Optional[str]) -> Optional[str]:
    value = normalize_text(value)
    if value is None:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits or None


def parse_time(value: Optional[str]) -> Optional[datetime]:
    value = normalize_text(value)
    if value is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError("Unsupported datetime format: {!r}".format(value))


def read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def build_patient_counts(
    admissions_path: Path,
    patient_col: str,
    visit_col: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    seen_visits: Set[Tuple[str, str]] = set()
    for row in read_csv_rows(admissions_path):
        patient_id = normalize_text(row.get(patient_col))
        visit_id = normalize_text(row.get(visit_col))
        if patient_id is None or visit_id is None:
            continue
        key = (patient_id, visit_id)
        if key in seen_visits:
            continue
        seen_visits.add(key)
        counts[patient_id] += 1
    return counts


def build_core_visit_structures(
    admissions_path: Path,
    patient_col: str,
    visit_col: str,
    admit_col: str,
    discharge_col: str,
    death_col: str,
    expire_col: str,
    eligible_patients: Set[str],
) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    visits: Dict[str, dict] = {}
    patient_visits: Dict[str, List[str]] = defaultdict(list)

    for row in read_csv_rows(admissions_path):
        patient_id = normalize_text(row.get(patient_col))
        visit_id = normalize_text(row.get(visit_col))
        if patient_id is None or visit_id is None or patient_id not in eligible_patients:
            continue
        if visit_id in visits:
            continue

        visits[visit_id] = {
            "dataset_patient_id": patient_id,
            "visit_id": visit_id,
            "admittime": normalize_text(row.get(admit_col)),
            "dischtime": normalize_text(row.get(discharge_col)),
            "deathtime": normalize_text(row.get(death_col)),
            "hospital_expire_flag": int(normalize_text(row.get(expire_col)) or "0"),
            "diagnoses": set(),
            "procedures": set(),
            "medications": set(),
        }
        patient_visits[patient_id].append(visit_id)

    return visits, patient_visits


def add_diagnoses(
    diagnoses_path: Path,
    patient_col: str,
    visit_col: str,
    code_col: str,
    version_col: Optional[str],
    eligible_patients: Set[str],
    visits: Dict[str, dict],
    default_version: Optional[int],
) -> None:
    for row in read_csv_rows(diagnoses_path):
        patient_id = normalize_text(row.get(patient_col))
        visit_id = normalize_text(row.get(visit_col))
        code = normalize_text(row.get(code_col))
        if (
            patient_id is None
            or visit_id is None
            or code is None
            or patient_id not in eligible_patients
            or visit_id not in visits
        ):
            continue
        version_value = default_version
        if version_col is not None:
            raw_version = normalize_text(row.get(version_col))
            version_value = int(raw_version) if raw_version is not None else None
        visits[visit_id]["diagnoses"].add((code, version_value))


def add_procedures(
    procedures_path: Path,
    patient_col: str,
    visit_col: str,
    code_col: str,
    version_col: Optional[str],
    eligible_patients: Set[str],
    visits: Dict[str, dict],
    default_version: Optional[int],
) -> None:
    for row in read_csv_rows(procedures_path):
        patient_id = normalize_text(row.get(patient_col))
        visit_id = normalize_text(row.get(visit_col))
        code = normalize_text(row.get(code_col))
        if (
            patient_id is None
            or visit_id is None
            or code is None
            or patient_id not in eligible_patients
            or visit_id not in visits
        ):
            continue
        version_value = default_version
        if version_col is not None:
            raw_version = normalize_text(row.get(version_col))
            version_value = int(raw_version) if raw_version is not None else None
        visits[visit_id]["procedures"].add((code, version_value))


def add_medications(
    medications_path: Path,
    patient_col: str,
    visit_col: str,
    med_col: str,
    generic_col: Optional[str],
    poe_col: Optional[str],
    formulary_col: Optional[str],
    gsn_col: Optional[str],
    ndc_col: Optional[str],
    eligible_patients: Set[str],
    visits: Dict[str, dict],
) -> None:
    for row in read_csv_rows(medications_path):
        patient_id = normalize_text(row.get(patient_col))
        visit_id = normalize_text(row.get(visit_col))
        medication_name = normalize_medication(row.get(med_col))
        generic_name = normalize_medication(row.get(generic_col)) if generic_col else None
        poe_name = normalize_medication(row.get(poe_col)) if poe_col else None
        formulary_drug_cd = normalize_medication_code(row.get(formulary_col)) if formulary_col else None
        gsn = normalize_medication_code(row.get(gsn_col)) if gsn_col else None
        ndc = normalize_ndc(row.get(ndc_col)) if ndc_col else None
        medication_key = medication_name or generic_name or poe_name
        if (
            patient_id is None
            or visit_id is None
            or medication_key is None
            or patient_id not in eligible_patients
            or visit_id not in visits
        ):
            continue
        visits[visit_id]["medications"].add(
            (
                medication_name,
                generic_name,
                poe_name,
                formulary_drug_cd,
                gsn,
                ndc,
            )
        )


def sort_code_tuples(items: Set[Tuple[str, Optional[int]]]) -> List[dict]:
    ordered = sorted(items, key=lambda item: ((item[1] is None), item[1], item[0]))
    return [{"code": code, "version": version} for code, version in ordered]


def sort_medication_records(items: Set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]) -> List[dict]:
    ordered = sorted(
        items,
        key=lambda item: tuple(value or "" for value in item),
    )
    records = []
    for medication_name, generic_name, poe_name, formulary_drug_cd, gsn, ndc in ordered:
        records.append(
            {
                "drug": medication_name,
                "drug_name_generic": generic_name,
                "drug_name_poe": poe_name,
                "formulary_drug_cd": formulary_drug_cd,
                "gsn": gsn,
                "ndc": ndc,
            }
        )
    return records


def visit_sort_key(visit_record: dict) -> Tuple[datetime, str]:
    admitted = parse_time(visit_record["admittime"])
    if admitted is None:
        admitted = datetime.max
    return admitted, visit_record["visit_id"]


def materialize_patients_jsonl(
    dataset_name: str,
    visits: Dict[str, dict],
    patient_visits: Dict[str, List[str]],
    output_jsonl: Path,
) -> dict:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    num_patients = 0
    total_visits = 0
    total_diagnoses = 0
    total_procedures = 0
    total_medications = 0

    with output_jsonl.open("w") as handle:
        for patient_id in sorted(patient_visits, key=lambda value: int(value)):
            ordered_visits = [visits[visit_id] for visit_id in patient_visits[patient_id]]
            ordered_visits.sort(key=visit_sort_key)

            visit_payloads = []
            for visit in ordered_visits:
                diagnoses = sort_code_tuples(visit["diagnoses"])
                procedures = sort_code_tuples(visit["procedures"])
                medications = sort_medication_records(visit["medications"])
                total_visits += 1
                total_diagnoses += len(diagnoses)
                total_procedures += len(procedures)
                total_medications += len(medications)

                visit_payloads.append(
                    {
                        "visit_id": visit["visit_id"],
                        "admittime": visit["admittime"],
                        "dischtime": visit["dischtime"],
                        "deathtime": visit["deathtime"],
                        "hospital_expire_flag": visit["hospital_expire_flag"],
                        "diagnoses": diagnoses,
                        "procedures": procedures,
                        "medications": medications,
                    }
                )

            payload = {
                "dataset": dataset_name,
                "patient_id": patient_id,
                "num_visits": len(visit_payloads),
                "visits": visit_payloads,
            }
            handle.write(json.dumps(payload) + "\n")
            num_patients += 1

    average_visits = (float(total_visits) / num_patients) if num_patients else 0.0
    return {
        "dataset": dataset_name,
        "num_patients": num_patients,
        "num_visits": total_visits,
        "avg_visits_per_patient": round(average_visits, 4),
        "total_diagnosis_entries": total_diagnoses,
        "total_procedure_entries": total_procedures,
        "total_medication_entries": total_medications,
        "output_jsonl": str(output_jsonl),
    }


def write_stats(stats_path: Path, stats: dict) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def preprocess_mimic3(input_dir: Path, output_root: Path) -> dict:
    patient_counts = build_patient_counts(
        admissions_path=input_dir / "ADMISSIONS.csv",
        patient_col="SUBJECT_ID",
        visit_col="HADM_ID",
    )
    eligible_patients = {patient_id for patient_id, count in patient_counts.items() if count >= 2}

    visits, patient_visits = build_core_visit_structures(
        admissions_path=input_dir / "ADMISSIONS.csv",
        patient_col="SUBJECT_ID",
        visit_col="HADM_ID",
        admit_col="ADMITTIME",
        discharge_col="DISCHTIME",
        death_col="DEATHTIME",
        expire_col="HOSPITAL_EXPIRE_FLAG",
        eligible_patients=eligible_patients,
    )
    add_diagnoses(
        diagnoses_path=input_dir / "DIAGNOSES_ICD.csv",
        patient_col="SUBJECT_ID",
        visit_col="HADM_ID",
        code_col="ICD9_CODE",
        version_col=None,
        eligible_patients=eligible_patients,
        visits=visits,
        default_version=9,
    )
    add_procedures(
        procedures_path=input_dir / "PROCEDURES_ICD.csv",
        patient_col="SUBJECT_ID",
        visit_col="HADM_ID",
        code_col="ICD9_CODE",
        version_col=None,
        eligible_patients=eligible_patients,
        visits=visits,
        default_version=9,
    )
    add_medications(
        medications_path=input_dir / "PRESCRIPTIONS.csv",
        patient_col="SUBJECT_ID",
        visit_col="HADM_ID",
        med_col="DRUG",
        generic_col="DRUG_NAME_GENERIC",
        poe_col="DRUG_NAME_POE",
        formulary_col="FORMULARY_DRUG_CD",
        gsn_col="GSN",
        ndc_col="NDC",
        eligible_patients=eligible_patients,
        visits=visits,
    )

    dataset_output = output_root / "mimic3"
    stats = materialize_patients_jsonl(
        dataset_name="mimic3",
        visits=visits,
        patient_visits=patient_visits,
        output_jsonl=dataset_output / "patients_visits.jsonl",
    )
    stats["eligible_patient_threshold"] = 2
    write_stats(dataset_output / "cohort_stats.json", stats)
    return stats


def preprocess_mimic4(input_dir: Path, output_root: Path) -> dict:
    patient_counts = build_patient_counts(
        admissions_path=input_dir / "admissions.csv",
        patient_col="subject_id",
        visit_col="hadm_id",
    )
    eligible_patients = {patient_id for patient_id, count in patient_counts.items() if count >= 2}

    visits, patient_visits = build_core_visit_structures(
        admissions_path=input_dir / "admissions.csv",
        patient_col="subject_id",
        visit_col="hadm_id",
        admit_col="admittime",
        discharge_col="dischtime",
        death_col="deathtime",
        expire_col="hospital_expire_flag",
        eligible_patients=eligible_patients,
    )
    add_diagnoses(
        diagnoses_path=input_dir / "diagnoses_icd.csv",
        patient_col="subject_id",
        visit_col="hadm_id",
        code_col="icd_code",
        version_col="icd_version",
        eligible_patients=eligible_patients,
        visits=visits,
        default_version=None,
    )
    add_procedures(
        procedures_path=input_dir / "procedures_icd.csv",
        patient_col="subject_id",
        visit_col="hadm_id",
        code_col="icd_code",
        version_col="icd_version",
        eligible_patients=eligible_patients,
        visits=visits,
        default_version=None,
    )
    add_medications(
        medications_path=input_dir / "prescriptions.csv",
        patient_col="subject_id",
        visit_col="hadm_id",
        med_col="drug",
        generic_col=None,
        poe_col=None,
        formulary_col="formulary_drug_cd",
        gsn_col="gsn",
        ndc_col="ndc",
        eligible_patients=eligible_patients,
        visits=visits,
    )

    dataset_output = output_root / "mimic4"
    stats = materialize_patients_jsonl(
        dataset_name="mimic4",
        visits=visits,
        patient_visits=patient_visits,
        output_jsonl=dataset_output / "patients_visits.jsonl",
    )
    stats["eligible_patient_threshold"] = 2
    write_stats(dataset_output / "cohort_stats.json", stats)
    return stats


def main() -> None:
    args = parse_args()
    stats = {}

    if args.dataset in ("mimic3", "both"):
        stats["mimic3"] = preprocess_mimic3(args.mimic3_dir, args.output_root)
    if args.dataset in ("mimic4", "both"):
        stats["mimic4"] = preprocess_mimic4(args.mimic4_dir, args.output_root)

    write_stats(args.output_root / "preprocess_summary.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
