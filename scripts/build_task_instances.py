#!/usr/bin/env python3
"""Build task instances from patient-visit JSONL files."""

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from path_roots import PREPROCESSED_ROOT, TASKS_ROOT


OBSERVATION_DAYS = 365
PREDICTION_DAYS = 365
READMISSION_WINDOW_DAYS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mimic3", "mimic4", "both"],
        default="mimic3",
        help="Which preprocessed dataset(s) to convert into task instances.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PREPROCESSED_ROOT,
        help="Root directory containing patients_visits.jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TASKS_ROOT,
        help="Root directory where task instances will be written.",
    )
    return parser.parse_args()


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_icd_code(code: str) -> str:
    return code.replace(".", "").strip().upper()


def is_t2dm(code: str, version: Optional[int], dataset: str) -> bool:
    normalized = normalize_icd_code(code)
    if dataset == "mimic3":
        return version == 9 and normalized.startswith("250") and normalized[-1:] in {"0", "2"}
    return version == 10 and normalized.startswith("E11")


def is_cad(code: str, version: Optional[int], dataset: str) -> bool:
    normalized = normalize_icd_code(code)
    if dataset == "mimic3":
        if version != 9 or len(normalized) < 3 or not normalized[:3].isdigit():
            return False
        prefix = int(normalized[:3])
        return 410 <= prefix <= 414
    return version == 10 and normalized[:3] in {"I20", "I21", "I22", "I23", "I24", "I25"}


def first_disease_time(
    visits: List[dict],
    dataset: str,
    matcher,
) -> Optional[datetime]:
    for visit in visits:
        visit_time = parse_time(visit["admittime"])
        if visit_time is None:
            continue
        for diagnosis in visit.get("diagnoses", []):
            if matcher(diagnosis["code"], diagnosis.get("version"), dataset):
                return visit_time
    return None


def build_visit_level_instances(patient: dict) -> Tuple[List[dict], List[dict]]:
    visits = patient["visits"]
    mortality_instances: List[dict] = []
    readmission_instances: List[dict] = []

    for index, visit in enumerate(visits):
        if index == 0:
            continue

        admit_time = parse_time(visit["admittime"])
        discharge_time = parse_time(visit["dischtime"])
        death_time = parse_time(visit.get("deathtime"))

        mortality_label = 0
        if admit_time is not None and discharge_time is not None and death_time is not None:
            if admit_time <= death_time <= discharge_time:
                mortality_label = 1
        elif int(visit.get("hospital_expire_flag", 0)) == 1:
            mortality_label = 1

        context_visit_ids = [item["visit_id"] for item in visits[:index]]
        mortality_instances.append(
            {
                "task": "mortality",
                "granularity": "visit",
                "dataset": patient["dataset"],
                "patient_id": patient["patient_id"],
                "target_visit_id": visit["visit_id"],
                "target_visit_index": index,
                "prediction_time": visit["admittime"],
                "context_visit_ids": context_visit_ids,
                "context_num_visits": len(context_visit_ids),
                "label": mortality_label,
                "label_source": {
                    "deathtime": visit.get("deathtime"),
                    "hospital_expire_flag": int(visit.get("hospital_expire_flag", 0)),
                },
            }
        )

        next_admit_time = None
        days_to_next_admission = None
        if index + 1 < len(visits):
            next_admit_time = visits[index + 1]["admittime"]
            if discharge_time is not None:
                next_time = parse_time(next_admit_time)
                if next_time is not None:
                    days_to_next_admission = (next_time - discharge_time).total_seconds() / 86400.0

        readmission_label = int(
            days_to_next_admission is not None and days_to_next_admission <= READMISSION_WINDOW_DAYS
        )
        readmission_instances.append(
            {
                "task": "readmission_30d",
                "granularity": "visit",
                "dataset": patient["dataset"],
                "patient_id": patient["patient_id"],
                "target_visit_id": visit["visit_id"],
                "target_visit_index": index,
                "prediction_time": visit["dischtime"],
                "context_visit_ids": context_visit_ids,
                "context_num_visits": len(context_visit_ids),
                "label": readmission_label,
                "label_source": {
                    "next_visit_id": visits[index + 1]["visit_id"] if index + 1 < len(visits) else None,
                    "next_admittime": next_admit_time,
                    "days_to_next_admission": days_to_next_admission,
                },
            }
        )

    return mortality_instances, readmission_instances


def build_onset_instance(patient: dict, task_name: str, matcher) -> dict:
    visits = patient["visits"]
    first_visit_time = parse_time(visits[0]["admittime"])
    last_visit_time = parse_time(visits[-1]["dischtime"] or visits[-1]["admittime"])

    observation_end = first_visit_time + timedelta(days=OBSERVATION_DAYS)
    prediction_end = observation_end + timedelta(days=PREDICTION_DAYS)

    first_match_time = first_disease_time(visits, patient["dataset"], matcher)
    followup_sufficient = last_visit_time is not None and last_visit_time >= prediction_end

    eligible = True
    exclusion_reason = None
    if first_match_time is not None and first_match_time <= observation_end:
        eligible = False
        exclusion_reason = "prevalent_case_before_observation_end"
    elif not followup_sufficient:
        eligible = False
        exclusion_reason = "insufficient_followup_for_prediction_window"

    label = None
    if eligible:
        label = int(first_match_time is not None and observation_end < first_match_time <= prediction_end)

    observation_visit_ids = []
    for visit in visits:
        visit_time = parse_time(visit["admittime"])
        if visit_time is not None and visit_time <= observation_end:
            observation_visit_ids.append(visit["visit_id"])

    return {
        "task": task_name,
        "granularity": "patient",
        "dataset": patient["dataset"],
        "patient_id": patient["patient_id"],
        "prediction_time": observation_end.strftime("%Y-%m-%d %H:%M:%S"),
        "observation_window_days": OBSERVATION_DAYS,
        "prediction_window_days": PREDICTION_DAYS,
        "observation_start": visits[0]["admittime"],
        "observation_end": observation_end.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_end": prediction_end.strftime("%Y-%m-%d %H:%M:%S"),
        "observation_visit_ids": observation_visit_ids,
        "observation_num_visits": len(observation_visit_ids),
        "eligible": eligible,
        "exclusion_reason": exclusion_reason,
        "label": label,
        "first_disease_time": first_match_time.strftime("%Y-%m-%d %H:%M:%S") if first_match_time else None,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def summarize_instances(rows: List[dict], include_eligibility: bool = False) -> dict:
    stats = {
        "num_instances": len(rows),
    }
    if rows:
        label_counter = Counter(row["label"] for row in rows)
        stats["label_distribution"] = {str(key): value for key, value in sorted(label_counter.items(), key=lambda x: str(x[0]))}
    if include_eligibility:
        eligible_rows = [row for row in rows if row["eligible"]]
        exclusion_counter = Counter(row["exclusion_reason"] for row in rows if not row["eligible"])
        stats["num_eligible"] = len(eligible_rows)
        stats["num_excluded"] = len(rows) - len(eligible_rows)
        stats["eligible_label_distribution"] = {
            str(key): value for key, value in sorted(Counter(row["label"] for row in eligible_rows).items(), key=lambda x: str(x[0]))
        }
        stats["exclusion_reasons"] = dict(sorted(exclusion_counter.items()))
    return stats


def write_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


def process_dataset(dataset: str, input_root: Path, output_root: Path) -> dict:
    input_path = input_root / dataset / "patients_visits.jsonl"
    output_dir = output_root / dataset

    mortality_rows: List[dict] = []
    readmission_rows: List[dict] = []
    t2dm_rows: List[dict] = []
    cad_rows: List[dict] = []

    for patient in iter_jsonl(input_path):
        patient_mortality, patient_readmission = build_visit_level_instances(patient)
        mortality_rows.extend(patient_mortality)
        readmission_rows.extend(patient_readmission)
        t2dm_rows.append(build_onset_instance(patient, "t2dm_onset", is_t2dm))
        cad_rows.append(build_onset_instance(patient, "cad_onset", is_cad))

    write_jsonl(output_dir / "mortality_instances.jsonl", mortality_rows)
    write_jsonl(output_dir / "readmission_30d_instances.jsonl", readmission_rows)
    write_jsonl(output_dir / "t2dm_onset_instances.jsonl", t2dm_rows)
    write_jsonl(output_dir / "cad_onset_instances.jsonl", cad_rows)

    summary = {
        "dataset": dataset,
        "mortality": summarize_instances(mortality_rows),
        "readmission_30d": summarize_instances(readmission_rows),
        "t2dm_onset": summarize_instances(t2dm_rows, include_eligibility=True),
        "cad_onset": summarize_instances(cad_rows, include_eligibility=True),
    }
    write_stats(output_dir / "task_stats.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summaries = {}
    if args.dataset in {"mimic3", "both"}:
        summaries["mimic3"] = process_dataset("mimic3", args.input_root, args.output_root)
    if args.dataset in {"mimic4", "both"}:
        summaries["mimic4"] = process_dataset("mimic4", args.input_root, args.output_root)

    write_stats(args.output_root / "task_summary.json", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
