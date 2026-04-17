#!/usr/bin/env python3
"""Create deterministic patient-level train/valid/test splits for each task."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from path_roots import SPLITS_ROOT, TASKS_ROOT


TASK_FILES = {
    "mortality": "mortality_instances.jsonl",
    "readmission_30d": "readmission_30d_instances.jsonl",
    "t2dm_onset": "t2dm_onset_instances.jsonl",
    "cad_onset": "cad_onset_instances.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mimic3", "mimic4", "both"],
        default="mimic3",
        help="Which dataset(s) to split.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=TASKS_ROOT,
        help="Root directory containing task instance JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SPLITS_ROOT,
        help="Root directory for split manifests and split JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for deterministic patient shuffling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def filter_rows_for_task(task_name: str, rows: List[dict]) -> List[dict]:
    if task_name in {"t2dm_onset", "cad_onset"}:
        return [row for row in rows if row.get("eligible") is True and row.get("label") is not None]
    return rows


def make_patient_split(
    patient_ids: List[str],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> Dict[str, List[str]]:
    shuffled = list(patient_ids)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)

    return {
        "train": sorted(shuffled[:n_train], key=int),
        "valid": sorted(shuffled[n_train : n_train + n_valid], key=int),
        "test": sorted(shuffled[n_train + n_valid :], key=int),
    }


def build_split_stats(split_rows: Dict[str, List[dict]]) -> dict:
    stats = {}
    for split_name, rows in split_rows.items():
        label_counter = Counter(row["label"] for row in rows)
        stats[split_name] = {
            "num_instances": len(rows),
            "num_patients": len({row["patient_id"] for row in rows}),
            "label_distribution": {str(key): value for key, value in sorted(label_counter.items(), key=lambda x: str(x[0]))},
        }
    return stats


def process_task(
    dataset: str,
    task_name: str,
    input_path: Path,
    output_root: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict:
    raw_rows = list(iter_jsonl(input_path))
    rows = filter_rows_for_task(task_name, raw_rows)
    patient_ids = sorted({row["patient_id"] for row in rows}, key=int)
    split_patients = make_patient_split(patient_ids, seed, train_ratio, valid_ratio)

    split_sets = {name: set(ids) for name, ids in split_patients.items()}
    split_rows = {
        name: [row for row in rows if row["patient_id"] in patient_set]
        for name, patient_set in split_sets.items()
    }

    task_output_dir = output_root / dataset / task_name
    for split_name, payload in split_rows.items():
        write_jsonl(task_output_dir / f"{split_name}.jsonl", payload)

    manifest = {
        "dataset": dataset,
        "task": task_name,
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "num_source_rows": len(raw_rows),
        "num_rows_after_task_filter": len(rows),
        "split_patients": split_patients,
        "split_stats": build_split_stats(split_rows),
    }
    write_json(task_output_dir / "splits.json", manifest)
    return manifest


def process_dataset(
    dataset: str,
    input_root: Path,
    output_root: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> dict:
    dataset_summary = {"dataset": dataset}
    for task_name, filename in TASK_FILES.items():
        dataset_summary[task_name] = process_task(
            dataset=dataset,
            task_name=task_name,
            input_path=input_root / dataset / filename,
            output_root=output_root,
            seed=seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
    return dataset_summary


def main() -> None:
    args = parse_args()
    summary = {}
    if args.dataset in {"mimic3", "both"}:
        summary["mimic3"] = process_dataset(
            dataset="mimic3",
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )
    if args.dataset in {"mimic4", "both"}:
        summary["mimic4"] = process_dataset(
            dataset="mimic4",
            input_root=args.input_root,
            output_root=args.output_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )

    write_json(args.output_root / "split_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
