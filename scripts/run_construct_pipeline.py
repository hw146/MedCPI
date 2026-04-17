#!/usr/bin/env python3
"""Rebuild the MedCPI Construct stage with shared task schemas."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from path_roots import ALIGNED_ROOT, CONSTRUCT_ROOT, PROJECT_ROOT, SCRIPTS_ROOT

TASKS = ["mortality", "readmission_30d", "t2dm_onset", "cad_onset"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "both"], default="both")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--skip-aligned", action="store_true")
    parser.add_argument("--skip-inventory", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def selected_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "both":
        return ["mimic3", "mimic4"]
    return [dataset_arg]


def run_command(args: Iterable[str]) -> None:
    cmd = [str(arg) for arg in args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def clean_task_outputs(dataset: str, task: str) -> None:
    task_dir = CONSTRUCT_ROOT / dataset / task
    stale_paths = [
        task_dir / "train_anchor_concepts.json",
        task_dir / "train_anchor_concepts.jsonl",
        task_dir / "relation_inventory.json",
        task_dir / "relation_schema.template.json",
        task_dir / "relation_schema.json",
        task_dir / "construct_stats.json",
        task_dir / "concept_mkg_nodes.jsonl",
        task_dir / "concept_mkg_edges.jsonl",
    ]
    for path in stale_paths:
        if path.exists():
            path.unlink()
    llm_trace_dir = task_dir / "llm_schema_runs"
    if llm_trace_dir.exists():
        for child in sorted(llm_trace_dir.glob("*")):
            child.unlink()
        llm_trace_dir.rmdir()


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args.dataset)

    if not args.skip_aligned:
        for dataset in datasets:
            aligned_path = ALIGNED_ROOT / dataset / "patients_visits_aligned.jsonl"
            if aligned_path.exists() and not args.clean:
                continue
            run_command(
                [
                    sys.executable,
                    SCRIPTS_ROOT / "build_aligned_patient_ehr.py",
                    "--dataset",
                    dataset,
                ]
            )

    for dataset in datasets:
        for task in args.tasks:
            shared_schema_path = CONSTRUCT_ROOT / "shared" / task / "relation_schema.json"
            if not shared_schema_path.exists():
                raise FileNotFoundError(
                    "Missing shared task schema: {0}".format(shared_schema_path)
                )
            if args.clean:
                clean_task_outputs(dataset, task)

            run_command(
                [
                    sys.executable,
                    SCRIPTS_ROOT / "extract_task_train_concepts.py",
                    "--dataset",
                    dataset,
                    "--task",
                    task,
                ]
            )
            if not args.skip_inventory:
                run_command(
                    [
                        sys.executable,
                        SCRIPTS_ROOT / "export_relation_inventory.py",
                        "--dataset",
                        dataset,
                        "--task",
                        task,
                    ]
                )
            run_command(
                [
                    sys.executable,
                    SCRIPTS_ROOT / "build_concept_mkg.py",
                    "--dataset",
                    dataset,
                    "--task",
                    task,
                ]
            )


if __name__ == "__main__":
    main()
