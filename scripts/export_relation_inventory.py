#!/usr/bin/env python3
"""Export task-specific UMLS relation inventory for Construct."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from path_roots import CONSTRUCT_ROOT as DEFAULT_CONSTRUCT_ROOT, UMLS_ROOT

MRREL_PATH = UMLS_ROOT / "MRREL.RRF"
MRCONSO_PATH = UMLS_ROOT / "MRCONSO.RRF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4"], required=True)
    parser.add_argument("--task", choices=["mortality", "readmission_30d", "t2dm_onset", "cad_onset"], required=True)
    parser.add_argument("--construct-root", type=Path, default=DEFAULT_CONSTRUCT_ROOT)
    parser.add_argument("--mrrel-path", type=Path, default=MRREL_PATH)
    parser.add_argument("--mrconso-path", type=Path, default=MRCONSO_PATH)
    parser.add_argument("--max-samples-per-relation", type=int, default=5)
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


def load_anchor_concepts(path: Path) -> Set[str]:
    return {row["concept_id"] for row in iter_jsonl(path)}


def load_preferred_terms(mrconso_path: Path, cuis: Set[str]) -> Dict[str, str]:
    labels = {}
    scores = {}
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


def build_inventory(args: argparse.Namespace) -> Tuple[dict, dict]:
    task_dir = args.construct_root / args.dataset / args.task
    anchor_path = task_dir / "train_anchor_concepts.jsonl"
    inventory_path = task_dir / "relation_inventory.json"
    template_path = task_dir / "relation_schema.template.json"

    anchors = load_anchor_concepts(anchor_path)
    relation_counts = Counter()
    relation_examples = defaultdict(list)
    sample_cuis = set()

    with args.mrrel_path.open("r", errors="replace") as handle:
        for line in handle:
            row = parse_mrrel_row(line)
            if row is None:
                continue
            if row["suppress"] not in {"", "N"}:
                continue
            if row["cui1"] not in anchors and row["cui2"] not in anchors:
                continue

            relation = raw_relation_name(row)
            if not relation:
                continue

            relation_counts[relation] += 1
            if len(relation_examples[relation]) < args.max_samples_per_relation:
                relation_examples[relation].append(
                    {
                        "cui1": row["cui1"],
                        "cui2": row["cui2"],
                        "rel": row["rel"],
                        "rela": row["rela"],
                        "sab": row["sab"],
                        "anchor_side": "cui1" if row["cui1"] in anchors else "cui2",
                    }
                )
                sample_cuis.add(row["cui1"])
                sample_cuis.add(row["cui2"])

    labels = load_preferred_terms(args.mrconso_path, sample_cuis)
    relations = []
    for relation, count in sorted(relation_counts.items(), key=lambda item: (-item[1], item[0])):
        samples = []
        for sample in relation_examples[relation]:
            samples.append(
                {
                    "cui1": sample["cui1"],
                    "cui2": sample["cui2"],
                    "term1": labels.get(sample["cui1"]),
                    "term2": labels.get(sample["cui2"]),
                    "rel": sample["rel"],
                    "rela": sample["rela"],
                    "sab": sample["sab"],
                    "anchor_side": sample["anchor_side"],
                }
            )
        relations.append(
            {
                "raw_relation": relation,
                "count": count,
                "samples": samples,
            }
        )

    inventory = {
        "dataset": args.dataset,
        "task": args.task,
        "anchor_path": str(anchor_path),
        "mrrel_path": str(args.mrrel_path),
        "num_anchor_concepts": len(anchors),
        "num_raw_relations": len(relations),
        "relations": relations,
    }
    template = {
        "dataset": args.dataset,
        "task": args.task,
        "decision_source": "template",
        "notes": "Fill keep=true/false and canonical_relation before building Concept MKG.",
        "relations": [
            {
                "raw_relation": row["raw_relation"],
                "keep": False,
                "canonical_relation": row["raw_relation"],
                "rationale": "",
            }
            for row in relations
        ],
    }
    write_json(inventory_path, inventory)
    write_json(template_path, template)
    return inventory, {
        "inventory_path": str(inventory_path),
        "template_path": str(template_path),
        "num_raw_relations": len(relations),
    }


def main() -> None:
    args = parse_args()
    _, summary = build_inventory(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
