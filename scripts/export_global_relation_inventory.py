#!/usr/bin/env python3
"""Export a global UMLS relation inventory for task-only schema induction."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from path_roots import CONSTRUCT_ROOT, UMLS_ROOT

DEFAULT_OUTPUT_PATH = CONSTRUCT_ROOT / "shared" / "global_relation_inventory.json"
MRREL_PATH = UMLS_ROOT / "MRREL.RRF"
MRCONSO_PATH = UMLS_ROOT / "MRCONSO.RRF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrrel-path", type=Path, default=MRREL_PATH)
    parser.add_argument("--mrconso-path", type=Path, default=MRCONSO_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-samples-per-relation", type=int, default=5)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    relation_counts = Counter()
    relation_examples = defaultdict(list)
    sample_cuis: Set[str] = set()

    with args.mrrel_path.open("r", errors="replace") as handle:
        for line in handle:
            row = parse_mrrel_row(line)
            if row is None:
                continue
            if row["suppress"] not in {"", "N"}:
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
                }
            )
        relations.append(
            {
                "raw_relation": relation,
                "count": count,
                "samples": samples,
            }
        )

    payload = {
        "dataset": "shared",
        "scope": "global_umls",
        "mrrel_path": str(args.mrrel_path),
        "num_raw_relations": len(relations),
        "relations": relations,
    }
    write_json(args.output_path, payload)
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "num_raw_relations": len(relations),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
