#!/usr/bin/env python3
"""Build source-specific UMLS candidate tables for diagnosis/procedure/medication mapping."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from path_roots import MAPPING_ROOT, RXNORM_ROOT, UMLS_ROOT

MRCONSO_PATH = UMLS_ROOT / "MRCONSO.RRF"
MRSTY_PATH = UMLS_ROOT / "MRSTY.RRF"
RXNCONSO_PATH = RXNORM_ROOT / "RXNCONSO.RRF"
RXNSAT_PATH = RXNORM_ROOT / "RXNSAT.RRF"
OUTPUT_DIR = MAPPING_ROOT / "umls_index"

DIAG_SEMTYPES = {
    "Acquired Abnormality",
    "Anatomical Abnormality",
    "Cell or Molecular Dysfunction",
    "Clinical Attribute",
    "Congenital Abnormality",
    "Disease or Syndrome",
    "Finding",
    "Injury or Poisoning",
    "Laboratory or Test Result",
    "Mental or Behavioral Dysfunction",
    "Neoplastic Process",
    "Pathologic Function",
    "Sign or Symptom",
}

PROC_SEMTYPES = {
    "Diagnostic Procedure",
    "Health Care Activity",
    "Laboratory Procedure",
    "Therapeutic or Preventive Procedure",
}

SOURCE_CONFIG = {
    "diagnosis": {
        "allowed_sabs": {"ICD9CM", "ICD10CM"},
        "allowed_semtypes": DIAG_SEMTYPES,
    },
    "procedure": {
        "allowed_sabs": {"ICD9CM", "ICD10PCS"},
        "allowed_semtypes": PROC_SEMTYPES,
    },
    "medication": {
        "allowed_sabs": {"RXNORM"},
        "allowed_semtypes": None,
    },
}

MEDICATION_TTYS = {
    "BD",
    "BN",
    "BPCK",
    "GPCK",
    "IN",
    "MS",
    "MIN",
    "PIN",
    "SBD",
    "SBDC",
    "SBDF",
    "SBDFP",
    "SBDG",
    "SCD",
    "SCDC",
    "SCDF",
    "SCDFP",
    "SCDG",
    "SCDGP",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ndc(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def load_relevant_semtypes(source_type: str, mrsty_path: Path) -> dict[str, list[str]]:
    allowed = SOURCE_CONFIG[source_type]["allowed_semtypes"]
    if allowed is None:
        raise ValueError("Medication uses load_semtypes_for_cuis instead of load_relevant_semtypes.")

    semtypes: dict[str, set[str]] = defaultdict(set)
    with mrsty_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            semtype_name = parts[3]
            if semtype_name in allowed:
                semtypes[cui].add(semtype_name)
    return {cui: sorted(values) for cui, values in semtypes.items()}


def load_semtypes_for_cuis(cuis: set[str], mrsty_path: Path) -> dict[str, list[str]]:
    semtypes: dict[str, set[str]] = defaultdict(set)
    with mrsty_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[0]
            if cui not in cuis:
                continue
            semtypes[cui].add(parts[3])
    return {cui: sorted(values) for cui, values in semtypes.items()}


def build_code_candidates(source_type: str, mrconso_path: Path, mrsty_path: Path) -> tuple[list[dict], dict]:
    config = SOURCE_CONFIG[source_type]
    semtypes = load_relevant_semtypes(source_type, mrsty_path)
    candidates: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    sab_counter: Counter[str] = Counter()
    semtype_counter: Counter[str] = Counter()

    with mrconso_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 18:
                continue

            (
                cui,
                lat,
                _ts,
                _lui,
                _stt,
                _sui,
                _ispref,
                _aui,
                _saui,
                _scui,
                _sdui,
                sab,
                tty,
                code,
                term_text,
                _srl,
                suppress,
                _cvf,
            ) = parts[:18]

            if lat != "ENG" or suppress != "N":
                continue
            if sab not in config["allowed_sabs"]:
                continue

            cui_semtypes = semtypes.get(cui)
            if not cui_semtypes:
                continue

            normalized_text = normalize_text(term_text)
            if not normalized_text:
                continue

            dedup_key = (cui, sab, tty, code, normalized_text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            candidate = {
                "candidate_id": f"{source_type}_{len(candidates)}",
                "source_type": source_type,
                "cui": cui,
                "sab": sab,
                "tty": tty,
                "code": code,
                "term_text": term_text,
                "normalized_text": normalized_text,
                "semantic_types": cui_semtypes,
            }
            candidates.append(candidate)
            sab_counter[sab] += 1
            for semtype_name in cui_semtypes:
                semtype_counter[semtype_name] += 1

    stats = {
        "source_type": source_type,
        "num_candidates": len(candidates),
        "allowed_sabs": sorted(config["allowed_sabs"]),
        "allowed_semtypes": sorted(config["allowed_semtypes"]),
        "counts_by_sab": dict(sorted(sab_counter.items())),
        "top_semantic_types": semtype_counter.most_common(20),
    }
    return candidates, stats


def load_rxnorm_cui_map(mrconso_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with mrconso_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 18:
                continue
            if parts[1] != "ENG" or parts[11] != "RXNORM" or parts[16] != "N":
                continue
            cui = parts[0]
            rxcui = parts[13]
            if rxcui and rxcui not in mapping:
                mapping[rxcui] = cui
    return mapping


def load_rxnorm_ndc_map(rxnsat_path: Path, allowed_rxcuis: set[str]) -> dict[str, list[str]]:
    ndcs_by_rxcui: dict[str, set[str]] = defaultdict(set)
    with rxnsat_path.open(errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 12:
                continue
            rxcui = parts[0]
            atn = parts[8]
            atv = parts[10]
            suppress = parts[11]
            if rxcui not in allowed_rxcuis or atn != "NDC" or suppress == "O":
                continue
            normalized_ndc = normalize_ndc(atv)
            if normalized_ndc:
                ndcs_by_rxcui[rxcui].add(normalized_ndc)
    return {rxcui: sorted(values) for rxcui, values in ndcs_by_rxcui.items()}


def build_medication_candidates(
    mrconso_path: Path,
    mrsty_path: Path,
    rxnconso_path: Path,
    rxnsat_path: Path,
) -> tuple[list[dict], dict]:
    rxcui_to_cui = load_rxnorm_cui_map(mrconso_path)
    semtypes = load_semtypes_for_cuis(set(rxcui_to_cui.values()), mrsty_path)
    ndc_map = load_rxnorm_ndc_map(rxnsat_path, set(rxcui_to_cui))

    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    tty_counter: Counter[str] = Counter()
    semtype_counter: Counter[str] = Counter()
    ndc_candidate_count = 0
    missing_umls_cui = 0

    with rxnconso_path.open(errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 18:
                continue

            (
                rxcui,
                lat,
                _ts,
                _lui,
                _stt,
                _sui,
                _ispref,
                _rxaui,
                _saui,
                _scui,
                _sdui,
                sab,
                tty,
                code,
                term_text,
                _srl,
                suppress,
                _cvf,
            ) = parts[:18]

            if lat != "ENG" or suppress != "N":
                continue
            if tty not in MEDICATION_TTYS:
                continue

            umls_cui = rxcui_to_cui.get(rxcui)
            concept_cui = umls_cui or f"RXNORM:{rxcui}"

            normalized_text = normalize_text(term_text)
            if not normalized_text:
                continue

            dedup_key = (rxcui, tty, normalized_text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            candidate_ndcs = ndc_map.get(rxcui, [])
            if candidate_ndcs:
                ndc_candidate_count += 1

            cui_semtypes = semtypes.get(umls_cui, []) if umls_cui else []
            if umls_cui is None:
                missing_umls_cui += 1
            candidate = {
                "candidate_id": f"medication_{len(candidates)}",
                "source_type": "medication",
                "cui": concept_cui,
                "umls_cui": umls_cui,
                "sab": "RXNORM",
                "source_sab": sab,
                "tty": tty,
                "code": code or rxcui,
                "rxcui": rxcui,
                "term_text": term_text,
                "normalized_text": normalized_text,
                "semantic_types": cui_semtypes,
                "ndc_codes": candidate_ndcs,
            }
            candidates.append(candidate)
            tty_counter[tty] += 1
            for semtype_name in cui_semtypes:
                semtype_counter[semtype_name] += 1

    stats = {
        "source_type": "medication",
        "num_candidates": len(candidates),
        "allowed_sabs": ["RXNORM"],
        "allowed_semtypes": None,
        "allowed_ttys": sorted(MEDICATION_TTYS),
        "counts_by_sab": {"RXNORM": len(candidates)},
        "top_source_sabs": Counter(row["source_sab"] for row in candidates).most_common(20),
        "top_term_types": tty_counter.most_common(20),
        "top_semantic_types": semtype_counter.most_common(20),
        "num_distinct_rxcui": len({row["rxcui"] for row in candidates}),
        "num_candidates_with_ndc": ndc_candidate_count,
        "num_distinct_ndc": sum(len(values) for values in ndc_map.values()),
        "num_candidates_without_umls_cui": missing_umls_cui,
        "rxnconso_path": str(rxnconso_path),
        "rxnsat_path": str(rxnsat_path),
    }
    return candidates, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-type",
        choices=sorted(SOURCE_CONFIG.keys()),
        required=True,
        help="Which candidate table to build.",
    )
    parser.add_argument("--mrconso-path", type=Path, default=MRCONSO_PATH)
    parser.add_argument("--mrsty-path", type=Path, default=MRSTY_PATH)
    parser.add_argument("--rxnconso-path", type=Path, default=RXNCONSO_PATH)
    parser.add_argument("--rxnsat-path", type=Path, default=RXNSAT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.source_type == "medication":
        candidates, stats = build_medication_candidates(
            mrconso_path=args.mrconso_path,
            mrsty_path=args.mrsty_path,
            rxnconso_path=args.rxnconso_path,
            rxnsat_path=args.rxnsat_path,
        )
    else:
        candidates, stats = build_code_candidates(
            source_type=args.source_type,
            mrconso_path=args.mrconso_path,
            mrsty_path=args.mrsty_path,
        )

    candidates_path = OUTPUT_DIR / f"{args.source_type}_candidates.jsonl"
    stats_path = OUTPUT_DIR / f"{args.source_type}_candidate_stats.json"

    with candidates_path.open("w") as handle:
        for row in candidates:
            handle.write(json.dumps(row) + "\n")

    with stats_path.open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps({"candidates_path": str(candidates_path), "stats_path": str(stats_path), **stats}, indent=2))


if __name__ == "__main__":
    main()
