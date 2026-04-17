#!/usr/bin/env python3
"""Map unique mentions to UMLS candidates with SapBERT on GPU."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from path_roots import MODEL_CACHE_ROOT


MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
HF_CACHE = MODEL_CACHE_ROOT
FALLBACK_MODES = {"code_system_subset", "code_system_lexical_subset", "subset_full"}
MEDICATION_TTY_PRIORITY = {
    "IN": 0,
    "PIN": 1,
    "MIN": 2,
    "BN": 3,
    "SCD": 4,
    "SBD": 5,
    "SCDC": 6,
    "SBDC": 7,
    "SCDF": 8,
    "SBDF": 9,
    "SCDG": 10,
    "SBDG": 11,
    "SCDFP": 12,
    "SBDFP": 13,
    "SCDGP": 14,
    "BPCK": 15,
    "GPCK": 16,
}
MEDICATION_TOKEN_STOPWORDS = {
    "bag",
    "cap",
    "capsule",
    "cream",
    "gel",
    "inj",
    "injectable",
    "injection",
    "kit",
    "needle",
    "oral",
    "pen",
    "powder",
    "solution",
    "soln",
    "susp",
    "suspension",
    "syringe",
    "tab",
    "tablet",
    "topical",
    "ultra",
    "fine",
}
MEDICATION_TEXT_EXPANSIONS = {
    "ns": ["sodium chloride"],
    "1 2 ns": ["sodium chloride 0 45"],
    "1 4 ns": ["sodium chloride 0 225"],
    "d5w": ["dextrose 5"],
    "d5 1 2 ns": ["dextrose 5 sodium chloride 0 45"],
    "d5 1 4 ns": ["dextrose 5 sodium chloride 0 225"],
    "sw": ["sterile water"],
    "acd a": ["acd a solution", "anticoagulant citrate dextrose a"],
    "ashlyna": ["ashlyna tablet"],
}
MEDICATION_SUPPLY_KEYWORDS = {
    "needle",
    "syringe",
    "lancet",
    "catheter",
    "tray",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    cleaned = []
    prev_space = False
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            prev_space = False
        else:
            if not prev_space:
                cleaned.append(" ")
                prev_space = True
    return "".join(cleaned).strip()


def medication_boundary_variant(text: str) -> str:
    value = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    value = re.sub(r"([0-9])([a-z])", r"\1 \2", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_code(code: str | None) -> str:
    if not code:
        return ""
    return "".join(ch for ch in str(code).upper() if ch.isalnum())


def normalize_ndc(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def normalized_tokens(text: str | None) -> list[str]:
    normalized = normalize_text(text or "")
    return [token for token in normalized.split() if token]


def medication_token_set(text: str | None) -> set[str]:
    tokens = set()
    for token in normalized_tokens(text):
        if len(token) <= 2:
            continue
        if token in MEDICATION_TOKEN_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def expected_sab_for_mention(code_system: str | None, source_type: str) -> str | None:
    if source_type == "medication":
        return "RXNORM"
    if code_system == "ICD9":
        return "ICD9CM"
    if code_system == "ICD10" and source_type == "diagnosis":
        return "ICD10CM"
    if code_system == "ICD10" and source_type == "procedure":
        return "ICD10PCS"
    return None


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = pooled / denom
            embeddings = F.normalize(embeddings, dim=1)
        batches.append(embeddings.cpu())
    return torch.cat(batches, dim=0)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mentions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-type", choices=["diagnosis", "procedure", "medication"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-mentions", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fallback-score-threshold", type=float, default=0.9)
    return parser.parse_args()


def build_candidate_indices(
    candidates: list[dict],
) -> tuple[
    dict[str, torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    subset_map: dict[str, list[int]] = defaultdict(list)
    exact_sab_map: dict[tuple[str, str], list[int]] = defaultdict(list)
    code_sab_map: dict[tuple[str, str], list[int]] = defaultdict(list)
    exact_global_map: dict[str, list[int]] = defaultdict(list)
    ndc_map: dict[str, list[int]] = defaultdict(list)
    token_map: dict[str, list[int]] = defaultdict(list)

    for idx, candidate in enumerate(candidates):
        sab = candidate["sab"]
        normalized_text = candidate["normalized_text"]
        subset_map[sab].append(idx)
        exact_sab_map[(sab, normalized_text)].append(idx)
        exact_global_map[normalized_text].append(idx)

        normalized_code = normalize_code(candidate.get("code"))
        if normalized_code:
            code_sab_map[(sab, normalized_code)].append(idx)

        for ndc in candidate.get("ndc_codes", []):
            normalized_ndc = normalize_ndc(ndc)
            if normalized_ndc:
                ndc_map[normalized_ndc].append(idx)

        if candidate.get("source_type") == "medication":
            for token in medication_token_set(candidate.get("term_text")):
                token_map[token].append(idx)

    subset_tensors = {
        sab: torch.tensor(indices, dtype=torch.long) for sab, indices in subset_map.items() if indices
    }
    exact_sab_tensors = {
        key: torch.tensor(indices, dtype=torch.long) for key, indices in exact_sab_map.items() if indices
    }
    code_sab_tensors = {
        key: torch.tensor(indices, dtype=torch.long) for key, indices in code_sab_map.items() if indices
    }
    exact_global_tensors = {
        key: torch.tensor(indices, dtype=torch.long) for key, indices in exact_global_map.items() if indices
    }
    ndc_tensors = {
        key: torch.tensor(indices, dtype=torch.long) for key, indices in ndc_map.items() if indices
    }
    token_tensors = {
        key: torch.tensor(indices, dtype=torch.long) for key, indices in token_map.items() if indices
    }

    return subset_tensors, exact_sab_tensors, code_sab_tensors, exact_global_tensors, ndc_tensors, token_tensors


def medication_aliases(row: dict) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_alias(mode: str, value: str | None) -> None:
        normalized = normalize_text(value or "")
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        aliases.append((mode, normalized))
        boundary_variant = medication_boundary_variant(normalized)
        if boundary_variant and boundary_variant not in seen:
            seen.add(boundary_variant)
            aliases.append((mode, boundary_variant))

    add_alias("exact_text", row.get("mention_text"))
    for alias in row.get("text_aliases", []):
        source = alias.get("source")
        if source == "ehr_medication_generic_name":
            mode = "generic_name_exact"
        elif source == "ehr_medication_poe_name":
            mode = "poe_name_exact"
        else:
            mode = "alias_text_exact"
        add_alias(mode, alias.get("text"))
    for alias_mode, alias_text in list(aliases):
        for expanded in MEDICATION_TEXT_EXPANSIONS.get(alias_text, []):
            add_alias(alias_mode, expanded)
        token_set = set(normalized_tokens(alias_text))
        if "ns" in token_set:
            add_alias(alias_mode, "sodium chloride")
        if "sw" in token_set:
            add_alias(alias_mode, "sterile water")
        if "d5w" in token_set:
            add_alias(alias_mode, "dextrose 5")
        for token in ("novolog", "nephrocaps", "ashlyna", "bactrim"):
            if token in token_set:
                add_alias(alias_mode, token)
        if {"citrate", "dextrose", "acd"} <= token_set:
            add_alias(alias_mode, "acd a solution")
            add_alias(alias_mode, "anticoagulant citrate dextrose a")
    return aliases


def medication_candidate_rank(candidate: dict, alias_texts: list[str]) -> tuple[int, int, int, str]:
    candidate_tokens = medication_token_set(candidate.get("term_text"))
    best_overlap = 0
    for alias_text in alias_texts:
        best_overlap = max(best_overlap, len(candidate_tokens & medication_token_set(alias_text)))
    tty_rank = MEDICATION_TTY_PRIORITY.get(candidate.get("tty"), 999)
    text_len_rank = len(candidate_tokens)
    return (-best_overlap, tty_rank, text_len_rank, candidate.get("term_text", ""))


def choose_medication_candidate(indices: list[int], candidates: list[dict], alias_texts: list[str]) -> tuple[int, float]:
    ranked = sorted(indices, key=lambda idx: medication_candidate_rank(candidates[idx], alias_texts))
    best_idx = ranked[0]
    best_candidate = candidates[best_idx]
    best_overlap = -medication_candidate_rank(best_candidate, alias_texts)[0]
    confidence = 1.0 if best_overlap > 0 else 0.95
    return best_idx, confidence


def medication_lexical_subset(
    row: dict,
    token_indices: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor | None:
    gathered: list[torch.Tensor] = []
    for _mode, alias_text in medication_aliases(row):
        for token in medication_token_set(alias_text):
            tensor = token_indices.get(token)
            if tensor is not None:
                gathered.append(tensor)
    if not gathered:
        return None
    merged = torch.cat(gathered)
    if merged.numel() == 0:
        return None
    return torch.unique(merged).to(device)


def medication_supply_like(row: dict) -> bool:
    mention_tokens = set(normalized_tokens(row.get("mention_text") or ""))
    if not (mention_tokens & MEDICATION_SUPPLY_KEYWORDS):
        return False
    if mention_tokens & {"ns", "sw", "d5w", "prismasol", "acd"}:
        return False
    if row.get("ndc") and normalize_ndc(row.get("ndc")) not in {"", "0"}:
        return False
    return not medication_token_set(row.get("mention_text"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    mentions = [
        row
        for row in load_jsonl(args.mentions)
        if row.get("source_type") == args.source_type and row.get("mention_text")
    ]
    if args.max_mentions is not None:
        mentions = mentions[: args.max_mentions]
    if not mentions:
        raise SystemExit("No mentions selected for mapping.")

    candidates = load_jsonl(args.candidates)
    if not candidates:
        raise SystemExit("No candidates loaded.")

    embeddings_path = args.output_dir / f"{args.source_type}_candidate_embeddings.pt"
    if embeddings_path.exists():
        candidate_embeddings = torch.load(embeddings_path, map_location="cpu")
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE))
        model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE))
        model.to(device)
        model.eval()
        candidate_embeddings = encode_texts(
            [row["term_text"] for row in candidates],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        ).half()
        torch.save(candidate_embeddings, embeddings_path)
        del model
        torch.cuda.empty_cache()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE))
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE))
    model.to(device)
    model.eval()

    candidate_embeddings = candidate_embeddings.to(device)
    subset_indices, exact_sab_indices, code_sab_indices, exact_global_indices, ndc_indices, token_indices = build_candidate_indices(candidates)

    mention_texts = [row["mention_text"] for row in mentions]
    mention_embeddings = encode_texts(
        mention_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    ).to(device).half()

    results = []
    stats = Counter()
    score_sum = 0.0

    for row, mention_embedding in zip(mentions, mention_embeddings):
        normalized_text = normalize_text(row["mention_text"])
        normalized_code = normalize_code(row.get("normalized_code") or row.get("raw_code"))
        normalized_ndc = normalize_ndc(row.get("ndc"))
        expected_sab = expected_sab_for_mention(row.get("code_system"), args.source_type)
        candidate_idx = None
        retrieval_mode = "subset_full"
        forced_best_idx = None
        forced_score = None
        if args.source_type == "medication" and medication_supply_like(row):
            result = {
                **row,
                "matched_cui": None,
                "matched_term_text": None,
                "matched_sab": None,
                "matched_tty": None,
                "matched_code": None,
                "matched_rxcui": None,
                "matched_ndc_codes": [],
                "matched_semantic_types": [],
                "expected_sab": expected_sab,
                "score": 0.0,
                "retrieval_mode": "non_drug_supply",
                "review_required": False,
                "top_k": [],
            }
            results.append(result)
            stats["non_drug_supply"] += 1
            stats["unmapped_total"] += 1
            continue

        if args.source_type == "medication" and normalized_ndc:
            exact_ndc_indices = ndc_indices.get(normalized_ndc)
            if exact_ndc_indices is not None:
                ndc_index_list = exact_ndc_indices.tolist()
                ndc_rxcuis = {candidates[idx].get("rxcui") for idx in ndc_index_list}
                alias_texts = [alias_text for _, alias_text in medication_aliases(row)]
                if len(ndc_rxcuis) == 1:
                    forced_best_idx, forced_score = choose_medication_candidate(ndc_index_list, candidates, alias_texts)
                    retrieval_mode = "exact_ndc"
                else:
                    candidate_idx = exact_ndc_indices.to(device)
                    retrieval_mode = "exact_ndc"

        if candidate_idx is None and expected_sab and normalized_code:
            exact_code_indices = code_sab_indices.get((expected_sab, normalized_code))
            if exact_code_indices is not None:
                candidate_idx = exact_code_indices.to(device)
                retrieval_mode = "exact_code"

        if candidate_idx is None and args.source_type == "medication" and expected_sab:
            for alias_mode, alias_text in medication_aliases(row):
                exact_indices = exact_sab_indices.get((expected_sab, alias_text))
                if exact_indices is not None:
                    alias_texts = [value for _, value in medication_aliases(row)]
                    forced_best_idx, forced_score = choose_medication_candidate(exact_indices.tolist(), candidates, alias_texts)
                    candidate_idx = exact_indices.to(device)
                    retrieval_mode = alias_mode
                    break

        if candidate_idx is None and expected_sab:
            exact_indices = exact_sab_indices.get((expected_sab, normalized_text))
            if exact_indices is not None:
                if args.source_type == "medication":
                    alias_texts = [value for _, value in medication_aliases(row)]
                    forced_best_idx, forced_score = choose_medication_candidate(exact_indices.tolist(), candidates, alias_texts)
                candidate_idx = exact_indices.to(device)
                retrieval_mode = "exact_text"

        if candidate_idx is None and expected_sab:
            by_code_system = subset_indices.get(expected_sab)
            if by_code_system is not None:
                candidate_idx = by_code_system.to(device)
                retrieval_mode = "code_system_subset"

        if candidate_idx is not None and args.source_type == "medication" and retrieval_mode == "code_system_subset":
            lexical_subset = medication_lexical_subset(row, token_indices, device)
            if lexical_subset is not None and lexical_subset.numel() > 0:
                candidate_idx = lexical_subset
                retrieval_mode = "code_system_lexical_subset"
                if lexical_subset.numel() == 1:
                    alias_texts = [alias_text for _, alias_text in medication_aliases(row)]
                    forced_best_idx, forced_score = choose_medication_candidate(
                        lexical_subset.tolist(), candidates, alias_texts
                    )

        if candidate_idx is None:
            exact_indices = exact_global_indices.get(normalized_text)
            if exact_indices is not None:
                candidate_idx = exact_indices.to(device)
                retrieval_mode = "exact_text_cross_sab"
            else:
                candidate_idx = torch.arange(candidate_embeddings.shape[0], device=device)

        if forced_best_idx is not None:
            best_candidate = candidates[forced_best_idx]
            best_global_idx = forced_best_idx
            top_score = round(float(forced_score), 6)
            ranked_idx = [forced_best_idx]
            if normalized_ndc:
                ndc_tensor = ndc_indices.get(normalized_ndc)
                if ndc_tensor is not None:
                    alias_texts = [alias_text for _, alias_text in medication_aliases(row)]
                    remaining = [idx for idx in ndc_tensor.tolist() if idx != forced_best_idx]
                    remaining.sort(key=lambda idx: medication_candidate_rank(candidates[idx], alias_texts))
                    ranked_idx.extend(remaining[: max(0, args.top_k - 1)])
            top_matches = []
            for global_idx in ranked_idx[: args.top_k]:
                cand = candidates[global_idx]
                match_score = top_score if global_idx == forced_best_idx else 0.95
                top_matches.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "cui": cand["cui"],
                        "sab": cand["sab"],
                        "tty": cand["tty"],
                        "code": cand["code"],
                        "term_text": cand["term_text"],
                        "score": round(float(match_score), 6),
                    }
                )
        else:
            sub_embeddings = candidate_embeddings.index_select(0, candidate_idx)
            scores = torch.matmul(sub_embeddings, mention_embedding.unsqueeze(-1)).squeeze(-1)
            top_k = min(args.top_k, scores.shape[0])
            top_scores, top_positions = torch.topk(scores, k=top_k)
            best_global_idx = candidate_idx[top_positions[0]].item()
            best_candidate = candidates[best_global_idx]
            top_score = round(float(top_scores[0].item()), 6)
            top_matches = []
            for score_tensor, pos_tensor in zip(top_scores, top_positions):
                global_idx = candidate_idx[pos_tensor].item()
                cand = candidates[global_idx]
                top_matches.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "cui": cand["cui"],
                        "sab": cand["sab"],
                        "tty": cand["tty"],
                        "code": cand["code"],
                        "term_text": cand["term_text"],
                        "score": round(float(score_tensor.item()), 6),
                    }
                )
        review_required = retrieval_mode in FALLBACK_MODES and top_score < args.fallback_score_threshold

        result = {
            **row,
            "matched_cui": best_candidate["cui"],
            "matched_umls_cui": best_candidate.get("umls_cui"),
            "matched_term_text": best_candidate["term_text"],
            "matched_sab": best_candidate["sab"],
            "matched_tty": best_candidate["tty"],
            "matched_code": best_candidate["code"],
            "matched_rxcui": best_candidate.get("rxcui"),
            "matched_ndc_codes": best_candidate.get("ndc_codes", []),
            "matched_semantic_types": best_candidate["semantic_types"],
            "expected_sab": expected_sab,
            "score": top_score,
            "retrieval_mode": retrieval_mode,
            "review_required": review_required,
            "top_k": top_matches,
        }
        results.append(result)
        stats[retrieval_mode] += 1
        stats[best_candidate["sab"]] += 1
        stats["exact_total"] += int(retrieval_mode not in FALLBACK_MODES)
        stats["fallback_total"] += int(retrieval_mode in FALLBACK_MODES)
        stats["review_required"] += int(review_required)
        score_sum += float(top_score)

    result_path = args.output_dir / f"{args.source_type}_mention_to_umls.jsonl"
    stats_path = args.output_dir / f"{args.source_type}_mapping_stats.json"

    with result_path.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "source_type": args.source_type,
        "mentions_path": str(args.mentions),
        "candidates_path": str(args.candidates),
        "num_mentions": len(results),
        "num_candidates": len(candidates),
        "candidate_embeddings_path": str(embeddings_path),
        "device": device,
        "torch_cuda_available": torch.cuda.is_available(),
        "fallback_score_threshold": args.fallback_score_threshold,
        "avg_top1_score": round(score_sum / len(results), 6),
        "counts": dict(sorted(stats.items())),
    }
    with stats_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps({"result_path": str(result_path), "stats_path": str(stats_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
