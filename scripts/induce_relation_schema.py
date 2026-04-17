#!/usr/bin/env python3
"""Run cluster-based LLM schema induction from a precomputed relation inventory."""

import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoTokenizer

from path_roots import CONSTRUCT_ROOT as DEFAULT_CONSTRUCT_ROOT

DEFAULT_ENCODER_MODEL_NAME_OR_PATH = os.environ.get(
    "MEDCPI_ENCODER_MODEL",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
)
TASK_DESCRIPTIONS = {
    "mortality": "In-hospital mortality prediction from longitudinal EHR.",
    "readmission_30d": "30-day readmission prediction from longitudinal EHR.",
    "t2dm_onset": "Future type 2 diabetes mellitus onset prediction at the patient level.",
    "cad_onset": "Future coronary artery disease onset prediction at the patient level.",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in clinical informatics and medical knowledge graphs. "
    "Return only a compact JSON object."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "shared"], required=True)
    parser.add_argument("--task", choices=["mortality", "readmission_30d", "t2dm_onset", "cad_onset"], required=True)
    parser.add_argument("--construct-root", type=Path, default=DEFAULT_CONSTRUCT_ROOT)
    parser.add_argument("--inventory-path", type=Path, default=None)
    parser.add_argument("--schema-output-path", type=Path, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--llm-model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL of an OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key or access token for the selected endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--encoder-model-name-or-path",
        type=str,
        default=DEFAULT_ENCODER_MODEL_NAME_OR_PATH,
    )
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--encoder-max-length", type=int, default=128)
    parser.add_argument("--distance-threshold", type=float, default=0.15)
    parser.add_argument("--relation-sample-limit", type=int, default=5)
    parser.add_argument("--cluster-sample-limit", type=int, default=12)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for relation embedding.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def schema_task_dir(construct_root: Path, dataset: str, task: str) -> Path:
    if dataset == "shared":
        return construct_root / "shared" / task
    return construct_root / dataset / task


def default_inventory_path(construct_root: Path, dataset: str, task: str) -> Path:
    if dataset == "shared":
        return construct_root / "shared" / "global_relation_inventory.json"
    return schema_task_dir(construct_root, dataset, task) / "relation_inventory.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def sample_to_snippet(sample: dict, fallback_relation: str) -> str:
    head = sample.get("term1") or sample.get("cui1") or "HEAD"
    relation = sample.get("rela") or sample.get("rel") or fallback_relation
    tail = sample.get("term2") or sample.get("cui2") or "TAIL"
    sab = sample.get("sab")
    if sab:
        return "{0} -- {1} -- {2} [SAB={3}]".format(head, relation, tail, sab)
    return "{0} -- {1} -- {2}".format(head, relation, tail)


def build_relation_text(row: dict, sample_limit: int) -> str:
    parts = ["relation: {0}".format(row["raw_relation"])]
    for sample in row.get("samples", [])[:sample_limit]:
        parts.append(sample_to_snippet(sample, row["raw_relation"]))
    return "\n".join(parts)


def embed_texts(
    texts: Sequence[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def cluster_relations(rows: List[dict], embeddings: np.ndarray, distance_threshold: float) -> List[dict]:
    if not rows:
        return []
    if len(rows) == 1:
        return [{"cluster_id": 0, "rows": rows}]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(embeddings)
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for label, row in zip(labels.tolist(), rows):
        grouped[label].append(row)

    clusters = []
    for cluster_id, label in enumerate(sorted(grouped)):
        members = sorted(grouped[label], key=lambda row: (-row["count"], row["raw_relation"]))
        clusters.append({"cluster_id": cluster_id, "rows": members})
    return clusters


def build_cluster_summary(cluster: dict, sample_limit: int, relation_sample_limit: int) -> dict:
    samples = []
    for row in cluster["rows"]:
        for sample in row.get("samples", [])[:relation_sample_limit]:
            samples.append(
                {
                    "source_relation": row["raw_relation"],
                    "count": row["count"],
                    "snippet": sample_to_snippet(sample, row["raw_relation"]),
                }
            )
    samples = samples[:sample_limit]

    return {
        "cluster_id": cluster["cluster_id"],
        "num_relations": len(cluster["rows"]),
        "total_relation_count": sum(row["count"] for row in cluster["rows"]),
        "member_relations": [
            {
                "raw_relation": row["raw_relation"],
                "count": row["count"],
                "samples": [sample_to_snippet(sample, row["raw_relation"]) for sample in row.get("samples", [])[:relation_sample_limit]],
            }
            for row in cluster["rows"]
        ],
        "representative_triples": samples,
    }


def normalize_canonical_name(name: Optional[str]) -> str:
    if name is None:
        return "IGNORE"
    value = name.strip()
    if not value or value.upper() == "IGNORE":
        return "IGNORE"
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "IGNORE"


def llm_user_prompt(task: str, cluster_summary: dict, cluster_index: int, total_clusters: int) -> str:
    payload = {
        "task": task,
        "task_description": TASK_DESCRIPTIONS[task],
        "cluster_index": cluster_index,
        "total_clusters": total_clusters,
        "instructions": {
            "role": "Assess whether a cluster of relation types is useful for the clinical prediction task.",
            "decision_criteria": [
                "Set keep=true only if the relations in the cluster provide task-relevant clinical knowledge that can plausibly support prediction for the given task directly or via short clinical reasoning.",
                "Otherwise set keep=false and output canonical_name=\"IGNORE\".",
            ],
            "naming_rules": [
                "Use lower_snake_case.",
                "Use a verb-like relation name such as treated_by, diagnosed_by, has_risk_factor, has_complication, or has_symptom.",
                "Use at most four words.",
                "Prefer specific, clinically meaningful names when possible; avoid overly vague names unless a broader association label is needed to capture the retained relation family.",
                "Output one canonical name only.",
            ],
            "output_format": {
                "keep": True,
                "canonical_name": "name or IGNORE",
            },
        },
        "relation_cluster_summary": cluster_summary,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def call_chat_completion(base_url: str, model: str, api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def validate_cluster_decision(cluster: dict, parsed: dict) -> dict:
    keep = bool(parsed.get("keep"))
    canonical_name = normalize_canonical_name(parsed.get("canonical_name"))
    if not keep:
        canonical_name = "IGNORE"
    return {
        "cluster_id": cluster["cluster_id"],
        "member_relations": [row["raw_relation"] for row in cluster["rows"]],
        "keep": keep and canonical_name != "IGNORE",
        "canonical_name": canonical_name if keep else "IGNORE",
        "rationale": (parsed.get("rationale") or "").strip(),
    }


def resolve_consistency(cluster_decisions: List[dict]) -> List[dict]:
    canonical_to_cluster_ids: Dict[str, List[int]] = defaultdict(list)
    for decision in cluster_decisions:
        if decision["keep"]:
            decision["canonical_name"] = normalize_canonical_name(decision["canonical_name"])
            if decision["canonical_name"] == "IGNORE":
                decision["keep"] = False
            else:
                canonical_to_cluster_ids[decision["canonical_name"]].append(decision["cluster_id"])
        if not decision["keep"]:
            decision["canonical_name"] = "IGNORE"

    for decision in cluster_decisions:
        decision["merged_cluster_ids"] = canonical_to_cluster_ids.get(decision["canonical_name"], [])
    return sorted(cluster_decisions, key=lambda row: row["cluster_id"])


def relation_decisions_from_clusters(cluster_decisions: List[dict], clusters: List[dict]) -> List[dict]:
    cluster_map = {cluster["cluster_id"]: cluster for cluster in clusters}
    relation_rows = []
    for decision in cluster_decisions:
        cluster = cluster_map[decision["cluster_id"]]
        for row in cluster["rows"]:
            relation_rows.append(
                {
                    "raw_relation": row["raw_relation"],
                    "keep": decision["keep"],
                    "canonical_relation": decision["canonical_name"] if decision["keep"] else "IGNORE",
                    "rationale": decision["rationale"],
                    "cluster_id": decision["cluster_id"],
                }
            )
    return sorted(relation_rows, key=lambda row: row["raw_relation"])


def main() -> None:
    args = parse_args()
    task_dir = schema_task_dir(args.construct_root, args.dataset, args.task)
    inventory_path = args.inventory_path or default_inventory_path(args.construct_root, args.dataset, args.task)
    schema_output_path = args.schema_output_path or (task_dir / "relation_schema.json")
    llm_trace_dir = task_dir / "llm_schema_runs"
    llm_trace_dir.mkdir(parents=True, exist_ok=True)

    if not inventory_path.exists():
        raise FileNotFoundError("Missing relation inventory: {0}".format(inventory_path))

    payload = json.loads(inventory_path.read_text())
    relation_rows = payload["relations"]
    relation_texts = [build_relation_text(row, args.relation_sample_limit) for row in relation_rows]
    embeddings = embed_texts(
        texts=relation_texts,
        model_name_or_path=args.encoder_model_name_or_path,
        batch_size=args.encoder_batch_size,
        max_length=args.encoder_max_length,
        device=choose_device(args.device),
    )
    clusters = cluster_relations(
        rows=relation_rows,
        embeddings=embeddings,
        distance_threshold=args.distance_threshold,
    )
    cluster_summaries = [
        build_cluster_summary(
            cluster=cluster,
            sample_limit=args.cluster_sample_limit,
            relation_sample_limit=args.relation_sample_limit,
        )
        for cluster in clusters
    ]
    write_json(
        llm_trace_dir / "relation_clusters.json",
        {
            "inventory_path": str(inventory_path),
            "encoder_model_name_or_path": args.encoder_model_name_or_path,
            "distance_threshold": args.distance_threshold,
            "relation_sample_limit": args.relation_sample_limit,
            "cluster_sample_limit": args.cluster_sample_limit,
            "num_raw_relations": len(relation_rows),
            "num_clusters": len(cluster_summaries),
            "clusters": cluster_summaries,
        },
    )

    if args.dry_run:
        prompt = llm_user_prompt(args.task, cluster_summaries[0], 1, len(cluster_summaries))
        print(prompt)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError("Missing API key env var: {0}".format(args.api_key_env))

    cluster_decisions = []
    cluster_run_summaries = []
    for cluster_index, (cluster, cluster_summary) in enumerate(zip(clusters, cluster_summaries), start=1):
        prompt = llm_user_prompt(args.task, cluster_summary, cluster_index, len(cluster_summaries))
        prompt_path = llm_trace_dir / "cluster_{0:03d}.prompt.json".format(cluster_index)
        prompt_path.write_text(prompt)

        for attempt in range(1, args.max_retries + 1):
            try:
                content = call_chat_completion(
                    base_url=args.llm_base_url,
                    model=args.llm_model,
                    api_key=api_key,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    temperature=args.temperature,
                )
                raw_response_path = llm_trace_dir / "cluster_{0:03d}.response.txt".format(cluster_index)
                raw_response_path.write_text(content)
                parsed = json.loads(strip_json_fence(content))
                decision = validate_cluster_decision(cluster, parsed)
                decisions_path = llm_trace_dir / "cluster_{0:03d}.decision.json".format(cluster_index)
                write_json(decisions_path, decision)
                cluster_decisions.append(decision)
                cluster_run_summaries.append(
                    {
                        "cluster_index": cluster_index,
                        "cluster_id": cluster["cluster_id"],
                        "num_relations": len(cluster["rows"]),
                        "prompt_path": str(prompt_path),
                        "response_path": str(raw_response_path),
                        "decision_path": str(decisions_path),
                    }
                )
                break
            except Exception:
                if attempt == args.max_retries:
                    raise
                time.sleep(args.sleep_seconds)

    cluster_decisions = resolve_consistency(cluster_decisions)
    relation_decisions = relation_decisions_from_clusters(cluster_decisions, clusters)
    canonical_relations = sorted(
        {
            row["canonical_relation"]
            for row in relation_decisions
            if row["keep"] and row["canonical_relation"] != "IGNORE"
        }
    )
    final_schema = {
        "dataset": args.dataset,
        "task": args.task,
        "schema_scope": "task_specific_shared_across_datasets" if args.dataset == "shared" else "task_specific",
        "shared_across_datasets": args.dataset == "shared",
        "decision_source": "llm",
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "inventory_path": str(inventory_path),
        "inventory_scope": payload.get("scope", "anchor_incident_relations"),
        "encoder_model_name_or_path": args.encoder_model_name_or_path,
        "distance_threshold": args.distance_threshold,
        "relation_sample_limit": args.relation_sample_limit,
        "cluster_sample_limit": args.cluster_sample_limit,
        "num_raw_relations": len(relation_rows),
        "num_relation_clusters": len(clusters),
        "num_kept_clusters": sum(1 for row in cluster_decisions if row["keep"]),
        "num_kept_relations": sum(1 for row in relation_decisions if row["keep"]),
        "num_canonical_relations": len(canonical_relations),
        "canonical_relations": canonical_relations,
        "clusters": cluster_decisions,
        "cluster_runs": cluster_run_summaries,
        "relations": relation_decisions,
    }
    write_json(schema_output_path, final_schema)
    print(
        json.dumps(
            {
                "schema_output_path": str(schema_output_path),
                "num_raw_relations": len(relation_rows),
                "num_relation_clusters": len(clusters),
                "num_kept_relations": final_schema["num_kept_relations"],
                "num_canonical_relations": len(canonical_relations),
                "llm_model": args.llm_model,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
