#!/usr/bin/env python3
"""Shared path resolution for the public MedCPI release."""

import os
from pathlib import Path


def resolve_root(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


PROJECT_ROOT = resolve_root(
    "MEDCPI_PROJECT_ROOT",
    Path(__file__).resolve().parents[1],
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
DATA_ROOT = resolve_root("MEDCPI_DATA_ROOT", PROJECT_ROOT / "data")
ASSETS_ROOT = resolve_root("MEDCPI_ASSETS_ROOT", PROJECT_ROOT / "assets")
OUTPUT_ROOT = resolve_root("MEDCPI_OUTPUT_ROOT", PROJECT_ROOT / "outputs")

RAW_DATA_ROOT = resolve_root("MEDCPI_RAW_DATA_ROOT", DATA_ROOT / "raw")
MIMIC3_RAW_ROOT = resolve_root("MEDCPI_MIMIC3_DIR", RAW_DATA_ROOT / "mimic3")
MIMIC4_RAW_ROOT = resolve_root("MEDCPI_MIMIC4_DIR", RAW_DATA_ROOT / "mimic4")

PREPROCESSED_ROOT = resolve_root("MEDCPI_PREPROCESSED_ROOT", OUTPUT_ROOT / "preprocessed")
MAPPING_ROOT = resolve_root("MEDCPI_MAPPING_ROOT", OUTPUT_ROOT / "mapping")
ALIGNED_ROOT = resolve_root("MEDCPI_ALIGNED_ROOT", OUTPUT_ROOT / "aligned_ehr")
TASKS_ROOT = resolve_root("MEDCPI_TASKS_ROOT", OUTPUT_ROOT / "tasks")
SPLITS_ROOT = resolve_root("MEDCPI_SPLITS_ROOT", OUTPUT_ROOT / "splits")
CONSTRUCT_ROOT = resolve_root("MEDCPI_CONSTRUCT_ROOT", OUTPUT_ROOT / "construct")
PERSONALIZE_ROOT = resolve_root("MEDCPI_PERSONALIZE_ROOT", OUTPUT_ROOT / "personalize")
INTEGRATE_ROOT = resolve_root("MEDCPI_INTEGRATE_ROOT", OUTPUT_ROOT / "integrate")

UMLS_ROOT = resolve_root("MEDCPI_UMLS_ROOT", ASSETS_ROOT / "umls")
RXNORM_ROOT = resolve_root("MEDCPI_RXNORM_ROOT", ASSETS_ROOT / "rxnorm")
CODEBOOK_ROOT = resolve_root("MEDCPI_CODEBOOK_ROOT", ASSETS_ROOT / "codebooks")
MODEL_CACHE_ROOT = resolve_root("MEDCPI_MODEL_CACHE_ROOT", ASSETS_ROOT / "models" / "hf-cache")
