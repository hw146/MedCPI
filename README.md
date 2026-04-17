# MedCPI

Code for the paper: `MedCPI: A Construct--Personalize--Integrate Framework for KG-enhanced Clinical Prediction`

## Requirements

Python 3.10+

```bash
pip install -r requirements.txt
```

## Schema-induction backend

The schema-induction script uses an OpenAI-compatible chat completions endpoint.
This supports both hosted APIs and local deployments that expose the same interface.

OpenAI API example:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-5
```

Local deployment example:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export OPENAI_MODEL=your-local-model-name
```

The relation-text encoder defaults to `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
and can be overridden with `MEDCPI_ENCODER_MODEL`.

## Data

Due to privacy, ethics, and licensing considerations, this repository does not include:

- MIMIC patient data
- UMLS / RxNorm release files

Expected paths:

```text
data/raw/mimic3/
data/raw/mimic4/
assets/
outputs/
```

Path overrides can be set with environment variables in `.env.example`.
Shared path resolution is implemented in `scripts/path_roots.py`.

## Code Organization

### 1. EHR preprocessing and concept alignment

```text
scripts/preprocess_ehr_to_jsonl.py
scripts/prepare_mapping_inputs.py
scripts/build_umls_candidate_index.py
scripts/sapbert_map_mentions.py
scripts/build_aligned_patient_ehr.py
scripts/build_task_instances.py
scripts/create_task_splits.py
```

### 2. Construct

```text
scripts/export_global_relation_inventory.py
scripts/induce_relation_schema.py
scripts/extract_task_train_concepts.py
scripts/export_relation_inventory.py
scripts/build_concept_mkg.py
scripts/verify_construct_outputs.py
scripts/run_construct_pipeline.py
```

### 3. Personalize

```text
scripts/build_patient_pkg.py
scripts/pkg_utils.py
```

### 4. Integrate

```text
scripts/prepare_integrate_inputs.py
scripts/integrate_data.py
scripts/integrate_dataset.py
scripts/integrate_model.py
scripts/train_integrate_formal.py
```

## Minimal Workflow

```bash
python3 scripts/preprocess_ehr_to_jsonl.py --dataset mimic3
python3 scripts/prepare_mapping_inputs.py --dataset mimic3
python3 scripts/build_umls_candidate_index.py
python3 scripts/sapbert_map_mentions.py --dataset mimic3
python3 scripts/build_aligned_patient_ehr.py --dataset mimic3
python3 scripts/build_task_instances.py --dataset mimic3
python3 scripts/create_task_splits.py --dataset mimic3

python3 scripts/export_global_relation_inventory.py
python3 scripts/induce_relation_schema.py --dataset shared --task mortality
python3 scripts/run_construct_pipeline.py --dataset mimic3 --tasks mortality

python3 scripts/build_patient_pkg.py \
  --dataset mimic3 \
  --task mortality \
  --split all \
  --attach-bridge-paths

python3 scripts/prepare_integrate_inputs.py \
  --dataset mimic3 \
  --task mortality \
  --split all \
  --build-manifest

python3 scripts/integrate_data.py --dataset mimic3 --task mortality --split train --build-indexes
python3 scripts/integrate_data.py --dataset mimic3 --task mortality --split valid --build-indexes
python3 scripts/integrate_data.py --dataset mimic3 --task mortality --split test --build-indexes

python3 scripts/train_integrate_formal.py \
  --dataset mimic3 \
  --task mortality \
  --selection-metric auroc \
  --seeds 0,1,2,3,4 \
  --emit-summary
```

The shared schema-induction step writes task-specific shared schemas to
`outputs/construct/shared/<task>/relation_schema.json`.

This release already includes lightweight shared schema files under
`outputs/construct/shared/` so the downstream Construct, Personalize, and
Integrate stages can be run without re-calling the LLM schema-induction step.
