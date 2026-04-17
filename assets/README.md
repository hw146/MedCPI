# External Assets

This repository does not include external terminology resources or model caches.

## Not Included In GitHub

- UMLS Metathesaurus files
- RxNorm RRF files
- SapBERT or other Hugging Face model caches
- ICD codebook source files

## Expected Layout

```text
assets/
├── codebooks/
│   ├── icd9/
│   │   ├── CMS32_DESC_LONG_DX.txt
│   │   ├── CMS32_DESC_LONG_SG.txt
│   │   ├── CMS32_DESC_SHORT_DX.txt
│   │   └── CMS32_DESC_SHORT_SG.txt
│   └── icd10/
│       ├── diagnosis/
│       │   └── icd10cm_codes.txt
│       └── procedure/
│           └── icd10pcs_order.txt
├── models/
│   └── hf-cache/
├── rxnorm/
│   ├── RXNCONSO.RRF
│   └── RXNSAT.RRF
└── umls/
    ├── MRCONSO.RRF
    ├── MRREL.RRF
    ├── MRSTY.RRF
    └── SemGroups.txt
```

## Notes

- `build_umls_candidate_index.py` uses `MRCONSO.RRF`, `MRSTY.RRF`, `RXNCONSO.RRF`, and `RXNSAT.RRF`.
- `export_global_relation_inventory.py`, `export_relation_inventory.py`, `build_concept_mkg.py`, and `build_patient_pkg.py` use UMLS files under `assets/umls/`.
- `sapbert_map_mentions.py` downloads or reuses the SapBERT model through the Hugging Face cache path.

Set environment variables if you want to place these resources outside the repository:

- `MEDCPI_ASSETS_ROOT`
- `MEDCPI_UMLS_ROOT`
- `MEDCPI_RXNORM_ROOT`
- `MEDCPI_MODEL_CACHE_ROOT`
