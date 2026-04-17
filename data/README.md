# Data Policy

This public release does not include any patient-level data.

## Not Included

- Raw MIMIC-III tables
- Raw MIMIC-IV tables
- Any preprocessed patient JSONL files
- Any task instances, split manifests, aligned EHR files, PKGs, manifests, checkpoints, or evaluation outputs derived from patient records

## Expected Layout

```text
data/
└── raw/
    ├── mimic3/
    │   ├── ADMISSIONS.csv
    │   ├── PATIENTS.csv
    │   ├── DIAGNOSES_ICD.csv
    │   ├── PROCEDURES_ICD.csv
    │   └── PRESCRIPTIONS.csv
    └── mimic4/
        ├── admissions.csv
        ├── patients.csv
        ├── diagnoses_icd.csv
        ├── procedures_icd.csv
        └── prescriptions.csv
```

## Data Paths

Raw MIMIC data should remain outside version control.

If your data are stored outside the default layout, set:

- `MEDCPI_MIMIC3_DIR`
- `MEDCPI_MIMIC4_DIR`
