Released task-specific shared schemas for the public MedCPI codebase.

These files are lightweight public artifacts intended for reproducibility:

- They keep the canonical relation set and the retained raw-relation mappings.
- They do not include patient-level outputs.
- They do not include relation-cluster traces or sampled UMLS triples.

Downstream scripts such as `build_concept_mkg.py`, `build_patient_pkg.py`, and
`train_integrate_formal.py` can consume these schema files directly.

To regenerate schemas instead of using the released ones, run:

```bash
python3 scripts/export_global_relation_inventory.py
python3 scripts/induce_relation_schema.py --dataset shared --task mortality
python3 scripts/induce_relation_schema.py --dataset shared --task readmission_30d
python3 scripts/induce_relation_schema.py --dataset shared --task t2dm_onset
python3 scripts/induce_relation_schema.py --dataset shared --task cad_onset
```
