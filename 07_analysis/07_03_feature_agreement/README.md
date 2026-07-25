# 07_03_feature_agreement

Quantifies how segmentation variability (Reader 1 vs Reader 2 vs automated)
propagates into the actual biomarker values — 2D SMI, 2D MRA, and the 2D
spectral CT muscle fat fraction — on the same 15-patient manual-annotation
subset used by `07_02_segmentation_agreement`. Corresponds to Appendix S4,
"Biomarker Agreement across Segmentation Methods."

## Layout

| File | Role |
|---|---|
| `config.py` | Cohort/run config (reuses `02_feature_extraction/utils/common_config.py`), rater identifiers, biomarker list, output paths. |
| `data_loader.py` | Selects the `manual_annotation == 1` subset; for each patient, extracts the L3 CT slice, L3 muscle-fat-map slice, and each rater's L3 mask (`manual_and_model_masksets`, reused from `02_feature_extraction`). |
| `feature_utils.py` | Computes SMI/MRA/muscle-fat-fraction per rater mask, reusing the exact per-slice computation from `02_feature_extraction/utils/clinical_features.py` (HU-thresholded mask → area/mean-intensity). |
| `icc_utils.py` | ICC(2,1) with 95% CI via `pingouin.intraclass_corr` (`Type == "ICC2"` — two-way random, absolute agreement, single measure, matching ICC(2,1) exactly). Point estimate cross-checked against the from-scratch formula in `03_feature_stability/utils/icc.py`. |
| `run.py` | Orchestrates all of the above. |

## Running

```bash
python run.py
```

## Output layout

`output/07_03_feature_agreement/`:

- `feature_agreement_raw_values.csv` — one row per (patient, rater): `smi_2d`,
  `mra_2d`, `ff_2d`.
- `feature_agreement_icc_summary.csv` — one row per biomarker: ICC(2,1) point
  estimate, 95% CI, F-statistic, df1/df2, p-value, n_subjects, n_raters.

