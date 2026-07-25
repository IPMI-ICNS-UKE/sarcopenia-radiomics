# 06_modeling

Predictive modeling stage: fits elastic-net logistic/linear models on the
features produced by `05_feature_selection` and evaluates them with repeated
nested cross-validation, an internal (cohort 1) held-out test set, and an
external (cohort 2, liver-cohort) validation set (Appendix S6/S7):

- `06_01_conventional_ct` — conventional-CT SMI/MRA baseline
- `06_02_01_spectral_radiomics_mean_fraction` — per-map spectral mean-fraction biomarkers
- `06_02_02_spectral_radiomics_signature` — per-map LASSO-selected radiomics signatures (spectral maps)
- `06_03_deep_radiomics` — deep (foundation-model) radiomics embeddings

Shared logic is stored in `utils/`:
- `models_base.py`: `ModelSpec` + pipeline construction (elastic-net logistic/linear).
- `cv_base.py`: repeated nested-CV loop, refit-on-full-train, test/cohort-2 evaluation.
- `stats.py`: bootstrap CIs, Youden threshold, calibration, decision-curve analysis.
- `plots.py`: ROC, calibration, decision-curve, coefficient, scatter plots.
- `io_utils.py`: metric-table / CI-column formatting helpers.

## Layout

Each submodule follows the same internal shape:

| File | Role |
|---|---|
| `config.py` | `PathsConfig`, `CVConfig`, `ElasticNetConfig`, `TaskConfig`/`TASKS`, `ModelBlockConfig`, bundled into `RunConfig` (singleton `RCFG`). |
| `dataset.py` | Reads the ground-truth table + upstream feature CSVs, applies cohort filters, merges, returns one long DataFrame with a `cohort` column (`cohort1`/`cohort2`) and a `split` column (`train`/`test`). |
| `models.py` | Declares the catalogue of `ModelSpec`s to fit for a given task (see "Model taxonomy" below). |
| `cv.py` *(06_01 only)* | Thin wrapper around `cv_base.run_cv_for_specs` for that submodule's spec list. |
| `run_task.py` | Orchestrates one task: build dataset → run nested CV → refit on full cohort-1 train → evaluate on cohort-1 test and cohort-2 → save metrics/tables/predictions → generate plots. |
| `run_all.py` | Loops `run_task` over all tasks in `TASKS`. |
| `run_dataset.py` | Standalone dataset-inspection entry point (build + print shapes/counts). |

## Model taxonomy (`ModelSpec.kind`)

| `kind` | Meaning |
|---|---|
| `direct` | Single-stage elastic-net fit directly on `base_features` (imaging only; `fit_direct()` ignores `clinical_features`). |
| `score_plus_clinical` | Stage 1 fits imaging features → one score; stage 2 refits on `[score, sex]`. |
| `multi_score` | One stage-1 sub-model per feature group (e.g. per map); stage 2 combines the scores (no clinical). |
| `multi_score_plus_clinical` | Same as `multi_score`, plus clinical features in stage 2. |
| `two_scores_plus_clinical` | Specialization of `multi_score_plus_clinical` for exactly two groups (CT + MuscleFat deep radiomics in `06_03`). |


## Running

```bash
cd 06_01_conventional_ct && python run_all.py
cd ../06_02_01_spectral_radiomics_mean_fraction && python run_all.py
cd ../06_02_02_spectral_radiomics_signature && python run_all.py
cd ../06_03_deep_radiomics && python run_all.py
```

Each submodule reads `05_feature_selection`'s outputs, so run that first.
`run_all.py` loops over all three tasks (`config.py:TASKS`):
`sarcopenia_composite_cls`, `hand_grip_reg`, `chair_rise_cls`.

## 2D vs 3D

Every `PathsConfig` has a `use_3d: bool` flag (default `True`) that switches
both the upstream feature directory and this stage's output directory between
a `_3d` suffix and no suffix.

## Output layout

`output/<stage_name>[_3d]/{metrics,predictions,models,tables,plots,plots_interactive}`,
where `<stage_name>` is e.g. `06_01_conventional_ct`, `06_02_02_spectral_radiomics_signature`.

## Notes

- Ground-truth table paths in each `config.py` point to local data storage
  and must be adjusted to your own paths before running.
