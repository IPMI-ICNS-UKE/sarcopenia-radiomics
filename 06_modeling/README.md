# 06_modeling

Predictive modeling stage: fits elastic-net logistic/linear models on the
features produced by `05_feature_selection` and evaluates them with repeated
nested cross-validation, an internal (cohort 1) held-out test set, and an
external (cohort 2, liver-cohort) validation set. This is the stage that
produces the coefficients, AUCs/RMSEs, calibration, and decision-curve results
reported in the Main Text (Table 3/4) and Appendix S6/S7 of the manuscript.

## Layout

```
06_modeling/
├── utils/                                     shared engine, used by every submodule
│   ├── models_base.py     ModelSpec + pipeline construction (elastic-net logistic/linear)
│   ├── cv_base.py         repeated nested-CV loop, refit-on-full-train, test/cohort-2 evaluation
│   ├── stats.py           bootstrap CIs, Youden threshold, calibration, decision-curve analysis
│   ├── plots.py           ROC, calibration, decision-curve, coefficient, scatter plots
│   └── io_utils.py        metric-table / CI-column formatting helpers
├── 06_01_conventional_ct/                     conventional-CT SMI/MRA baseline
├── 06_02_01_spectral_radiomics_mean_fraction/  per-map spectral mean-fraction biomarkers
├── 06_02_02_spectral_radiomics_signature/      per-map LASSO-selected radiomics signatures (spectral maps)
└── 06_03_deep_radiomics/                       deep (foundation-model) radiomics embeddings
```

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

Outputs are written to `output/<stage_name>[_3d]/{metrics,predictions,models,tables,plots,plots_interactive}`,
where `<stage_name>` is e.g. `06_01_conventional_ct`, `06_02_02_spectral_radiomics_signature`, etc.

## Tasks

Every submodule fits the same three prediction targets (`config.py:TASKS`):

- `sarcopenia_composite_cls` — classification, composite functional-sarcopenia label.
- `hand_grip_reg` — regression, continuous hand-grip strength.
- `chair_rise_cls` — classification, chair-rise-test impairment.


## Cross-validation / statistics (matches Appendix S6/S7)

- Outer loop: `RepeatedStratifiedKFold`/`RepeatedKFold`, 5 splits × 5 repeats (`CVConfig`).
- Inner loop: 5-fold grid search over elastic-net `l1_ratio ∈ {0, 0.5, 1}` and
  `C ∈ {0.01…100}` (classification) / `alpha ∈ {0.01…100}` (regression).
- Final model per task/spec is refit on the full cohort-1 training split and
  evaluated on the cohort-1 held-out test split and the cohort-2 external set.
- Metrics: bootstrap CIs (2000 resamples), Youden-index threshold selection
  (`threshold_selection_method="youden_train"`), calibration intercept/slope +
  Brier score, decision-curve net benefit over `[0.10, 0.80]` (`DecisionCurveConfig`).

## 2D vs 3D

Every `PathsConfig` has a `use_3d: bool` flag (default `True`) that switches
both the upstream feature directory and this stage's output directory between
a `_3d` suffix and no suffix, e.g. `output/06_01_conventional_ct_3d` vs.
`output/06_01_conventional_ct`. This is resolved once per instance in
`__post_init__` (not baked into the class at import time), so
`PathsConfig(use_3d=False)` correctly switches every derived path.
