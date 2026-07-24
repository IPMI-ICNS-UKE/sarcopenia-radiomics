# 05_feature_selection

LASSO-based feature selection (Appendix S5, "Selection of Informative
Handcrafted Radiomics Features") and final biomarker/signature export for
modeling, per imaging feature family:

- `05_01_conventional_ct` — SMI / MRA signature export, **no selection**
  (these are fixed biomarkers, used as-is).
- `05_02_01_spectral_radiomics_mean_fraction` — exports the plain mean-value
  spectral CT biomarkers (muscle fat fraction, VNC, electron density, iodine,
  effective Z, monoenergetic maps). Reads directly from `02_feature_extraction`'s
  raw handcrafted-radiomics output and takes the `original_firstorder_Mean_{map}`
  feature, which is mathematically identical to the simple ROI mean voxel value
  (Appendix S4, eq. 8) — no selection, since these are biomarkers, not filtered
  radiomics features.
- `05_02_02_spectral_radiomics_signature` — the actual LASSO selection
  pipeline, run separately per spectral map on `04_feature_filtering`'s
  filtered/stable handcrafted radiomics features.
- `05_03_deep_radiomics` — MedDINOv3 embedding signature export, **no
  selection** (matches the main text: deep-radiomics features are used
  unfiltered). Reads directly from `02_feature_extraction`'s raw output,
  restricted to `status == "ok"` and `mask == "a_original"`.

## LASSO selection pipeline (`05_02_02`)

According to Appendix S5:
1. Median imputation (training-set medians) and standardization
   (`StandardScaler`, fit on training data only), inside an sklearn `Pipeline`.
2. L1-penalized logistic regression (`solver="saga"`, `class_weight="balanced"`,
   `max_iter=10000`, `tol=1e-4`, `random_state=42`).
3. Hyperparameter tuning via 5-fold stratified, shuffled cross-validation
   (`random_state=42`), scored on AUC, over the 16-point log-spaced `C` grid
   (0.001–100). Ties are broken toward the smaller `C` — this falls out of
   `GridSearchCV`'s default behavior (it picks the first-ranked entry in grid
   order, and the grid is ascending), matching the paper's stated tie-break
   rule without needing extra logic.
4. Features with non-zero coefficients at the selected `C` are the final
   signature.

A bootstrap-based **selection-frequency** table (how often each feature gets a
non-zero coefficient across repeated resamples at the chosen `C`) is also
computed and saved as an additional stability diagnostic. It does not affect
which features are selected unless `selection_frequency_threshold > 0` (default
`0.0`, i.e. inactive) — selection is otherwise governed solely by the non-zero
coefficient at the tuned `C`, as in the manuscript.

## Running

```bash
python 05_01_conventional_ct/run.py
python 05_02_01_spectral_radiomics_mean_fraction/run.py
python 05_02_02_spectral_radiomics_signature/run.py
python 05_03_deep_radiomics/run.py
```

`utils/` is a proper Python package; each `run.py` adds `05_feature_selection/`
to `sys.path` and imports it as `utils.config`, `utils.io`, `utils.selection`,
`utils.signature`, `utils.plots`.

## 2D vs 3D

Each config class in `utils/config.py` has a `use_3d: bool` field, switchable
either by editing the default or via e.g. `SpectralSignatureConfig(use_3d=False)`
(dependent paths are derived in `__post_init__`). Defaults:
`ConventionalCTConfig.use_3d = False`; the other three default to `True` —
matching which mode `04_feature_filtering`/`02_feature_extraction` default to.
As in `04_feature_filtering`, each config's input path resolves to
`<repo_root>/output/..._<mode>/`, so make sure the `use_3d` you run here
matches the mode of the upstream stage's output you're reading from — a
mismatch fails fast with a missing-file error rather than silently mixing 2D
and 3D data.

## Output layout

Each stage writes to `<repo_root>/output/05_0X_<method>[_3d]/`:

- `05_01_conventional_ct`: `signature_baseline_ct[_cohort_2].csv` plus
  `tables/` (split-mismatch QC and a JSON summary).
- `05_02_01_spectral_radiomics_mean_fraction` / `05_02_02_spectral_radiomics_signature`
  / `05_03_deep_radiomics`: one subfolder per map, each with its exported
  signature CSV(s) and a `tables/` (and, for `05_02_02`, `plots/` — CV tuning
  curve, coefficient paths, selected coefficients, selection frequency).

## Notes on paths

- `ground_truth_path` / `ground_truth_path_cohort_2` in `utils/config.py`
  (`BaseStage05Config`) point to local data storage and must be adjusted to
  your own paths before running.
- Requires `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
