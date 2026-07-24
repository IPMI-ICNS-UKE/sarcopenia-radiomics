# 04_feature_filtering

Near-zero-variance filtering and Spearman correlation-based redundancy filtering
(Appendix S5, "Near-Zero Variance Filtering" / "Spearman Correlation-Based
Redundancy Filtering"), applied to the stable features produced by
`03_feature_stability`:

- `04_01_conventional_ct` — SMI / MRA
- `04_02_spectral_radiomics` — handcrafted radiomics (conventional CT + spectral maps)
- `04_03_deep_radiomics` — MedDINOv3 embedding dimensions

## Method-specific behavior

- **`04_01_conventional_ct`**: no features are removed (SMI/MRA are fixed
  biomarkers). Variance and Spearman-correlation diagnostics are computed on
  the training split only and saved for QC.
- **`04_02_spectral_radiomics`**: the actual filtering pipeline, learned
  exclusively on cohort-1 training data per map type, then applied unchanged
  to cohort-1 test and cohort 2:
  1. **Near-zero-variance filtering** — drops features with training-set
     variance ≤ `1e-8` (`variance_threshold`).
  2. **Spearman correlation-based redundancy filtering** — average-linkage
     hierarchical clustering on `1 - |Spearman rho|`, cut at
     `spearman_rho_threshold = 0.90`; within each cluster the representative
     feature is the one with the highest stability ICC (from
     `03_02_spectral_radiomics`'s `stability_summary_all_features_{map}.csv`),
     ties broken by training-set variance, then alphabetically — matching
     Appendix S5 exactly.
- **`04_03_deep_radiomics`**: no embedding dimensions are removed (used
  unfiltered, matching the main text). Variance and Spearman-correlation QC
  are computed and saved per map, and the stable embeddings are copied
  through to the filtered outputs unchanged.

## Running

```bash
cd 04_01_conventional_ct && python run.py
cd ../04_02_spectral_radiomics && python run.py
cd ../04_03_deep_radiomics && python run.py
```

Each stage reads `03_feature_stability`'s CSV outputs, so run that first.

## 2D vs 3D

Each `config.py` has a `use_3d: bool` field, switchable either by editing the
default or via `RunConfig(use_3d=True)` (dependent paths are derived in
`__post_init__`).

## Output layout

Each stage writes to `<repo_root>/output/04_0X_<method>[_3d]/`:

- `04_01_conventional_ct`: `filtered_baseline_ct[_cohort_2].csv` plus
  `tables/` and `plots/`.
- `04_02_spectral_radiomics` / `04_03_deep_radiomics`: one subfolder per
  spectral map (e.g. `musclefat/`, `40kev/`), each with its own `tables/` and
  `plots/`, plus a top-level `filtering_summary_all_maps.csv` (or
  `analysis_summary_all_maps.csv` for deep radiomics).

## Notes

- `ground_truth_path_cohort_1` / `ground_truth_path_cohort_2` in each
  `config.py` point to local data storage and must be adjusted to your own
  paths before running.
- Requires `pandas`, `numpy`, `scipy`, `matplotlib`.
