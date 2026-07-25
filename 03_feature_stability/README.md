# 03_feature_stability

Feature stability testing (Appendix S5, "Feature Stability") for the three
feature families produced by `02_feature_extraction`:

- `03_01_conventional_ct` — SMI / MRA
- `03_02_spectral_radiomics` — handcrafted radiomics (conventional CT + spectral maps)
- `03_03_deep_radiomics` — MedDINOv3 embedding dimensions

Each computes ICC(2,1) (two-way random-effects, absolute-agreement, single-measure
intraclass correlation) in two scenarios:

- **Simulated masks** — dilation/erosion perturbations of the automated muscle
  mask (`a_original`, `a_dilate_p1/p2`, `a_erode_p1/p2`), across all patients.
- **Manual vs model** — the 15-patient manual-segmentation subset (Appendix S3),
  comparing the two readers and the automated model (`i`/`j`/`a`) at the L2/L3/L4
  levels. 2D-only, since manual annotations only exist for single axial slices.

## Method-specific behavior

- `03_01_conventional_ct` and `03_03_deep_radiomics` compute and report stability
  but do not drop any features — SMI/MRA are fixed biomarkers, and deep-radiomics
  embedding dimensions are used unfiltered in the manuscript.
- `03_02_spectral_radiomics` performs the actual ICC-based feature filtering
  (ICC(2,1) > 0.80, matching Appendix S5) on the training split, separately per
  map type, and writes the resulting stable feature sets. `selection_mode`
  controls which scenario a feature must pass: `"both"` (simulated **and**
  manual) for 2D, `"simulated"` only for 3D (no manual-annotation data exists
  in 3D mode).
- Tables are written to `tables/`; plots (ICC heatmaps, ICC histograms, and
  per-feature profile plots across mask variants/raters) to `plots/`.

## Running

Each stage reads the corresponding `02_feature_extraction` CSV outputs, so run
that first. From this folder:

```bash
cd 03_01_conventional_ct && python run.py
cd ../03_02_spectral_radiomics && python run.py
cd ../03_03_deep_radiomics && python run.py
```

`utils/` is a proper Python package (`utils/__init__.py`); each `run.py` adds
`03_feature_stability/` to `sys.path` and imports it as `utils.icc`, `utils.io`,
`utils.plots`, `utils.tables`.

## 2D vs 3D

Each `config.py` has a `use_3d: bool` field (default `False` for
`03_01_conventional_ct`, `True` for `03_02_spectral_radiomics` and
`03_03_deep_radiomics`, matching which mode the manuscript reports for each
feature family). All dependent paths (input CSV, output directory, and — for
`03_01`/`03_02` — `feature_cols`/`selection_mode`) are derived from `use_3d` in
`__post_init__`, so it can be switched either by editing the default or at
construction time, e.g. `RunConfig(use_3d=True)`.

## Output layout

Each stage writes to `<repo_root>/output/03_0X_<method>[_3d]/`:

- `03_01_conventional_ct`: `stable_baseline_ct[_cohort_2].csv` plus `tables/`
  and `plots/` (ICC tables/heatmaps/profiles for cohort 1 train/test and,
  separately, cohort 2).
- `03_02_spectral_radiomics` / `03_03_deep_radiomics`: one subfolder per
  spectral map (e.g. `musclefat/`, `40kev/`), each with its own `tables/` and
  `plots/`, plus a top-level `tables/map_type_selection_summary.csv` (or
  `map_type_stability_summary.csv` for deep radiomics).

## Notes

- `ground_truth_xlsx` in each `config.py` points to local data storage and
  must be adjusted to your own path before running.
- Requires `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.
