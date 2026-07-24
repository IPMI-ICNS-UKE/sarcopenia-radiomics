# 01_data_description

Builds the baseline characteristics table (Table 1 in the manuscript) and runs an
image-intensity QC pass used to choose clipping/normalization ranges for downstream
feature extraction (Appendix S4).

## Scripts

| Script | Purpose |
|---|---|
| `sct_data_loader.py` | `DataLoader` class: reads a cohort's `ground_truth.xlsx`, applies the approved filters (`use`, `image_present`, `labels_present` all == 1), and splits Cohort 1 into training (`test_temporal == 0`) and primary test (`test_temporal == 1`); returns the full filtered table for Cohort 2. |
| `sct_statistics.py` | Statistics helpers: continuous-variable summaries (mean ± SD, range, IQR), sex/ECOG counts, and pairwise group comparisons (Welch t-test or Mann-Whitney U for continuous variables depending on Shapiro normality; chi-square or Fisher exact for categorical variables), assembled into a publication-style long-format table via `build_data_description_table_three_groups`. |
| `run_data_description.py` | Entry point: loads training / primary test / secondary test sets and writes `data_description_statistics.xlsx` (dataset overview + baseline characteristics table, i.e. Table 1). |
| `run_intensity_range_assessment.py` | Entry point: for every conventional CT and spectral CT map, computes whole-image and within-muscle-mask intensity statistics (min/max/mean/quartiles) per patient, then aggregates across patients into `intensity_ranges.xlsx`. Used to inform the clipping bounds documented in Appendix S4. |

## Usage

Run from within this folder so the local imports (`sct_data_loader`, `sct_statistics`)
resolve correctly:

```bash
cd 01_data_description
python run_data_description.py
python run_intensity_range_assessment.py
```

Outputs are written to `<repo_root>/output/01_data_statistics/`.

## Notes

- Input data paths (`COHORT1_TABLE_PATH`, `COHORT2_TABLE_PATH` in
  `run_data_description.py`; `table_path`, `images_root`, `labels_root` in
  `run_intensity_range_assessment.py`) point to local data storage and must be
  adjusted to your own paths before running.
- Requires `pandas`, `numpy`, `scipy`, `openpyxl`, `SimpleITK`, `tqdm`.
