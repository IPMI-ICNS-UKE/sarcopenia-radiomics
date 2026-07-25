# Pipeline Output

Generated pipeline outputs (CSVs, tables, plots, models) land here, one
subfolder per stage. 

| Path | Written by | Contents |
|---|---|---|
| `01_data_statistics/` | [`01_data_description`](../01_data_description/README.md) | Table 1 baseline-characteristics workbook, image-intensity QC. |
| `02_01_conventional_ct/`, `02_02_spectral_radiomics/`, `02_03_deep_radiomics/` | [`02_feature_extraction`](../02_feature_extraction/README.md) | Raw per-patient feature CSVs, one per method/mode/cohort (`raw_*[_cohort_2][_3d].csv`). |
| `03_0X_<method>[_3d]/` | [`03_feature_stability`](../03_feature_stability/README.md) | ICC(2,1) stability-filtered feature CSVs, plus `tables/`/`plots/` per map. |
| `04_0X_<method>[_3d]/` | [`04_feature_filtering`](../04_feature_filtering/README.md) | Near-zero-variance/correlation-filtered feature CSVs, plus `tables/`/`plots/` per map. |
| `05_0X_<method>[_3d]/` | [`05_feature_selection`](../05_feature_selection/README.md) | Final exported biomarker/signature CSVs (LASSO-selected where applicable), plus `tables/`/`plots/`. |
| `06_0X_<method>[_3d]/{metrics,predictions,models,tables,plots,plots_interactive}` | [`06_modeling`](../06_modeling/README.md) | Trained model joblib files, per-split predictions, performance metrics, coefficient tables, ROC/calibration plots. |
| `07_01_cohort_description/` … `07_09_case_analysis/` | [`07_analysis`](../07_analysis/README.md) | Manuscript figures and tables — one subfolder per `07_0X_*` analysis, see each subfolder's own README for its exact files. |

`00_data_utilities` operates on raw cohort data directories directly and does
not write here.

