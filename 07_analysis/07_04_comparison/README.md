# 07_04_comparison

Cross-stage statistical comparison: pulls the metrics/predictions already
produced by `06_modeling` for every trained model, runs pairwise significance
testing against the SMI/MRA baselines, and exports publication-ready
comparison tables, p-value heatmaps, and a plain-text interpretation report.
This is a downstream reporting stage — it does not fit any models itself.

## What it produces (in `output/07_04_comparison/`)

- `comparison_cls.xlsx` — sarcopenia-composite classification: AUC, balanced
  accuracy, sensitivity, specificity (with 95% CI), plus DeLong p-values vs.
  SMI and MRA, for train / test-1 / test-2.
- `comparison_reg.xlsx` — hand-grip regression: RMSE, MAE, R², Spearman
  correlation, plus Wilcoxon signed-rank p-values vs. SMI and MRA.
- `comparison_cls_2.xlsx` — same structure as `comparison_cls.xlsx` for the
  chair-rise classification target.
- `heatmap_pvalue_<task>_<subset>.png` — 300 dpi p-value heatmaps (SMI/MRA
  rows × muscle-fat mean-fraction / signature / CT+MF-radiomics columns).
- `pvalue_interpretation.txt` — human-readable verdict ("SIGNIFICANT" /
  "NOT SIGNIFICANT") for every configured comparison pair, across every
  task and data subset.

## Layout

| File | Role |
|---|---|
| `config_extended.py` | Single source of config: every individual spectral map (2D+3D, mean-fraction/signature/deep) as a separate comparable method, ~70 entries total. |
| `data_loader.py` | Reads each method's metrics `.xlsx` and predictions `.csv` from its `06_*` output folder. |
| `statistical_tests.py` | DeLong test (custom implementation) for AUC comparison; Wilcoxon signed-rank test on paired absolute errors for regression; BH/Bonferroni/none correction. |
| `table_builder.py` | Assembles and Excel-formats `comparison_cls.xlsx` / `comparison_reg.xlsx` / `comparison_cls_2.xlsx`. |
| `heatmap_plotter.py` | Renders the 300 dpi p-value heatmaps. |
| `interpretation_report.py` | Writes `pvalue_interpretation.txt`. |
| `run_comparison.py` | Orchestrates all of the above. |


## Statistical methodology

- **Classification**: two-sided DeLong test (self-contained implementation
  in `statistical_tests.py`, verified against the standard DeLong 1988
  variance/covariance formulas — same formula family used to report the
  manuscript's Table 2 comparisons).
- **Regression**: two-sided Wilcoxon signed-rank test on paired absolute
  errors (`|error_new| - |error_baseline|`) — non-parametric, exploits the
  paired-patient structure, appropriate for the small sample sizes here.
- **Multiple-comparison correction**: configurable (`fdr_bh` / `bonferroni`
  / `none`), applied within each task × data-subset block (6+ comparisons
  per block against SMI/MRA baselines). Default is `fdr_bh`, matching
  Appendix S7: *"Benjamini–Hochberg false-discovery-rate adjustment was used
  for multiple comparisons correction... performed separately within the
  training set, primary test set, and secondary test set."*
