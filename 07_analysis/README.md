# 07_analysis

Downstream analysis, statistical comparison, and figure/table
generation, built on top of `06_modeling`'s trained models and metrics.
Each subfolder is independent and has its own README:

| Folder | Purpose |
|---|---|
| [`07_01_cohort_description`](07_01_cohort_description/README.md) | Manuscript Table 1 and cohort-specific clinical-characteristics tables. |
| [`07_02_segmentation_agreement`](07_02_segmentation_agreement/README.md) | Manual-reader vs. automated segmentation agreement (Dice). |
| [`07_03_feature_agreement`](07_03_feature_agreement/README.md) | Segmentation-variability effect on SMI/MRA/muscle-fat-fraction (ICC(2,1)). |
| [`07_04_comparison`](07_04_comparison/README.md) | Cross-model statistical comparison tables and p-value heatmaps. |
| [`07_05_roc`](07_05_roc/README.md) | ROC curves for functional sarcopenia impairment classification. |
| [`07_06_cohens_kappa`](07_06_cohens_kappa/README.md) | Cohen's κ agreement heatmaps across classification approaches. |
| [`07_07_interpretability`](07_07_interpretability/README.md) | Forest plots (odds ratios) and SHAP analysis for the selected model. |
| [`07_08_probability_distribution`](07_08_probability_distribution/README.md) | Case-level predicted-probability bar charts. |
| [`07_09_case_analysis`](07_09_case_analysis/README.md) | Representative TP/TN/FP/FN case illustrations and summary table. |

All subfolders read `06_modeling`'s saved metrics/predictions/models, so run
that stage first.
