# 07_06_cohens_kappa

Generates pairwise Cohen's κ agreement heatmaps between the functional
sarcopenia impairment reference standard, the Van der Werf SMI/MRA cut-off
definitions, and the imaging-model-derived classifications.

## Layout

| File | Role |
|---|---|
| `config.py` | Ground-truth paths, model-name pins per source folder, Van der Werf sex-dependent thresholds, display labels/order, plot aesthetics. |
| `data_loader.py` | Builds one aligned per-patient DataFrame per split: merges ground truth, applies Van der Werf thresholds, and merges in each model's `pred_label` from its `06_*` predictions CSV. |
| `kappa_utils.py` | Computes the full pairwise Cohen's κ matrix (`sklearn.cohen_kappa_score`) with pairwise-complete-case handling. |
| `plot_utils.py` | Renders the lower-triangle heatmap. |
| `run.py` | Orchestrates loading → κ computation → plotting for all three splits. |

## Running

```bash
python run.py
```

## Output layout

`output/07_06_cohens_kappa/`:

- `cohens_kappa_train.png`
- `cohens_kappa_test_1.png`
- `cohens_kappa_test_2.png`
