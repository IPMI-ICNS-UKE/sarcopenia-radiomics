# 07_08_probability_distribution

Case-level predicted-probability bar charts for the 3D CT+muscle-fat
handcrafted-radiomics model (`mf_ct_scores_clinical_3d`), one per data
split. 

## Layout

| File | Role |
|---|---|
| `config.py` | Model identity, input/output paths, color palette (solid = male, faded = female; red = sarcopenia, blue = no sarcopenia), bar-gap and threshold-line styling. |
| `data_loader.py` | Loads a split's prediction CSV, filters to `MODEL_NAME`, merges in `sex` from the ground-truth Excel. |
| `plot_static.py` | 300 dpi matplotlib bar chart — hatched (`xx`) bars for misclassified (FP/FN) cases, dashed Youden-threshold line, per-patient x-tick labels. |
| `plot_interactive.py` | Plotly HTML equivalent — misclassified cases get an `x` marker overlay (Plotly has no native per-bar hatching) plus full hover tooltips (patient ID, probability, true label, prediction, sex, TP/TN/FP/FN). |
| `run.py` | Loads all three splits, resolves one shared decision threshold, recomputes `pred_label` from it for consistent FP/FN highlighting across all three panels, then renders both output formats per split. |

## Running

```bash
python run.py
```

## Shared decision threshold

The Youden threshold is selected once from the aggregated
training-set out-of-fold predictions and then applied unchanged to both test
sets. `run.py` reads `selected_threshold` preferring `test_1` → `test_2` →
`train` (all three prediction files store the same fixed value) and
recomputes `pred_label` from that single value across every split, so the
FP/FN hatching is consistent with the one cut-point drawn on all three
panels.

## Output layout

`output/07_08_probability_distribution/`:

- `probability_distribution_train.png` / `.html`
- `probability_distribution_test_1.png` / `.html`
- `probability_distribution_test_2.png` / `.html` (skipped if the cohort-2
  predictions file isn't present yet)
