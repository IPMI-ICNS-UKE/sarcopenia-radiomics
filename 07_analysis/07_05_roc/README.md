# 07_05_roc

Generates the ROC curve figures for classification of functional sarcopenia
impairment — both a publication-ready static PNG (300 dpi) and a
self-contained interactive HTML per data split. Reproduces Figure 3A–C of
the manuscript.

## What it produces (in `output/07_05_roc/`)

- `ROC_train.png` / `ROC_train.html`
- `ROC_test_1.png` / `ROC_test_1.html`
- `ROC_test_2.png` / `ROC_test_2.html`

Each figure overlays 6 curves: 2D SMI, 2D MRA, 3D SMI, 3D MRA, 3D muscle-fat
mean fraction, and the 3D handcrafted-radiomics (CT + muscle fat) model —
matching Figure 3's caption exactly.

## Layout

| File | Role |
|---|---|
| `config.py` | Model sources/display names/order, dataset-split file names, plot aesthetics (colorblind-safe palette + distinct dash patterns per curve for greyscale-safe printing). |
| `data_loader.py` | Reads each model's metrics `.xlsx` (AUC + CI95) and predictions `.csv` from its `06_*` output folder. |
| `roc_utils.py` | Computes per-model FPR/TPR via `sklearn.roc_curve`; AUC/CI pulled from the pre-computed metrics file (falls back to `sklearn.roc_auc_score` with no CI if unavailable). |
| `plot_roc_static.py` | Matplotlib/seaborn static PNG, Radiology-journal styling (Arial, 300 dpi, single-column width, distinguishable in greyscale). |
| `plot_roc_interactive.py` | Plotly interactive HTML with per-point hover tooltips. |
| `run.py` | Orchestrates both outputs for all three splits. |
