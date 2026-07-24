# 07_02_segmentation_agreement

Quantifies pairwise agreement between the two manual readers and the
automated segmentation model on the 15-patient manual-annotation subset of
cohort 1, and renders both the quantitative Dice boxplot and qualitative
per-patient contour-overlay figures. Corresponds to Appendix S3
("Segmentation Agreement Analysis") and Figures S3/S4.

## What it produces (in `output/07_02_segmentation_agreement/`)

- `segmentation_agreement_per_level.csv` — one row per (patient, L2/L3/L4,
  pair): the raw Dice coefficient.
- `segmentation_agreement_per_patient.csv` — the 3 level-specific Dice values
  averaged within each patient (patient is the independent unit of analysis,
  per Appendix S3).
- `segmentation_agreement_summary.csv` — mean ± SD, median, IQR across the 15
  patients, per pair (Reader 1 vs Reader 2, Reader 1 vs Automated, Reader 2
  vs Automated).
- `segmentation_agreement_boxplot.png` — 300 dpi publication figure
  (box + per-patient scatter), matching Figure S3.
- `segmentation_overlay_<worst|median|best>_<patient_id>_<level>.png`
  (from `run_overlay.py`) — CT slice with all three raters' contours
  overlaid, for the patients with the lowest, median, and highest overall
  Dice agreement, matching Figure S4's style.

## Layout

| File | Role |
|---|---|
| `config.py` | Cohort/run config (reuses `02_feature_extraction/utils/common_config.py`), reader/level identifiers, output paths. |
| `data_loader.py` | Selects the `manual_annotation == 1` patient subset and extracts the L2/L3/L4 masks for both readers and the automated model, reusing `manual_and_model_masksets()` from `02_feature_extraction/utils/mask.py`. |
| `dice_utils.py` | Per-level Dice → per-patient average → across-patient mean/SD/median/IQR, exactly matching the Appendix S3 aggregation order. |
| `plot_utils.py` | The box + scatter Dice figure. |
| `overlay_utils.py` | The qualitative contour-overlay figure for one patient/level. |
| `run.py` | Orchestrates the quantitative Dice analysis (steps 1–4). |
| `run_overlay.py` | Separate entry point for the qualitative overlays; picks the worst/median/best-agreement patients from `run.py`'s per-patient CSV, so `run.py` must be run first. |

## Methodology cross-check against Appendix S3

Matches exactly:
- 15-patient subset selected via `manual_annotation == 1`.
- Dice computed per level (L2/L3/L4) → averaged within patient → mean/SD
  summarized across the 15 patients, patient as the independent unit.
- Three pairwise comparisons (R1–R2, R1–Auto, R2–Auto), no method treated as
  reference standard.