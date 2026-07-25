# 07_09_case_analysis

Generates representative true-positive / true-negative / false-positive /
false-negative case illustrations (CT + muscle-fat heatmap) and a matching
Excel summary table, for a hand-picked patient per outcome category in each
of train / test-1 / test-2. 

## Layout

| File | Role |
|---|---|
| `config.py` | Model identity, hardcoded `CASES` dict (patient IDs per set/case-type), imaging/figure/table sub-configs, paths. |
| `imaging.py` | Loads CT + MuscleFat map + abdominal-wall/paravertebral masks for one patient, resamples to a common grid, locates the L3 slice via vertebra centroid, applies HU-based muscle refinement, returns a `PatientImageContext`. |
| `scoring.py` | Loads the saved `mf_ct_scores_clinical_3d` joblib model and runs it on one patient's assembled feature row to get per-map scores, combined score, predicted probability, threshold. |
| `figure_export.py` | Renders the single-case figure. |
| `table_export.py` | Builds and color-formats the Excel case table. |
| `models_base.py` | A customized fork of `06_modeling/utils/models_base.py`, used only for its prediction-dispatch path (`predict_multi_score`). |
| `run_case_analysis.py` | Orchestrates: load GT → load images → score → build figure → build table, per set. |

## Running

```bash
python run_case_analysis.py                  # all sets
python run_case_analysis.py --sets train      # single set
```

## Output layout (currently)

`output/07_09_case_analysis/`:

- `figures/case_{set}_{TP,TN,FP,FN}.png` — one PNG **per case** (4 separate
  files per set), each a single-panel CT image with a semi-transparent
  muscle-fat-fraction heatmap overlay + colorbar, titled TP/TN/FP/FN.
- `tables/cases_{set}.xlsx` — one row per case type, columns grouped and
  color-coded by category (identifiers / demographics / functional tests /
  model scores), per `config.py:_COLUMN_SPEC`.
