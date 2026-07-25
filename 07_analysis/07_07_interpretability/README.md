# 07_07_interpretability

Interpretability analysis for the 3D CT+muscle-fat handcrafted-radiomics
model (`mf_ct_scores_clinical_3d`). Produces a train-set forest plot (odds
ratios) and test-set SHAP bee-swarm plots, at two levels of granularity.

## Layout

| File | Role |
|---|---|
| `config.py` | Model identity (`MODEL_NAME`), score/clinical feature lists, display labels, bootstrap/plot settings, paths. |
| `dataset.py` | Loads train/test1/test2 splits with ground truth + the raw radiomics feature tables for each map (mirrors `06_02_02/dataset.py`, scoped to this stage). |
| `forest.py` | Loads the already-fitted `.joblib` model saved by `06_modeling`; computes multivariable OR (from the fitted coefficients, with bootstrap 95% CI) and univariable OR (one unpenalized logistic regression per predictor), at both score and feature level. |
| `shap_analysis.py` | Computes exact SHAP values (`shap.LinearExplainer`) for the same two levels, on test1/test2. |
| `plots.py` | Forest-plot and SHAP-bee-swarm rendering (300 dpi, static PNG only — no interactive output, by design). |
| `label_utils.py` | Feature-name → display-label formatting, shared by `forest.py`/`shap_analysis.py`. |
| `run.py` | Orchestrates load model → forest (train) → SHAP (test1/test2) → save tables + plots. |

## Running

```bash
python run.py                    # all steps
python run.py --no-feature-level # score level only (faster)
python run.py --n-boot 500       # override bootstrap iterations
```

## Output layout

`output/07_07_interpretability/`:

- `interpretability_train_forest_score.png` / `..._feature.png` — forest
  plots of odds ratios (univariable + bootstrap-CI multivariable), at the
  **score level** (3 combined-stage inputs: muscle-fat score, CT score, sex)
  and the **feature level** (all underlying radiomics features from each
  map's stage-1 model, plus sex).
- `interpretability_test1_shap_score.png` / `_feature.png`,
  `interpretability_test2_shap_score.png` / `_feature.png` — SHAP bee-swarm
  plots on both test sets, same two levels.
- `tables/multivariable_or_train_{score,feature}.csv`,
  `tables/shap_values_{split}_{level}.csv` — the underlying numeric tables.

## Notes

- Depends on `06_02_02_spectral_radiomics_signature_3d`'s saved joblib model
  and coefficient table — run that stage first.
