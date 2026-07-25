# 07_01_cohort_description

Builds the manuscript's cohort/patient-characteristics tables (Main Text
Table 1, and the cohort-specific clinical-characteristics tables) as a
formatted Excel workbook, plus an optional "Statistical methods" sheet
documenting exactly which test was used and the resulting p-value for every
compared variable.

## Layout

| File | Role |
|---|---|
| `config.py` | Paths, column-name mapping (`Columns`), categorical encodings (sex, tumor type, Child-Pugh grade), group/split definitions, formatting and statistics options, bundled into `RunConfig`. |
| `run.py` | CLI entry point: parses args, loads groups, builds all tables, writes the workbook. |
| `build_tables.py` | Study-specific logic: turns loaded groups into `TableModel`s for Table 1/2/3 and the methods sheet. |
| `utils/data_loading.py` | Reads both ground-truth Excel tables, applies the `use == 1` filter, and splits into `training` / `test1` / `test2` groups. |
| `utils/statistics.py` | Group-comparison tests: Shapiro-Wilk + Levene → one-way ANOVA or Kruskal-Wallis for continuous variables; Pearson chi-square → Fisher exact (2×2) or seeded Monte-Carlo chi-square (r×c) for categorical variables when expected counts are low. |
| `utils/formatters.py` | Cell-text formatting (`mean ± SD`, `(min–max)`, `n (%)`, p-value with `<cutoff` convention). |
| `utils/table_model.py` | Presentation-agnostic `TableModel`/`Row` data model (kept separate from Excel rendering so it could target CSV/LaTeX/Word later without touching `build_tables.py`). |
| `utils/excel_export.py` | Renders `TableModel`s into a styled `.xlsx` workbook (one sheet per table). |

## Running

```bash
python run.py
```

## Groups

Derived once in `utils/data_loading.py:load_groups()`:

| Group | Source | Filter |
|---|---|---|
| `training` | cohort 1 | `use == 1` and `test_temporal == 0` |
| `test1` | cohort 1 | `use == 1` and `test_temporal == 1` |
| `test2` | cohort 2 (external, liver-transplant) | `use == 1` (always "test") |


## Statistics

Configured in `config.py` (`CONTINUOUS_TEST`, `CATEGORICAL_TEST`, both default
`"auto"`):

- **Continuous** (age, BMI, hand-grip, chair-rise): Shapiro-Wilk per group +
  Levene's test → one-way ANOVA if all groups are approximately normal with
  homogeneous variance, otherwise Kruskal-Wallis.
- **Categorical** (sex, sarcopenia flags): Pearson chi-square; if any expected
  cell count < 5, falls back to Fisher's exact (2×2) or a seeded
  Monte-Carlo permutation chi-square (r×c, 10000 resamples, `RANDOM_SEED=2025`
  for reproducibility).

Every test function is defensive — it returns `p_value = nan` with an
explanatory test name on degenerate input rather than raising, so one missing
or degenerate variable never aborts the whole run.

## Output layout

One `.xlsx` workbook (`cohort_description_tables.xlsx` by default) with:

- **Table 1 — Patient characteristics**: age, sex, BMI, hand-grip strength,
  chair-rise time, and sarcopenia status (by hand-grip test, chair-rise test,
  and the composite definition), compared across the three groups above.
- **Table 2 — Clinical characteristics of cohort 1**: ECOG status, tumor
  type, metastasis flags, chemotherapy cycles (training vs. testing set 1
  only).
- **Table 3 — Clinical characteristics of cohort 2**: MELD score, Child-Pugh
  grade, hepatocellular carcinoma, transplant-listing status (testing set 2
  only).
- **Statistical methods** (optional, on by default): one row per compared
  variable with the test used, exact p-value, and n.

## Notes

- Ground-truth table paths in `config.py` point to local data storage and
  must be adjusted to your own paths before running.
