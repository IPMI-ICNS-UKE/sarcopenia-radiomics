from typing import List

import numpy as np
import pandas as pd
from pingouin import intraclass_corr

import config


def compute_icc_with_ci(raw_values_df: pd.DataFrame, biomarker: str) -> dict:
    """ICC(2,1) and 95% CI for one biomarker, across the three raters."""
    long_df = raw_values_df[["patient_id", "rater", biomarker]].dropna()

    n_subjects = long_df["patient_id"].nunique()
    n_raters = long_df["rater"].nunique()
    if n_subjects < 2 or n_raters < 2:
        return {
            "feature": biomarker,
            "icc": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "f_stat": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "p_value": np.nan,
            "n_subjects": int(n_subjects),
            "n_raters": int(n_raters),
        }

    icc_table = intraclass_corr(
        data=long_df, targets="patient_id", raters="rater", ratings=biomarker
    )
    row = icc_table[icc_table["Type"] == "ICC2"].iloc[0]
    ci_low, ci_high = row["CI95%"]

    return {
        "feature": biomarker,
        "icc": float(row["ICC"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "f_stat": float(row["F"]),
        "df1": float(row["df1"]),
        "df2": float(row["df2"]),
        "p_value": float(row["pval"]),
        "n_subjects": int(n_subjects),
        "n_raters": int(n_raters),
    }


def build_icc_summary(raw_values_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = [
        compute_icc_with_ci(raw_values_df, biomarker) for biomarker in config.BIOMARKERS
    ]
    summary = pd.DataFrame(rows)
    summary.insert(1, "feature_label", summary["feature"].map(config.BIOMARKER_LABELS))
    return summary
