from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


# Generic helpers
def round_df(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round all float columns in *df* to *decimals* decimal places."""
    out = df.copy()
    for col in out.select_dtypes(include=["floating"]).columns:
        out[col] = out[col].round(decimals)
    return out


def save_df(df: pd.DataFrame, path: Path, decimals: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_df(df, decimals).to_csv(path, index=False)
    print(f"Saved: {path}")


def save_excel(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1", decimals: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_df(df, decimals).to_excel(path, sheet_name=sheet_name, index=False)
    print(f"Saved: {path}")


def save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved: {path}")


# Formatting helpers shared across report builders
def format_ci(point: float, low: float, high: float, decimals: int) -> str:
    """Return 'point (low-high)' string; gracefully handles NaN."""
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.{decimals}f}"
    return f"{point:.{decimals}f} ({low:.{decimals}f}-{high:.{decimals}f})"


def build_ci_column(
    df: pd.DataFrame,
    metric: str,
    decimals: int,
) -> List[str]:
    """Build a list of formatted CI strings from a merged metrics+bootstrap table."""
    low_col = f"{metric}_ci_low"
    high_col = f"{metric}_ci_high"
    return [
        format_ci(
            point,
            row.get(low_col, float("nan")),
            row.get(high_col, float("nan")),
            decimals,
        )
        for point, row in zip(df.get(metric, [float("nan")] * len(df)), df.to_dict("records"))
    ]


# Structured metrics table builders (publication-ready)
def build_classification_metrics_table(
    df_metrics: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Build a classification metrics table.

    Expected columns in *df_metrics* (point estimates + bootstrap CI columns):
        model_name, n_features, selected_threshold,
        auc, auc_ci_low, auc_ci_high,
        balanced_accuracy, balanced_accuracy_ci_low, balanced_accuracy_ci_high,
        sensitivity, sensitivity_ci_low, sensitivity_ci_high,
        specificity, specificity_ci_low, specificity_ci_high,
        ppv, ppv_ci_low, ppv_ci_high,
        npv, npv_ci_low, npv_ci_high,
        brier, brier_ci_low, brier_ci_high,
        tp, tn, fp, fn, n
    """
    if df_metrics.empty:
        return pd.DataFrame()

    def _get(row: dict, col: str):
        return row.get(col, np.nan)

    rows = []
    for _, r in df_metrics.iterrows():
        row_d = r.to_dict()
        rows.append({
            "model_name": r["model_name"],
            "n_features": _get(row_d, "n_features"),
            "selected_threshold": round(float(_get(row_d, "selected_threshold")), decimals)
                if not pd.isna(_get(row_d, "selected_threshold")) else float("nan"),
            "AUC": round(float(r["auc"]), decimals) if not pd.isna(r["auc"]) else float("nan"),
            "AUC_CI95": format_ci(_get(row_d, "auc"), _get(row_d, "auc_ci_low"), _get(row_d, "auc_ci_high"), decimals),
            "Balanced_Accuracy": round(float(_get(row_d, "balanced_accuracy")), decimals)
                if not pd.isna(_get(row_d, "balanced_accuracy")) else float("nan"),
            "Balanced_Accuracy_CI95": format_ci(_get(row_d, "balanced_accuracy"), _get(row_d, "balanced_accuracy_ci_low"), _get(row_d, "balanced_accuracy_ci_high"), decimals),
            "Sensitivity": round(float(_get(row_d, "sensitivity")), decimals)
                if not pd.isna(_get(row_d, "sensitivity")) else float("nan"),
            "Sensitivity_CI95": format_ci(_get(row_d, "sensitivity"), _get(row_d, "sensitivity_ci_low"), _get(row_d, "sensitivity_ci_high"), decimals),
            "Specificity": round(float(_get(row_d, "specificity")), decimals)
                if not pd.isna(_get(row_d, "specificity")) else float("nan"),
            "Specificity_CI95": format_ci(_get(row_d, "specificity"), _get(row_d, "specificity_ci_low"), _get(row_d, "specificity_ci_high"), decimals),
            "PPV": round(float(_get(row_d, "ppv")), decimals)
                if not pd.isna(_get(row_d, "ppv")) else float("nan"),
            "PPV_CI95": format_ci(_get(row_d, "ppv"), _get(row_d, "ppv_ci_low"), _get(row_d, "ppv_ci_high"), decimals),
            "NPV": round(float(_get(row_d, "npv")), decimals)
                if not pd.isna(_get(row_d, "npv")) else float("nan"),
            "NPV_CI95": format_ci(_get(row_d, "npv"), _get(row_d, "npv_ci_low"), _get(row_d, "npv_ci_high"), decimals),
            "Brier": round(float(_get(row_d, "brier")), decimals)
                if not pd.isna(_get(row_d, "brier")) else float("nan"),
            "Brier_CI95": format_ci(_get(row_d, "brier"), _get(row_d, "brier_ci_low"), _get(row_d, "brier_ci_high"), decimals),
            "TP": int(_get(row_d, "tp")) if not pd.isna(_get(row_d, "tp")) else float("nan"),
            "TN": int(_get(row_d, "tn")) if not pd.isna(_get(row_d, "tn")) else float("nan"),
            "FP": int(_get(row_d, "fp")) if not pd.isna(_get(row_d, "fp")) else float("nan"),
            "FN": int(_get(row_d, "fn")) if not pd.isna(_get(row_d, "fn")) else float("nan"),
            "N": int(_get(row_d, "n")) if not pd.isna(_get(row_d, "n")) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def build_regression_metrics_table(
    df_metrics: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Build a regression metrics table.

    Expected columns: model_name, n_features,
        rmse, rmse_ci_low, rmse_ci_high,
        mae, mae_ci_low, mae_ci_high,
        r2, r2_ci_low, r2_ci_high,
        spearman_r, spearman_p, n
    """
    if df_metrics.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_metrics.iterrows():
        row_d = r.to_dict()

        def _g(col):
            return row_d.get(col, np.nan)

        rows.append({
            "model_name": r["model_name"],
            "n_features": _g("n_features"),
            "RMSE": round(float(_g("rmse")), decimals) if not pd.isna(_g("rmse")) else float("nan"),
            "RMSE_CI95": format_ci(_g("rmse"), _g("rmse_ci_low"), _g("rmse_ci_high"), decimals),
            "MAE": round(float(_g("mae")), decimals) if not pd.isna(_g("mae")) else float("nan"),
            "MAE_CI95": format_ci(_g("mae"), _g("mae_ci_low"), _g("mae_ci_high"), decimals),
            "R2": round(float(_g("r2")), decimals) if not pd.isna(_g("r2")) else float("nan"),
            "R2_CI95": format_ci(_g("r2"), _g("r2_ci_low"), _g("r2_ci_high"), decimals),
            "Spearman_R": round(float(_g("spearman_r")), decimals) if not pd.isna(_g("spearman_r")) else float("nan"),
            "Spearman_P": round(float(_g("spearman_p")), decimals) if not pd.isna(_g("spearman_p")) else float("nan"),
            "N": int(_g("n")) if not pd.isna(_g("n")) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
