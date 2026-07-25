import os
import warnings
from typing import Dict, Optional, Tuple

import pandas as pd

from config import MODEL_SOURCES, OUTPUT_ROOT, SPLITS


# Internal helpers


def _metrics_path(subfolder: str, filename: str) -> str:
    return os.path.join(OUTPUT_ROOT, subfolder, "metrics", filename)


def _predictions_path(subfolder: str, filename: str) -> str:
    return os.path.join(OUTPUT_ROOT, subfolder, "predictions", filename)


def _parse_ci95(ci_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a CI-95 string like '0.XXX (0.YYY-0.ZZZ)' into (lower, upper).
    Returns (None, None) on failure.
    """
    try:
        inner = ci_str.split("(")[1].rstrip(")")
        lo, hi = inner.split("-")
        return float(lo.strip()), float(hi.strip())
    except Exception:
        return None, None


# Public API


def load_metrics_for_split(split_key: str) -> pd.DataFrame:
    """
    Load and concatenate AUC metrics for all models for a given split.

    Returns a DataFrame with columns:
        model_name, AUC, AUC_lo, AUC_hi, AUC_CI95
    """
    metrics_filename, _, _ = SPLITS[split_key]
    rows = []

    for subfolder, model_names in MODEL_SOURCES:
        path = _metrics_path(subfolder, metrics_filename)
        if not os.path.exists(path):
            warnings.warn(f"Metrics file not found, skipping: {path}")
            continue

        df = pd.read_excel(path)

        for model_name in model_names:
            row = df[df["model_name"] == model_name]
            if row.empty:
                warnings.warn(f"Model '{model_name}' not found in {path}. Skipping.")
                continue

            row = row.iloc[0]
            auc = float(row["AUC"])
            ci_str = str(row.get("AUC_CI95", ""))
            lo, hi = _parse_ci95(ci_str)

            rows.append(
                {
                    "model_name": model_name,
                    "AUC": auc,
                    "AUC_lo": lo,
                    "AUC_hi": hi,
                    "AUC_CI95": ci_str,
                }
            )

    return pd.DataFrame(rows)


def load_predictions_for_split(split_key: str) -> pd.DataFrame:
    """
    Load and concatenate raw prediction scores for all models for a given split.

    Returns a DataFrame with columns:
        model_name, patient_id, target, pred_score
    """
    _, predictions_filename, _ = SPLITS[split_key]
    frames = []

    for subfolder, model_names in MODEL_SOURCES:
        path = _predictions_path(subfolder, predictions_filename)
        if not os.path.exists(path):
            warnings.warn(f"Predictions file not found, skipping: {path}")
            continue

        df = pd.read_csv(path)

        for model_name in model_names:
            sub = df[df["model_name"] == model_name].copy()
            if sub.empty:
                warnings.warn(f"Model '{model_name}' not found in {path}. Skipping.")
                continue

            required_cols = {"patient_id", "target", "pred_score", "model_name"}
            missing = required_cols - set(sub.columns)
            if missing:
                warnings.warn(f"Missing columns {missing} in {path}. Skipping model.")
                continue

            frames.append(sub[list(required_cols)])

    if not frames:
        return pd.DataFrame(columns=["model_name", "patient_id", "target", "pred_score"])

    return pd.concat(frames, ignore_index=True)


def load_all_splits() -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load metrics and predictions for all splits.

    Returns:
        dict mapping split_key → (metrics_df, predictions_df)
    """
    result = {}
    for split_key in SPLITS:
        metrics_df = load_metrics_for_split(split_key)
        predictions_df = load_predictions_for_split(split_key)
        result[split_key] = (metrics_df, predictions_df)
    return result
