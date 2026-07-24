from typing import Iterable, List
import pandas as pd

from .config import BaseStage05Config


def make_signature_dataset(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    cfg: BaseStage05Config,
) -> pd.DataFrame:
    feature_cols = list(feature_cols)
    keep_cols = [cfg.patient_id_col] + feature_cols

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required signature columns: {missing}")

    return df[keep_cols].copy()


def summarize_signature(
    signature_df: pd.DataFrame,
    selected_features: List[str],
    cfg: BaseStage05Config,
    cohort_key: str,
    extra: dict | None = None,
) -> dict:
    payload = {
        "cohort_key": cohort_key,
        "target_col": cfg.target_col,
        "n_rows": int(len(signature_df)),
        "n_unique_patients": int(signature_df[cfg.patient_id_col].nunique()),
        "n_features_exported": int(len(selected_features)),
        "exported_feature_columns": list(selected_features),
    }
    if extra:
        payload.update(extra)
    return payload
