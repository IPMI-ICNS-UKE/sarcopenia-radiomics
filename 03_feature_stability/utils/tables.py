from typing import Sequence
import pandas as pd


def build_original_mask_feature_table(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    keep_id_cols: Sequence[str],
    split_col: str | None = None,
) -> pd.DataFrame:
    """Return one row per original-mask measurement and selected feature columns."""
    sub = df[df["mask"] == "a_original"].copy()

    drop_cols = [c for c in ("status", "mask", "slice_name") if c in sub.columns]
    if split_col is not None and split_col in sub.columns:
        drop_cols.append(split_col)
    sub = sub.drop(columns=drop_cols)

    keep_cols = [c for c in keep_id_cols if c in sub.columns]
    ordered_cols = keep_cols + [c for c in feature_cols if c in sub.columns]
    ordered_cols = list(dict.fromkeys(ordered_cols))

    return sub[ordered_cols].reset_index(drop=True)
