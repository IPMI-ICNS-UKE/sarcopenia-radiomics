import re
from pathlib import Path
from typing import Dict, List, Sequence
import pandas as pd


def natural_patient_sort(patient_ids: Sequence[str]) -> List[str]:
    def key(s: str):
        m = re.search(r"(\d+)", str(s))
        return (int(m.group(1)) if m else 0, str(s))

    return sorted([str(x) for x in patient_ids], key=key)


def load_ground_truth(
    path: Path,
    filters: Sequence[tuple[str, int]],
    patient_id_col_gt: str,
    split_source_col: str,
    patient_id_col: str = "patient_id",
    split_col: str = "split",
    train_label: str = "train",
    test_label: str = "test",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth table: {path}")

    df = pd.read_excel(path)

    for col, val in filters:
        if col not in df.columns:
            raise KeyError(f"Required filter column missing in ground truth: {col}")
        df = df[df[col] == val].copy()

    required = [patient_id_col_gt, split_source_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in ground truth: {missing}")

    out = df[[patient_id_col_gt, split_source_col]].copy()
    out = out.rename(columns={patient_id_col_gt: patient_id_col})
    out[patient_id_col] = out[patient_id_col].astype(str)
    out[split_col] = out[split_source_col].map({0: train_label, 1: test_label})

    bad = out[out[split_col].isna()]
    if not bad.empty:
        raise ValueError(
            f"Unexpected values in split source column '{split_source_col}': "
            f"{sorted(bad[split_source_col].dropna().unique().tolist())}"
        )

    return out[[patient_id_col, split_col]].drop_duplicates().reset_index(drop=True)


def load_features_table(
    path: Path,
    required_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
    keep_status_value: str | None = "ok",
    patient_id_col: str = "patient_id",
    table_name: str = "features table",
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing features CSV: {path}")

    df = pd.read_csv(path)
    required = set(required_cols)
    if feature_cols is not None:
        required |= set(feature_cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {table_name}: {missing}")

    if keep_status_value is not None and "status" in df.columns:
        df = df[df["status"] == keep_status_value].copy()

    if patient_id_col in df.columns:
        patient_ids = natural_patient_sort(df[patient_id_col].astype(str).unique().tolist())
        df[patient_id_col] = pd.Categorical(
            df[patient_id_col].astype(str),
            categories=patient_ids,
            ordered=True,
        )
        sort_cols = [c for c in [patient_id_col, "mask"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df[patient_id_col] = df[patient_id_col].astype(str)

    return df.reset_index(drop=True)


def attach_splits(
    df_features: pd.DataFrame,
    df_splits: pd.DataFrame,
    patient_id_col: str = "patient_id",
    split_col: str = "split",
) -> pd.DataFrame:
    df = df_features.merge(
        df_splits,
        on=patient_id_col,
        how="left",
        validate="many_to_one",
    )

    if df[split_col].isna().any():
        missing = (
            df.loc[df[split_col].isna(), patient_id_col]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"Could not assign train/test split to patients: {missing[:10]}")

    return df.reset_index(drop=True)


def assign_constant_split(df_features: pd.DataFrame, split_col: str, split_name: str) -> pd.DataFrame:
    out = df_features.copy()
    out[split_col] = split_name
    return out.reset_index(drop=True)


def detect_feature_columns(
    df: pd.DataFrame,
    metadata_cols: Sequence[str],
    split_col: str = "split",
    prefix: str | None = None,
) -> List[str]:
    exclude = set(metadata_cols) | {split_col}
    feature_cols = [c for c in df.columns if c not in exclude]
    if prefix is not None:
        feature_cols = [c for c in feature_cols if str(c).startswith(prefix) and "_" in str(c)]
    if not feature_cols:
        raise ValueError("No feature columns were detected.")
    return sorted(feature_cols)


def get_map_type(feature_name: str) -> str:
    if "_" not in feature_name:
        raise ValueError(f"Cannot parse map type from feature name: {feature_name}")
    return str(feature_name).rsplit("_", 1)[-1]


def normalize_map_type(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isalnum()).lower()


def map_type_to_slug(map_type: str) -> str:
    return normalize_map_type(map_type)


def build_feature_groups_by_map_type(
    feature_cols: Sequence[str],
    target_map_types: Sequence[str],
) -> Dict[str, List[str]]:
    allowed_norm = {normalize_map_type(mt): mt for mt in target_map_types}
    grouped: Dict[str, List[str]] = {mt: [] for mt in target_map_types}

    for feat in feature_cols:
        map_type_norm = normalize_map_type(get_map_type(feat))
        if map_type_norm in allowed_norm:
            grouped[allowed_norm[map_type_norm]].append(feat)

    return {k: sorted(v) for k, v in grouped.items() if v}


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
