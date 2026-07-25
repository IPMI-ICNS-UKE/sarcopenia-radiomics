from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from config import ICFG, SCORE_MAP_NAMES, InterpretabilityConfig

# Constants
COHORT_1 = "cohort1"
COHORT_2 = "cohort2"

GT_REQUIRED_COLUMNS = [
    "Pat ID",
    "age",
    "sex",
    "bmi",
    "sarcopenia_composite",
    "test_temporal",
    "use",
]


# Internal helpers (mirror 06_02_02/dataset.py, scoped to interpretability)
def _read_gt(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in GT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"GT table {path} missing columns: {missing}")
    return df.copy()


def _apply_use_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["use"] == 1].reset_index(drop=True)


def _standardize_gt(
    df: pd.DataFrame, cohort_label: str, force_split: Optional[str] = None
) -> pd.DataFrame:
    out = df.rename(columns={"Pat ID": "patient_id"}).copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["cohort"] = cohort_label
    if force_split is not None:
        out["split"] = force_split
    else:
        out["split"] = out["test_temporal"].map({0: "train", 1: "test"})
        bad = out["split"].isna()
        if bad.any():
            raise ValueError(f"Cannot derive split for: {out.loc[bad, 'patient_id'].tolist()}")
    return out


def _read_map_features(
    cfg: InterpretabilityConfig, map_name: str, cohort_label: str
) -> pd.DataFrame:
    suffix = "" if cohort_label == COHORT_1 else "_cohort_2"
    fname = f"signature_spectral_radiomics_signature_{map_name}{suffix}.csv"
    path = cfg.paths.feature_root / map_name / fname
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise KeyError(f"Feature file {path} missing 'patient_id' column.")
    df["patient_id"] = df["patient_id"].astype(str)
    return df.copy()


def _read_all_map_features(cfg: InterpretabilityConfig, cohort_label: str) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for map_name in SCORE_MAP_NAMES:
        dfs.append(_read_map_features(cfg, map_name, cohort_label))
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="patient_id", how="inner")
    if merged.empty:
        raise ValueError(f"Feature table is empty after joining maps for {cohort_label}.")
    return merged


def feature_columns_for_map(df: pd.DataFrame, map_name: str) -> List[str]:
    """Return all feature columns that belong to a given map."""
    return sorted(c for c in df.columns if c != "patient_id" and c.endswith(f"_{map_name}"))


# API
def load_splits(
    cfg: InterpretabilityConfig = ICFG,
) -> Dict[str, pd.DataFrame]:
    cfg.paths.make_all()

    # ---- Cohort 1 ----
    gt1 = _read_gt(cfg.paths.gt_table_cohort1)
    gt1 = _apply_use_filter(gt1)
    gt1 = _standardize_gt(gt1, COHORT_1)
    gt1["target"] = gt1["sarcopenia_composite"]
    gt1 = gt1[~gt1["target"].isna()].reset_index(drop=True)

    feat1 = _read_all_map_features(cfg, COHORT_1)
    df1 = gt1.merge(feat1, on="patient_id", how="inner")

    # ---- Cohort 2 ----
    gt2 = _read_gt(cfg.paths.gt_table_cohort2)
    gt2 = _apply_use_filter(gt2)
    gt2 = _standardize_gt(gt2, COHORT_2, force_split="test")
    gt2["target"] = gt2["sarcopenia_composite"]
    gt2 = gt2[~gt2["target"].isna()].reset_index(drop=True)

    feat2 = _read_all_map_features(cfg, COHORT_2)
    df2 = gt2.merge(feat2, on="patient_id", how="inner")

    train = df1[df1["split"] == "train"].reset_index(drop=True)
    test1 = df1[df1["split"] == "test"].reset_index(drop=True)
    test2 = df2.reset_index(drop=True)

    for label, df in [("train", train), ("test1", test1), ("test2", test2)]:
        print(
            f"  [{label}] cohort={df['cohort'].iloc[0] if len(df) else '?'}  "
            f"n={len(df)}  sarcopenia={int(df['target'].sum()) if len(df) else 0}"
        )

    return {"train": train, "test1": test1, "test2": test2}
