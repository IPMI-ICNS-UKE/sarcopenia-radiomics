import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from config import RCFG, RunConfig, TASKS, TaskConfig, TaskName

GT_REQUIRED_COLUMNS = [
    "Pat ID",
    "age",
    "sex",
    "bmi",
    "hand_grip",
    "hand_grip_cont",
    "chair_rise",
    "sarcopenia_composite",
    "test_temporal",
    "use",
]

COHORT_1 = "cohort1"
COHORT_2 = "cohort2"


def _ensure_columns(df: pd.DataFrame, required, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _read_gt(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    _ensure_columns(df, GT_REQUIRED_COLUMNS, f"GT {path}")
    return df.copy()


def _apply_filters(df: pd.DataFrame, run_cfg: RunConfig) -> pd.DataFrame:
    out = df.copy()
    for col, val in run_cfg.filters.required_filters:
        out = out[out[col] == val]
    return out.reset_index(drop=True)


def _standardize_gt(
    df: pd.DataFrame, cohort_label: str, *, force_split: Optional[str] = None
) -> pd.DataFrame:
    out = df.rename(columns={"Pat ID": "patient_id"}).copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["cohort"] = cohort_label
    if force_split is None:
        out["split"] = out["test_temporal"].map({0: "train", 1: "test"})
        bad = out["split"].isna()
        if bad.any():
            raise ValueError(f"Cannot derive split for: {out.loc[bad,'patient_id'].tolist()}")
    else:
        out["split"] = force_split
    return out


def _read_map_features(run_cfg: RunConfig, map_name: str, cohort_2: bool) -> pd.DataFrame:
    """Read deep radiomics CSV for one map."""
    fname = run_cfg.blocks.sig_filename(map_name, cohort_2=cohort_2)
    path = run_cfg.paths.feature_root / map_name / fname
    if not path.exists():
        raise FileNotFoundError(f"Deep radiomics feature file not found: {path}")
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise KeyError(f"Feature file {path} missing 'patient_id' column.")
    df["patient_id"] = df["patient_id"].astype(str)
    # Keep only patient_id + feature columns (d<index>_<map>)
    feat_cols = run_cfg.blocks.feature_cols_for_map(df.columns, map_name)
    if not feat_cols:
        raise ValueError(f"No feature columns found for map '{map_name}' in {path}.")
    return df[["patient_id"] + list(feat_cols)].copy()


def _join_all_maps(run_cfg: RunConfig, cohort_2: bool) -> pd.DataFrame:
    """Inner-join feature tables for all maps on patient_id."""
    dfs: List[pd.DataFrame] = []
    for map_name in run_cfg.blocks.map_names:
        dfs.append(_read_map_features(run_cfg, map_name, cohort_2))
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="patient_id", how="inner")
    if merged.empty:
        raise ValueError("Feature table is empty after joining all maps.")
    return merged


def _build_cohort_dataset(
    *,
    task_cfg: TaskConfig,
    cohort_label: str,
    run_cfg: RunConfig,
    force_split: Optional[str] = None,
) -> pd.DataFrame:
    cohort_2 = cohort_label == COHORT_2
    gt_path = run_cfg.paths.gt_table if not cohort_2 else run_cfg.paths.gt_table_cohort_2
    df_gt = _read_gt(gt_path)
    df_gt = _apply_filters(df_gt, run_cfg)
    df_gt = _standardize_gt(df_gt, cohort_label, force_split=force_split)
    df_gt["task"] = task_cfg.name
    df_gt["target"] = df_gt[task_cfg.target_column_gt]
    df_gt = df_gt[~df_gt["target"].isna()].reset_index(drop=True)

    df_feat = _join_all_maps(run_cfg, cohort_2)

    df = df_gt.merge(df_feat, on="patient_id", how="inner")
    if df.empty:
        raise ValueError(f"Merged dataset is empty for {cohort_label}.")
    print(
        f"  [{cohort_label}] task={task_cfg.name}  rows={len(df)}  "
        f"train={int((df['split']=='train').sum())}  test={int((df['split']=='test').sum())}"
    )
    return df


def build_dataset(task: TaskName, run_cfg: RunConfig = RCFG, save_csv: bool = True) -> pd.DataFrame:
    task_cfg = TASKS[task]
    run_cfg.paths.make_all()
    c1 = _build_cohort_dataset(task_cfg=task_cfg, cohort_label=COHORT_1, run_cfg=run_cfg)
    c2 = _build_cohort_dataset(
        task_cfg=task_cfg, cohort_label=COHORT_2, run_cfg=run_cfg, force_split="test"
    )
    df = pd.concat([c1, c2], axis=0, ignore_index=True)
    if save_csv:
        path = run_cfg.paths.tables_dir / f"dataset_{task}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
    return df


def build_all_datasets(run_cfg: RunConfig = RCFG, save_csv: bool = True) -> Dict[str, pd.DataFrame]:
    return {task: build_dataset(task=task, run_cfg=run_cfg, save_csv=save_csv) for task in TASKS}
