import sys
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from config import RCFG, RunConfig, TASKS, TaskConfig, TaskName


GT_REQUIRED_COLUMNS = [
    "Pat ID", "age", "sex", "bmi",
    "hand_grip", "hand_grip_cont", "chair_rise",
    "sarcopenia_composite",
    "ct_muscle_mass_2d", "ct_muscle_density_2d",
    "test_temporal", "use",
]

COHORT_1 = "cohort1"
COHORT_2 = "cohort2"


# Validation
def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


# Readers
def read_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    _ensure_columns(df, GT_REQUIRED_COLUMNS, f"GT table {path}")
    return df.copy()


def read_signature(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.copy()


# Transformations
def apply_filters(df: pd.DataFrame, run_cfg: RunConfig) -> pd.DataFrame:
    out = df.copy()
    for col, val in run_cfg.filters.required_filters:
        if col not in out.columns:
            raise KeyError(f"Filter column '{col}' not found in table.")
        out = out[out[col] == val]
    return out.reset_index(drop=True)


def standardize_ground_truth(
    df: pd.DataFrame,
    cohort_label: str,
    *,
    force_split: Optional[str] = None,
) -> pd.DataFrame:
    out = df.rename(columns={
        "Pat ID": "patient_id",
        "ct_muscle_mass_2d": "gt_smi",
        "ct_muscle_density_2d": "gt_mra",
    }).copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["cohort"] = cohort_label
    if force_split is None:
        out["split"] = out["test_temporal"].map({0: "train", 1: "test"})
        bad = out["split"].isna()
        if bad.any():
            raise ValueError(
                f"Cannot derive split from 'test_temporal' for: "
                f"{out.loc[bad, 'patient_id'].tolist()}"
            )
    else:
        out["split"] = force_split
    return out


def standardize_signature(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to standard names, make sure that smi_2d and smi_3d are renamed to auto_smi,
    # and mra_2d and mra_3d are renamed to auto_mra
    out = df.rename(columns={
        "smi_2d": "auto_smi",
        "mra_2d": "auto_mra",
        "smi_3d": "auto_smi",
        "mra_3d": "auto_mra",
    }).copy()
    out["patient_id"] = out["patient_id"].astype(str)
    dups = out["patient_id"].duplicated(keep=False)
    if dups.any():
        raise ValueError(
            f"Duplicate patient_id in signature: "
            f"{sorted(out.loc[dups, 'patient_id'].unique())}"
        )
    return out[["patient_id", "auto_smi", "auto_mra"]].copy()


def add_task_target(df: pd.DataFrame, task_cfg: TaskConfig) -> pd.DataFrame:
    out = df.copy()
    out["task"] = task_cfg.name
    out["target"] = out[task_cfg.target_column_gt]
    return out[~out["target"].isna()].reset_index(drop=True)


def merge_gt_signature(df_gt: pd.DataFrame, df_sig: pd.DataFrame) -> pd.DataFrame:
    out = df_gt.merge(df_sig, on="patient_id", how="inner")
    if out.empty:
        raise ValueError("Merged dataset is empty after inner join.")
    return out


def finalize_table(df: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "patient_id", "cohort", "split", "task", "target",
        "age", "sex", "bmi",
        "gt_smi", "gt_mra",
        "auto_smi", "auto_mra",
        "hand_grip", "hand_grip_cont", "chair_rise",
        "sarcopenia_composite", "test_temporal",
    ]
    existing = [c for c in desired if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    out = df[existing + remaining].copy()
    # Sort: cohort1 train → cohort1 test → cohort2 test
    split_cat = pd.Categorical(out["split"], categories=["train", "test"], ordered=True)
    out = (
        out.assign(split=split_cat)
        .sort_values(["cohort", "split", "patient_id"])
        .reset_index(drop=True)
    )
    out["split"] = out["split"].astype(str)
    return out


# Internal builder per cohort
def _build_cohort(
    *,
    task_cfg: TaskConfig,
    gt_path: Path,
    sig_path: Path,
    cohort_label: str,
    run_cfg: RunConfig,
    force_split: Optional[str] = None,
) -> pd.DataFrame:
    df_gt = read_ground_truth(gt_path)
    df_gt = apply_filters(df_gt, run_cfg)
    df_gt = standardize_ground_truth(df_gt, cohort_label, force_split=force_split)
    df_gt = add_task_target(df_gt, task_cfg)

    df_sig = read_signature(sig_path)
    df_sig = standardize_signature(df_sig)

    df = merge_gt_signature(df_gt, df_sig)
    print(
        f"  [{cohort_label}] task={task_cfg.name}  rows={len(df)}  "
        f"train={int((df['split']=='train').sum())}  "
        f"test={int((df['split']=='test').sum())}"
    )
    return df


# Public API — unified dataset (cohort1 + cohort2 concatenated)
def build_dataset(
    task: TaskName,
    run_cfg: RunConfig = RCFG,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Build the unified modeling dataset for a task.

    Returns a single DataFrame containing both cohort1 (train+test split) and
    cohort2 (always split='test'), with a 'cohort' column to distinguish them.
    This matches the pattern used by 06_02 and 06_03 pipelines.
    """
    task_cfg = TASKS[task]
    run_cfg.paths.make_all()

    c1 = _build_cohort(
        task_cfg=task_cfg,
        gt_path=run_cfg.paths.gt_table,
        sig_path=run_cfg.paths.baseline_signature_dir / task_cfg.signature_filename,
        cohort_label=COHORT_1,
        run_cfg=run_cfg,
        force_split=None,
    )
    c2 = _build_cohort(
        task_cfg=task_cfg,
        gt_path=run_cfg.paths.gt_table_cohort_2,
        sig_path=run_cfg.paths.baseline_signature_dir / task_cfg.cohort_2_signature_filename,
        cohort_label=COHORT_2,
        run_cfg=run_cfg,
        force_split="test",
    )

    df = finalize_table(pd.concat([c1, c2], axis=0, ignore_index=True))

    if save_csv:
        path = run_cfg.paths.tables_dir / f"dataset_{task}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved dataset: {path}")
    return df


def build_all_datasets(
    run_cfg: RunConfig = RCFG,
    save_csv: bool = True,
) -> Dict[str, pd.DataFrame]:
    return {task: build_dataset(task=task, run_cfg=run_cfg, save_csv=save_csv)
            for task in TASKS}
