from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import pandas as pd


def read_csv_checked(path: Path, required_columns: Sequence[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    return df


def read_excel_checked(path: Path, required_columns: Sequence[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {path}")
    df = pd.read_excel(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    return df


def load_ground_truth_with_split(
    path: Path,
    patient_id_col: str,
    test_flag_col: Optional[str] = None,
    all_test: bool = False,
) -> pd.DataFrame:
    required = [patient_id_col] + ([] if all_test else [test_flag_col])
    df = read_excel_checked(path, required_columns=required).copy()
    df[patient_id_col] = df[patient_id_col].astype(str)
    if all_test:
        df["split"] = "test"
    else:
        df["split"] = df[test_flag_col].map(lambda x: "test" if int(x) == 1 else "train")
    return df[[patient_id_col, "split"]].copy()


def attach_split_labels(
    df_features: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    patient_id_col_data: str,
    patient_id_col_gt: str,
) -> pd.DataFrame:
    df = df_features.copy()
    df[patient_id_col_data] = df[patient_id_col_data].astype(str)
    merged = df.merge(
        df_ground_truth[[patient_id_col_gt, "split"]],
        left_on=patient_id_col_data,
        right_on=patient_id_col_gt,
        how="left",
    )
    missing_split = int(merged["split"].isna().sum())
    if missing_split:
        examples = (
            merged.loc[merged["split"].isna(), patient_id_col_data].astype(str).head(10).tolist()
        )
        raise ValueError(
            f"{missing_split} patients from feature table were not found in ground truth. "
            f"Examples: {examples}"
        )
    return merged.drop(columns=[patient_id_col_gt])


def ensure_cohort_column(df: pd.DataFrame, cohort_col: str, cohort_label: str) -> pd.DataFrame:
    out = df.copy()
    if cohort_col not in out.columns:
        out[cohort_col] = cohort_label
    else:
        out[cohort_col] = out[cohort_col].fillna(cohort_label).astype(str)
        out.loc[out[cohort_col].str.strip() == "", cohort_col] = cohort_label
    return out


def identify_feature_columns(df: pd.DataFrame, id_columns: Sequence[str]) -> List[str]:
    id_set = set(id_columns)
    return [c for c in df.columns if c not in id_set]


def ensure_output_dirs(out_dir: Path) -> Tuple[Path, Path, Path]:
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, tables_dir, plots_dir


def add_postfix_to_filename(filename: str, postfix: str) -> str:
    p = Path(filename)
    return f"{p.stem}{postfix}{p.suffix}"


def align_to_reference_features(
    df_reference: pd.DataFrame,
    df_target: pd.DataFrame,
    id_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    ref_features = identify_feature_columns(df_reference, id_columns)
    target_features = identify_feature_columns(df_target, id_columns)

    missing = sorted(set(ref_features) - set(target_features))
    extra = sorted(set(target_features) - set(ref_features))
    if missing:
        raise ValueError(
            f"{label}: target table is missing {len(missing)} feature columns. "
            f"Examples: {missing[:10]}"
        )
    if extra:
        df_target = df_target.drop(columns=extra)

    ordered_cols = [c for c in list(id_columns) + ref_features if c in df_target.columns]
    return df_target.loc[:, ordered_cols].copy()
