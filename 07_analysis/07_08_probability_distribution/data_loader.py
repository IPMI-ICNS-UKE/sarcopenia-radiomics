from pathlib import Path

import pandas as pd


def load_ground_truth_sex(gt_path: Path) -> pd.Series:
    """Return a Series: index = 'Pat ID' string, value = sex (int 1/0)."""
    df = pd.read_excel(gt_path, dtype={"Pat ID": str})
    df = df.rename(columns={"Pat ID": "patient_id"})
    df["patient_id"] = df["patient_id"].str.strip()
    return df.set_index("patient_id")["sex"]


def load_predictions(
    csv_path: Path,
    model_name: str,
    gt_sex: pd.Series,
) -> pd.DataFrame | None:
    """
    Load a classification prediction CSV, filter to *model_name*, and attach sex.

    Returns None when the file does not exist (e.g. cohort-2 CSV not yet
    available).  Returns an empty DataFrame when the model is absent.
    """
    if not csv_path.exists():
        print(f"[data_loader] File not found, skipping: {csv_path}")
        return None

    df = pd.read_csv(csv_path, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].str.strip()

    # Filter to target model
    df = df[df["model_name"] == model_name].copy()
    if df.empty:
        print(f"[data_loader] Model '{model_name}' not found in {csv_path}")
        return df

    # Merge sex (left join — keep all prediction rows)
    df["sex"] = df["patient_id"].map(gt_sex)

    # Validate required columns
    required = {"patient_id", "target", "pred_proba", "pred_label", "selected_threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df
