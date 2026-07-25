import os
import pandas as pd
from typing import Dict, Optional, Tuple

import config_extended as config


def _get_method_subfolder(method_name: str) -> str:
    """Return the output subfolder for a given method."""
    return config.METHOD_DIRS[method_name]


def _build_path(method_name: str, subfolder: str, filename: str) -> str:
    """Build absolute path to a metrics or predictions file."""
    method_dir = _get_method_subfolder(method_name)
    return os.path.join(config.ROOT_OUTPUT_DIR, method_dir, subfolder, filename)


def load_metrics(task: str, subset_key: str, method_name: str) -> Optional[pd.DataFrame]:
    filename = config.DATA_SUBSETS[task][subset_key]["metrics_file"]
    path = _build_path(method_name, "metrics", filename)
    if not os.path.isfile(path):
        print(f"  [WARNING] Metrics file not found: {path}")
        return None
    df = pd.read_excel(path)
    return df


def load_predictions(task: str, subset_key: str, method_name: str) -> Optional[pd.DataFrame]:
    filename = config.DATA_SUBSETS[task][subset_key]["predictions_file"]
    path = _build_path(method_name, "predictions", filename)
    if not os.path.isfile(path):
        print(f"  [WARNING] Predictions file not found: {path}")
        return None
    df = pd.read_csv(path)
    return df


def get_method_row(metrics_df: pd.DataFrame, method_name: str) -> Optional[pd.Series]:
    """
    Extract the row for a given model_name from a metrics DataFrame.
    Matches by the method_name key (exact) inside the 'model_name' column.
    """
    if metrics_df is None:
        return None
    matches = metrics_df[metrics_df["model_name"] == method_name]
    if matches.empty:
        print(f"  [WARNING] model_name '{method_name}' not found in metrics DataFrame.")
        return None
    return matches.iloc[0]


def load_all_data() -> Dict:
    data: Dict = {"metrics": {}, "predictions": {}}

    for task in ["cls", "reg", "cls_2"]:
        data["metrics"][task] = {}
        data["predictions"][task] = {}
        for subset_key in ["train", "test_1", "test_2"]:
            data["metrics"][task][subset_key] = {}
            data["predictions"][task][subset_key] = {}
            for method_name in config.METHOD_ORDER:
                print(f"  Loading [{task}][{subset_key}][{method_name}] ...")
                mdf = load_metrics(task, subset_key, method_name)
                row = get_method_row(mdf, method_name)
                data["metrics"][task][subset_key][method_name] = row

                pdf = load_predictions(task, subset_key, method_name)
                data["predictions"][task][subset_key][method_name] = pdf

    return data


def get_n_cases(predictions_df: pd.DataFrame, task: str) -> Tuple[int, Optional[int]]:
    if predictions_df is None or predictions_df.empty:
        return (0, None)

    # Deduplicate: one row per patient regardless of how many models are present
    unique_patients = predictions_df.drop_duplicates(subset=["patient_id"])

    n = len(unique_patients)
    if (task == "cls") or (task == "cls_2"):
        n_pos = (
            int(unique_patients["target"].dropna().astype(int).eq(1).sum())
            if "target" in unique_patients.columns
            else None
        )
        return (n, n_pos)
    return (n, None)
