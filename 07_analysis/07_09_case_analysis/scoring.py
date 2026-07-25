from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from config import RCFG, RunConfig, SCORE_MAP_NAMES


# Feature-table reading (mirrors dataset.py logic for the three relevant maps)
def _read_signature(
    feature_root: Path,
    map_name: str,
    cohort_label: str,
) -> pd.DataFrame:
    """Load signature CSV for one spectral map and one cohort."""
    suffix = "" if cohort_label == "cohort1" else "_cohort_2"
    fname = f"signature_spectral_radiomics_signature_{map_name}{suffix}.csv"
    path = feature_root / map_name / fname
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise KeyError(f"Signature file {path} missing 'patient_id' column.")
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def _build_feature_row(
    patient_id: str,
    cohort_label: str,
    feature_root: Path,
    gt_row: pd.Series,
    clinical_features: tuple = ("age", "sex", "bmi"),
) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with all signature features plus clinical
    columns needed for the mf_ct_scores_clinical_3d model.
    """
    row = {"patient_id": patient_id}

    # Add clinical features from the GT table row
    for col in clinical_features:
        row[col] = gt_row.get(col, np.nan)

    # Add a dummy target (required by get_X_y even though we do not use it)
    row["target"] = np.nan

    for map_name in SCORE_MAP_NAMES:
        df_sig = _read_signature(feature_root, map_name, cohort_label)
        patient_df = df_sig[df_sig["patient_id"] == patient_id]
        if patient_df.empty:
            raise ValueError(
                f"Patient {patient_id} not found in signature file for map '{map_name}'."
            )
        feat_cols = [c for c in patient_df.columns if c != "patient_id"]
        for col in feat_cols:
            row[col] = patient_df.iloc[0][col]

    return pd.DataFrame([row])


# Score extraction
def extract_patient_scores(
    patient_id: str,
    cohort_label: str,
    gt_row: pd.Series,
    run_cfg: RunConfig = RCFG,
) -> Dict[str, float]:
    # ---- Load saved model --------------------------------------------------
    model_path = run_cfg.paths.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    fitted = joblib.load(model_path)

    # ---- Retrieve stored threshold -----------------------------------------
    threshold = float(fitted.get("selected_threshold", 0.5))

    # ---- Build feature row -------------------------------------------------
    df_patient = _build_feature_row(
        patient_id=patient_id,
        cohort_label=cohort_label,
        feature_root=run_cfg.paths.feature_root,
        gt_row=gt_row,
    )

    # ---- Predict using the multi_score dispatch ----------------------------
    # Import lazily to avoid a hard dependency at module load time.
    try:
        from models_base import predict_multi_score
    except ImportError as e:
        raise ImportError("Could not import models_base (expected alongside this file).") from e

    pred = predict_multi_score(fitted, df_patient)

    # ---- Map output keys to readable names ---------------------------------
    # predict_multi_score returns:
    #   score__<group_name>  → per-map decision score, where <group_name> is
    #                          e.g. "musclefat_signature_3d" or "ct_signature_3d"
    #   score                → combined decision score (stage-2)
    #   proba                → predicted probability
    #
    # The exact group name (including any "_3d" postfix) is taken from the
    # fitted model itself rather than reconstructed, since it depends on
    # whether the model was trained on 2-D or 3-D features.
    result: Dict[str, float] = {}

    group_names = list(fitted["group_features"].keys())
    for map_name in SCORE_MAP_NAMES:
        group_name = next((g for g in group_names if g.startswith(map_name)), None)
        if group_name is None:
            raise KeyError(
                f"Could not find a score group starting with '{map_name}' "
                f"in fitted model. Available groups: {group_names}"
            )
        group_key = f"score__{group_name}"
        if group_key not in pred:
            raise KeyError(
                f"Expected key '{group_key}' in model output. "
                f"Available keys: {list(pred.keys())}"
            )
        result[f"{map_name}_score"] = float(pred[group_key][0])

    result["combined_score"] = float(pred["score"][0])
    result["pred_proba"] = float(pred["proba"][0])
    result["model_threshold"] = threshold

    return result
