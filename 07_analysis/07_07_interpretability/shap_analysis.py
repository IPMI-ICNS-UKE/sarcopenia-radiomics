from typing import Dict, List
import numpy as np
import pandas as pd

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from config import SCORE_MAP_NAMES, CLINICAL_FEATURES
from label_utils import make_feature_label, make_score_label


# Internal helpers
def _require_shap() -> None:
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "The 'shap' package is required for SHAP analysis. " "Install it with: pip install shap"
        )


def _get_pipeline_model(pipeline) -> object:
    """Return the final estimator step from a sklearn Pipeline."""
    return pipeline.named_steps["model"]


def _transform_through_scaler(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final 'model' step."""
    X_t = X.copy().astype(float)
    for name, step in pipeline.steps[:-1]:
        X_t = step.transform(X_t)
    return X_t


def _add_scores_to_df(fitted: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach per-map scores to a copy of df.

    Passes a named DataFrame to each pipeline to avoid sklearn's
    feature-name UserWarning from SimpleImputer.
    """
    import warnings as _warnings

    df_out = df.copy()
    for map_name in SCORE_MAP_NAMES:
        key = f"{map_name}_signature_3d"
        est = fitted["group_estimators"][key]
        feat_cols = list(fitted["group_features"][key])
        missing = [c for c in feat_cols if c not in df_out.columns]
        if missing:
            raise KeyError(f"DataFrame missing columns for map '{map_name}': {missing}")
        X_df = df_out[feat_cols]  # keep as DataFrame → preserves feature names
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            try:
                score = est.predict_proba(X_df)[:, 1]
            except AttributeError:
                score = est.predict(X_df)
        df_out[f"score__{key}"] = score
    return df_out


# Score-level SHAP
def compute_shap_score_level(
    fitted: Dict,
    df: pd.DataFrame,
    split_label: str,
) -> pd.DataFrame:
    """
    Compute SHAP values for the combined-stage model (score level).

    Parameters
    ----------
    fitted      : loaded joblib model dict
    df          : patient DataFrame with features and target
    split_label : "train", "test1", or "test2"

    Returns
    -------
    DataFrame with columns:
        patient_id, split, target,
        feature, label, shap_value, feature_value
    """
    _require_shap()

    comb_est = fitted["combined_estimator"]
    comb_feats = list(fitted["combined_features"])

    # Build combined-stage input
    df_scored = _add_scores_to_df(fitted, df)
    X_raw = df_scored[comb_feats].to_numpy(dtype=float)

    # Transform through scaler so LinearExplainer sees the same space as
    # the logistic regression model
    X_scaled = _transform_through_scaler(comb_est, X_raw)
    model_step = _get_pipeline_model(comb_est)

    explainer = shap.LinearExplainer(model_step, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)
    # shap_values shape: (n_samples, n_features)
    if isinstance(shap_values, list):
        # Multi-class: take positive class
        shap_values = shap_values[1]

    rows: List[Dict] = []
    for i, pid in enumerate(df["patient_id"].tolist()):
        for j, feat in enumerate(comb_feats):
            rows.append(
                {
                    "patient_id": pid,
                    "split": split_label,
                    "target": int(df["target"].iloc[i]),
                    "feature": feat,
                    "label": make_score_label(feat),
                    "shap_value": float(shap_values[i, j]),
                    "feature_value": float(X_raw[i, j]),
                }
            )

    return pd.DataFrame(rows)


# Feature-level SHAP
def compute_shap_feature_level(
    fitted: Dict,
    df: pd.DataFrame,
    split_label: str,
) -> pd.DataFrame:
    """
    Compute SHAP values at the raw feature level.

    For each per-map estimator we compute SHAP values on the raw radiomics
    features. Clinical features are explained via the combined-stage model.
    Results are concatenated.

    Returns
    -------
    DataFrame with columns:
        patient_id, split, target,
        feature, label, map_name, stage,
        shap_value, feature_value
    """
    _require_shap()

    rows: List[Dict] = []

    # ---- Per-map score-stage estimators ----
    for map_name in SCORE_MAP_NAMES:
        key = f"{map_name}_signature_3d"
        est = fitted["group_estimators"][key]
        feat_cols = list(fitted["group_features"][key])
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame missing columns for map '{map_name}': {missing}")
        X_raw = df[feat_cols].to_numpy(dtype=float)
        X_scaled = _transform_through_scaler(est, X_raw)
        model_step = _get_pipeline_model(est)

        explainer = shap.LinearExplainer(
            model_step, X_scaled, feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        for i, pid in enumerate(df["patient_id"].tolist()):
            for j, feat in enumerate(feat_cols):
                rows.append(
                    {
                        "patient_id": pid,
                        "split": split_label,
                        "target": int(df["target"].iloc[i]),
                        "feature": feat,
                        "label": make_feature_label(feat),
                        "map_name": map_name,
                        "stage": f"score_stage__{key}",
                        "shap_value": float(shap_values[i, j]),
                        "feature_value": float(X_raw[i, j]),
                    }
                )

    # ---- Clinical features from combined stage ----
    comb_est = fitted["combined_estimator"]
    comb_feats = list(fitted["combined_features"])
    df_scored = _add_scores_to_df(fitted, df)

    # Only clinical columns
    clinical_idxs = [i for i, f in enumerate(comb_feats) if f in CLINICAL_FEATURES]
    clinical_cols = [comb_feats[i] for i in clinical_idxs]

    X_comb_raw = df_scored[comb_feats].to_numpy(dtype=float)
    X_comb_scaled = _transform_through_scaler(comb_est, X_comb_raw)
    model_comb = _get_pipeline_model(comb_est)

    explainer_comb = shap.LinearExplainer(
        model_comb, X_comb_scaled, feature_perturbation="interventional"
    )
    shap_comb = explainer_comb.shap_values(X_comb_scaled)
    if isinstance(shap_comb, list):
        shap_comb = shap_comb[1]

    for i, pid in enumerate(df["patient_id"].tolist()):
        for j in clinical_idxs:
            feat = comb_feats[j]
            rows.append(
                {
                    "patient_id": pid,
                    "split": split_label,
                    "target": int(df["target"].iloc[i]),
                    "feature": feat,
                    "label": make_feature_label(feat),
                    "map_name": "clinical",
                    "stage": "combined_stage",
                    "shap_value": float(shap_comb[i, j]),
                    "feature_value": float(X_comb_raw[i, j]),
                }
            )

    return pd.DataFrame(rows)


# Public dispatch
def compute_shap(
    fitted: Dict,
    splits: Dict[str, pd.DataFrame],
    level: str,
) -> Dict[str, pd.DataFrame]:
    """
    Compute SHAP values for all requested splits.

    Parameters
    ----------
    fitted  : loaded model dict
    splits  : {"train": df_train, "test1": df_test1, "test2": df_test2}
    level   : "score" or "feature"

    Returns
    -------
    dict mapping split_label → SHAP DataFrame
    """
    fn = compute_shap_score_level if level == "score" else compute_shap_feature_level
    results: Dict[str, pd.DataFrame] = {}
    for split_label, df in splits.items():
        if df.empty:
            print(f"  [SHAP] Skipping empty split '{split_label}'.")
            continue
        print(f"  [SHAP-{level}] Computing for split '{split_label}' (n={len(df)}) ...")
        results[split_label] = fn(fitted, df, split_label)
    return results
