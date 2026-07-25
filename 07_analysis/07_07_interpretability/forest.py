import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import (
    ICFG,
    MODEL_NAME,
    SCORE_MAP_NAMES,
    CLINICAL_FEATURES,
    SCORE_LEVEL_LABELS,
    InterpretabilityConfig,
)


# Internal helpers
def _exp_coef_to_or(coef: float) -> float:
    return float(np.exp(coef))


def _bootstrap_p_value(boot_coefs: np.ndarray) -> float:
    """Two-sided bootstrap p-value: proportion of samples on the wrong side of 0."""
    n = len(boot_coefs)
    if n == 0:
        return np.nan
    # p = 2 * min(P(coef > 0), P(coef < 0))
    p_pos = float(np.sum(boot_coefs > 0)) / n
    p_neg = float(np.sum(boot_coefs < 0)) / n
    return float(2.0 * min(p_pos, p_neg))


def _standardize_X(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns; return (X_std, means, stds)."""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=1)
    stds = np.where(stds == 0, 1.0, stds)  # avoid div-by-zero for constant cols
    return (X - means) / stds, means, stds


def _safe_univariable_or(X_col: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a single unpenalised logistic regression; return (OR, ci_low, ci_high)."""
    X_col = X_col.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lr = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=10000)
            lr.fit(X_col, y)
            coef = float(lr.coef_.reshape(-1)[0])
            # Wald SE from Hessian (Fisher information) — simple approximation
            p_hat = lr.predict_proba(X_col)[:, 1]
            W = p_hat * (1 - p_hat)
            se = float(np.sqrt(1.0 / (np.sum(W) + 1e-12)))
            return (
                _exp_coef_to_or(coef),
                _exp_coef_to_or(coef - 1.96 * se),
                _exp_coef_to_or(coef + 1.96 * se),
            )
        except Exception:
            return np.nan, np.nan, np.nan


# Load model and coefficient table
_TASK_NAME = "sarcopenia_composite_cls"


def load_fitted_model(cfg: InterpretabilityConfig = ICFG) -> Dict:
    """Load the saved joblib model dict for MODEL_NAME."""
    path = cfg.paths.models_dir / f"{_TASK_NAME}__{MODEL_NAME}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Fitted model not found: {path}")
    return joblib.load(path)


def load_coefficient_table(cfg: InterpretabilityConfig = ICFG) -> pd.DataFrame:
    """Load the pre-saved coefficient table and filter to MODEL_NAME."""
    path = cfg.paths.coeff_table
    if not path.exists():
        raise FileNotFoundError(f"Coefficient table not found: {path}")
    df = pd.read_csv(path)
    mask = df["model_name"] == MODEL_NAME
    if not mask.any():
        raise ValueError(
            f"model_name '{MODEL_NAME}' not found in {path}. "
            f"Available: {df['model_name'].unique().tolist()}"
        )
    return df[mask].reset_index(drop=True)


# Bootstrap OR + CI from the fitted model on the train set
def _get_combined_estimator_and_features(
    fitted: Dict,
) -> Tuple[Pipeline, List[str]]:
    """Extract the combined-stage estimator and its input feature names."""
    return fitted["combined_estimator"], list(fitted["combined_features"])


def _get_score_estimator_and_features(fitted: Dict, map_name: str) -> Tuple[Pipeline, List[str]]:
    """Extract a per-map score estimator and its feature names."""
    key = f"{map_name}_signature_3d"
    est = fitted["group_estimators"][key]
    feats = list(fitted["group_features"][key])
    return est, feats


def _bootstrap_multivariable_or(
    fitted: Dict,
    df_train: pd.DataFrame,
    n_boot: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Bootstrap the combined-stage OR by:
      1. Resample rows of df_train with replacement.
      2. Compute scores using the fixed per-map estimators (no refit — fast,
         and the per-map models are already stable).
      3. Refit a logistic regression on the standardised combined-stage inputs
         of the resampled set. This gives a proper bootstrap distribution of
         the combined-stage coefficients.

    Returns a DataFrame with columns: feature, boot_coef.
    (n_boot rows per feature)
    """
    comb_est, comb_feats = _get_combined_estimator_and_features(fitted)

    # Extract combined-model hyperparameters to use in bootstrap refits
    try:
        comb_model = comb_est.named_steps["model"]
        C_val = float(comb_model.C)
    except Exception:
        C_val = 1.0

    # Build combined-stage design matrix for the full train set
    df_scored = _add_scores_to_df(fitted, df_train)
    X_all = df_scored[comb_feats].to_numpy(dtype=float)
    y_all = df_train["target"].to_numpy(dtype=int)

    rng = np.random.default_rng(random_state)
    boot_rows: List[Dict] = []

    for b in range(n_boot):
        idx = rng.integers(0, len(df_train), size=len(df_train))
        X_b = X_all[idx]
        y_b = y_all[idx]

        # Skip degenerate samples (only one class)
        if len(np.unique(y_b)) < 2:
            coefs = np.full(len(comb_feats), np.nan)
        else:
            # Standardise the boot sample
            X_b_std, _, _ = _standardize_X(X_b)
            try:
                lr = LogisticRegression(
                    C=C_val,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=0,
                )
                lr.fit(X_b_std, y_b)
                coefs = lr.coef_.reshape(-1)
            except Exception:
                coefs = np.full(len(comb_feats), np.nan)

        for feat, coef in zip(comb_feats, coefs):
            boot_rows.append({"feature": feat, "boot_coef": float(coef)})

    return pd.DataFrame(boot_rows)


def _add_scores_to_df(fitted: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add score columns to df using the saved per-map estimators.

    Passes a named DataFrame (not a raw array) to each pipeline so that
    sklearn's SimpleImputer does not emit a feature-name UserWarning.
    """
    import warnings as _warnings

    df_out = df.copy()
    for map_name in SCORE_MAP_NAMES:
        key = f"{map_name}_signature_3d"
        est = fitted["group_estimators"][key]
        feat_cols = list(fitted["group_features"][key])
        missing = [c for c in feat_cols if c not in df_out.columns]
        if missing:
            raise KeyError(f"Missing feature columns for map '{map_name}': {missing}")
        X_df = df_out[feat_cols]  # keep as DataFrame → preserves feature names
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            try:
                score = est.predict_proba(X_df)[:, 1]
            except AttributeError:
                score = est.predict(X_df)
        df_out[f"score__{key}"] = score
    return df_out


# Score-level OR table
def compute_score_level_or(
    fitted: Dict,
    df_train: pd.DataFrame,
    cfg: InterpretabilityConfig = ICFG,
) -> pd.DataFrame:
    """
    Compute multivariable and univariable OR at the score level (combined stage).

    Returns a DataFrame with columns:
        feature, label, or_multi, or_multi_ci_low, or_multi_ci_high,
        p_value_multi (optional), or_uni, or_uni_ci_low, or_uni_ci_high
    """
    bcfg = cfg.bootstrap
    comb_est, comb_feats = _get_combined_estimator_and_features(fitted)

    # ---- Build combined-stage input on train ----
    df_scored = _add_scores_to_df(fitted, df_train)
    X_comb = df_scored[comb_feats].to_numpy(dtype=float)
    y = df_train["target"].to_numpy(dtype=int)

    # ---- Multivariable: coefficients from the fitted estimator ----
    # The pipeline's scaler was fitted on training data; coefficients are in
    # standardised space.
    model_step = comb_est.named_steps["model"]
    mv_coefs = model_step.coef_.reshape(-1)

    # ---- Bootstrap CIs for multivariable OR ----
    print(f"  Bootstrap OR (score-level, n_boot={bcfg.n_boot}) ...")
    boot_df = _bootstrap_multivariable_or(
        fitted, df_train, n_boot=bcfg.n_boot, random_state=bcfg.random_state
    )

    rows: List[Dict] = []
    for feat, coef_mv in zip(comb_feats, mv_coefs):
        boot_coefs = boot_df.loc[boot_df["feature"] == feat, "boot_coef"].to_numpy()

        ci_low = float(np.percentile(boot_coefs, 2.5)) if len(boot_coefs) else np.nan
        ci_high = float(np.percentile(boot_coefs, 97.5)) if len(boot_coefs) else np.nan
        p_val = _bootstrap_p_value(boot_coefs) if len(boot_coefs) else np.nan

        # Univariable: standardise this one column and fit unpenalised LR
        col_idx = comb_feats.index(feat)
        x_col_std = (X_comb[:, col_idx] - np.nanmean(X_comb[:, col_idx])) / (
            np.nanstd(X_comb[:, col_idx], ddof=1) + 1e-12
        )
        or_uni, or_uni_lo, or_uni_hi = _safe_univariable_or(x_col_std, y)

        row = {
            "feature": feat,
            "label": SCORE_LEVEL_LABELS.get(feat, feat),
            "or_multi": _exp_coef_to_or(float(coef_mv)),
            "or_multi_ci_low": _exp_coef_to_or(ci_low),
            "or_multi_ci_high": _exp_coef_to_or(ci_high),
            "or_uni": or_uni,
            "or_uni_ci_low": or_uni_lo,
            "or_uni_ci_high": or_uni_hi,
        }
        if not bcfg.omit_pvalues:
            row["p_value_multi"] = round(p_val, 4) if not np.isnan(p_val) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# Feature-level OR table
def compute_feature_level_or(
    fitted: Dict,
    df_train: pd.DataFrame,
    cfg: InterpretabilityConfig = ICFG,
) -> pd.DataFrame:
    """
    Compute multivariable and univariable OR at the raw feature level.

    For each map's score-stage estimator we extract its coefficients (applied
    to its own raw radiomics features). Then we stack the results.
    The combined clinical features are appended from the combined-stage model.

    Returns a DataFrame with columns:
        feature, label, map_name, stage,
        or_multi, or_multi_ci_low, or_multi_ci_high,
        p_value_multi (optional), or_uni, or_uni_ci_low, or_uni_ci_high
    """
    from label_utils import make_feature_label  # defined in label_utils.py

    bcfg = cfg.bootstrap
    rng = np.random.default_rng(bcfg.random_state)
    y = df_train["target"].to_numpy(dtype=int)

    rows: List[Dict] = []

    # ---- Per-map score-stage estimators ----
    for map_name in SCORE_MAP_NAMES:
        key = f"{map_name}_signature_3d"
        est, feat_cols = _get_score_estimator_and_features(fitted, map_name)
        missing = [c for c in feat_cols if c not in df_train.columns]
        if missing:
            raise KeyError(f"Train DataFrame missing feature columns for '{map_name}': {missing}")
        X = df_train[feat_cols].to_numpy(dtype=float)
        model_step = est.named_steps["model"]
        mv_coefs = model_step.coef_.reshape(-1)

        # Bootstrap for this map's estimator — refit on each resample
        print(f"  Bootstrap OR (feature-level, map={map_name}, n_boot={bcfg.n_boot}) ...")
        try:
            C_map = float(est.named_steps["model"].C)
        except Exception:
            C_map = 1.0

        boot_coefs_by_feat: Dict[str, List[float]] = {f: [] for f in feat_cols}
        for _ in range(bcfg.n_boot):
            idx = rng.integers(0, len(df_train), size=len(df_train))
            X_b = X[idx]
            y_b = y[idx]
            if len(np.unique(y_b)) < 2:
                boot_coefs = np.full(len(feat_cols), np.nan)
            else:
                X_b_std, _, _ = _standardize_X(X_b)
                try:
                    lr_b = LogisticRegression(
                        C=C_map,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=0,
                    )
                    lr_b.fit(X_b_std, y_b)
                    boot_coefs = lr_b.coef_.reshape(-1)
                except Exception:
                    boot_coefs = np.full(len(feat_cols), np.nan)
            for feat, coef in zip(feat_cols, boot_coefs):
                boot_coefs_by_feat[feat].append(float(coef))

        for feat, coef_mv in zip(feat_cols, mv_coefs):
            boot_arr = np.array(boot_coefs_by_feat[feat])
            ci_low = float(np.percentile(boot_arr, 2.5))
            ci_high = float(np.percentile(boot_arr, 97.5))
            p_val = _bootstrap_p_value(boot_arr)

            col_idx = feat_cols.index(feat)
            x_col = X[:, col_idx]
            x_col_std = (x_col - np.nanmean(x_col)) / (np.nanstd(x_col, ddof=1) + 1e-12)
            or_uni, or_uni_lo, or_uni_hi = _safe_univariable_or(x_col_std, y)

            row = {
                "feature": feat,
                "label": make_feature_label(feat),
                "map_name": map_name,
                "stage": f"score_stage__{key}",
                "or_multi": _exp_coef_to_or(float(coef_mv)),
                "or_multi_ci_low": _exp_coef_to_or(ci_low),
                "or_multi_ci_high": _exp_coef_to_or(ci_high),
                "or_uni": or_uni,
                "or_uni_ci_low": or_uni_lo,
                "or_uni_ci_high": or_uni_hi,
            }
            if not bcfg.omit_pvalues:
                row["p_value_multi"] = round(p_val, 4) if not np.isnan(p_val) else np.nan
            rows.append(row)

    # ---- Combined-stage clinical features (appended at the bottom) ----
    comb_est, comb_feats = _get_combined_estimator_and_features(fitted)
    df_scored = _add_scores_to_df(fitted, df_train)
    X_comb = df_scored[comb_feats].to_numpy(dtype=float)
    model_step = comb_est.named_steps["model"]
    mv_coefs_comb = model_step.coef_.reshape(-1)

    for feat, coef_mv in zip(comb_feats, mv_coefs_comb):
        if feat not in CLINICAL_FEATURES:
            continue  # skip scores – they are covered above
        col_idx = comb_feats.index(feat)
        x_col = X_comb[:, col_idx]
        x_col_std = (x_col - np.nanmean(x_col)) / (np.nanstd(x_col, ddof=1) + 1e-12)
        or_uni, or_uni_lo, or_uni_hi = _safe_univariable_or(x_col_std, y)
        # CI filled in by the combined-stage bootstrap block below
        row = {
            "feature": feat,
            "label": make_feature_label(feat),
            "map_name": "clinical",
            "stage": "combined_stage",
            "or_multi": _exp_coef_to_or(float(coef_mv)),
            "or_multi_ci_low": np.nan,  # filled below
            "or_multi_ci_high": np.nan,
            "or_uni": or_uni,
            "or_uni_ci_low": or_uni_lo,
            "or_uni_ci_high": or_uni_hi,
        }
        if not bcfg.omit_pvalues:
            row["p_value_multi"] = np.nan
        rows.append(row)

    # Bootstrap CIs for clinical features in the combined stage
    print(
        f"  Bootstrap OR (feature-level, clinical in combined stage, " f"n_boot={bcfg.n_boot}) ..."
    )
    boot_df = _bootstrap_multivariable_or(
        fitted, df_train, n_boot=bcfg.n_boot, random_state=bcfg.random_state
    )
    df_result = pd.DataFrame(rows)
    for feat in CLINICAL_FEATURES:
        mask = df_result["feature"] == feat
        if not mask.any():
            continue
        boot_arr = boot_df.loc[boot_df["feature"] == feat, "boot_coef"].to_numpy()
        if len(boot_arr) == 0:
            continue
        ci_low = _exp_coef_to_or(float(np.percentile(boot_arr, 2.5)))
        ci_high = _exp_coef_to_or(float(np.percentile(boot_arr, 97.5)))
        p_val = _bootstrap_p_value(boot_arr)
        df_result.loc[mask, "or_multi_ci_low"] = ci_low
        df_result.loc[mask, "or_multi_ci_high"] = ci_high
        if not bcfg.omit_pvalues:
            df_result.loc[mask, "p_value_multi"] = (
                round(p_val, 4) if not np.isnan(p_val) else np.nan
            )

    return df_result.reset_index(drop=True)
