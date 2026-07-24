import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)


# Safe numeric helpers
def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


def safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def safe_nanstd(values: np.ndarray, ddof: int = 1) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[~np.isnan(values)]
    if finite.size <= ddof:
        return np.nan
    return float(np.nanstd(values, ddof=ddof))


def safe_nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmedian(values))


def safe_nanpercentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanpercentile(values, q))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho, pval = spearmanr(y_true, y_pred)
    rho = float(rho) if np.isfinite(rho) else np.nan
    pval = float(pval) if np.isfinite(pval) else np.nan
    return rho, pval


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return np.nan
    return float(1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1)))


# Point-estimate metrics
def classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute full set of binary-classification metrics at a given threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    auc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_proba))

    sens = safe_divide(tp, tp + fn)
    spec = safe_divide(tn, tn + fp)
    balanced_accuracy = np.nan if np.isnan(sens) or np.isnan(spec) else float((sens + spec) / 2.0)

    return {
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "brier": float(np.mean((y_proba - y_true) ** 2)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "prevalence": float(np.mean(y_true)),
        "n": float(len(y_true)),
        "selected_threshold": float(threshold),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> Dict[str, float]:
    """Compute full set of regression metrics."""
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    r2 = float(r2_score(y_true, y_pred))
    rho, pval = safe_spearman(y_true, y_pred)

    return {
        "r2": r2,
        "adjusted_r2": adjusted_r2(r2=r2, n=len(y_true), p=n_features),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman_r": rho,
        "spearman_p": pval,
        "n": float(len(y_true)),
    }


# Threshold selection
def compute_youden_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fallback_threshold: float = 0.5,
) -> float:
    """Select the Youden-optimal probability threshold from a training set."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(fallback_threshold)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    if youden.size == 0 or np.all(np.isnan(youden)):
        return float(fallback_threshold)

    best_idx = int(np.nanargmax(youden))
    thr = float(thresholds[best_idx])
    if not np.isfinite(thr):
        return float(fallback_threshold)
    return float(np.clip(thr, 0.0, 1.0))


# Bootstrap
@dataclass(frozen=True)
class BootstrapResult:
    samples: pd.DataFrame
    summary: pd.DataFrame


def _summarize_bootstrap_samples(
    df_samples: pd.DataFrame,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict] = []
    for model_name, g in df_samples.groupby("model_name"):
        row: Dict = {"model_name": model_name}
        for col in metric_cols:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = safe_nanmean(vals)
            row[f"{col}_sd"] = safe_nanstd(vals, ddof=1)
            row[f"{col}_median"] = safe_nanmedian(vals)
            row[f"{col}_ci_low"] = safe_nanpercentile(vals, 2.5)
            row[f"{col}_ci_high"] = safe_nanpercentile(vals, 97.5)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def bootstrap_classification_metrics(
    df_pred: pd.DataFrame,
    n_boot: int = 2000,
    threshold: float = 0.5,
    random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for classification metrics on a prediction table."""
    required = ["model_name", "target", "pred_proba"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Classification prediction table missing columns: {missing}")

    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []

    for model_name, g in df_pred.groupby("model_name"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        y_true_all = g["target"].to_numpy()
        y_proba_all = g["pred_proba"].to_numpy()
        model_thr = (
            float(g["selected_threshold"].iloc[0])
            if "selected_threshold" in g.columns
            else float(threshold)
        )
        for boot_idx in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = classification_metrics(y_true_all[idx], y_proba_all[idx], model_thr)
            rows.append({"model_name": model_name, "bootstrap_index": boot_idx, **m})

    df_samples = pd.DataFrame(rows)
    metric_cols = [
        "auc", "balanced_accuracy", "sensitivity", "specificity",
        "ppv", "npv", "brier", "selected_threshold",
    ]
    return BootstrapResult(
        samples=df_samples,
        summary=_summarize_bootstrap_samples(df_samples, metric_cols),
    )


def bootstrap_regression_metrics(
    df_pred: pd.DataFrame,
    n_boot: int = 2000,
    n_features_by_model: Optional[Dict[str, int]] = None,
    random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap CI for regression metrics on a prediction table."""
    required = ["model_name", "target", "pred_value"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Regression prediction table missing columns: {missing}")

    n_features_by_model = n_features_by_model or {}
    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []

    for model_name, g in df_pred.groupby("model_name"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        y_true_all = g["target"].to_numpy()
        y_pred_all = g["pred_value"].to_numpy()
        p = int(n_features_by_model.get(model_name, 1))
        for boot_idx in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = regression_metrics(y_true_all[idx], y_pred_all[idx], p)
            rows.append({"model_name": model_name, "bootstrap_index": boot_idx, **m})

    df_samples = pd.DataFrame(rows)
    metric_cols = ["r2", "adjusted_r2", "mae", "rmse", "spearman_r"]
    return BootstrapResult(
        samples=df_samples,
        summary=_summarize_bootstrap_samples(df_samples, metric_cols),
    )


# Calibration
@dataclass(frozen=True)
class CalibrationResult:
    metrics: pd.DataFrame
    curve: pd.DataFrame


def _clip_proba(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = _clip_proba(p, eps=eps)
    return np.log(p / (1.0 - p))


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    lp = _logit(np.asarray(y_proba, dtype=float)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan}
    try:
        model = LogisticRegression(C=1e12, penalty="l2", solver="lbfgs", max_iter=10000)
        model.fit(lp, y_true)
        return {
            "calibration_intercept": float(model.intercept_[0]),
            "calibration_slope": float(model.coef_.reshape(-1)[0]),
        }
    except Exception:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan}


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_proba, float) - np.asarray(y_true, float)) ** 2))


def calibration_curve_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    if strategy == "quantile":
        df["bin"] = pd.qcut(df["y_proba"], q=n_bins, duplicates="drop")
    elif strategy == "uniform":
        df["bin"] = pd.cut(df["y_proba"], bins=n_bins, include_lowest=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rows: List[Dict] = []
    for i, (label, g) in enumerate(df.groupby("bin", observed=False)):
        if len(g) == 0:
            continue
        rows.append({
            "bin_index": int(i),
            "bin_label": str(label),
            "n_bin": int(len(g)),
            "pred_mean": float(np.mean(g["y_proba"])),
            "pred_min": float(np.min(g["y_proba"])),
            "pred_max": float(np.max(g["y_proba"])),
            "obs_rate": float(np.mean(g["y_true"])),
            "obs_count": int(np.sum(g["y_true"])),
        })
    return pd.DataFrame(rows)


def evaluate_calibration(
    df_pred: pd.DataFrame,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> CalibrationResult:
    required = ["model_name", "target", "pred_proba"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Prediction table missing columns: {missing}")

    metric_rows: List[Dict] = []
    curve_rows: List[pd.DataFrame] = []

    for model_name, g in df_pred.groupby("model_name"):
        y_true = g["target"].to_numpy()
        y_proba = g["pred_proba"].to_numpy()
        cal = calibration_intercept_slope(y_true, y_proba)
        metric_rows.append({
            "model_name": model_name,
            "n": int(len(g)),
            "n_bins": int(n_bins),
            "binning_strategy": strategy,
            "brier": brier_score(y_true, y_proba),
            **cal,
        })
        curve = calibration_curve_table(y_true, y_proba, n_bins=n_bins, strategy=strategy)
        curve["model_name"] = model_name
        curve_rows.append(curve)

    return CalibrationResult(
        metrics=pd.DataFrame(metric_rows).sort_values("model_name").reset_index(drop=True),
        curve=pd.concat(curve_rows, axis=0, ignore_index=True) if curve_rows else pd.DataFrame(),
    )


def bootstrap_calibration(
    df_pred: pd.DataFrame,
    n_boot: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    required = ["model_name", "target", "pred_proba"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Prediction table missing columns: {missing}")

    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []
    for model_name, g in df_pred.groupby("model_name"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        y_true_all = g["target"].to_numpy()
        y_proba_all = g["pred_proba"].to_numpy()
        for boot_idx in range(n_boot):
            idx = rng.integers(0, n, size=n)
            cal = calibration_intercept_slope(y_true_all[idx], y_proba_all[idx])
            rows.append({
                "model_name": model_name,
                "bootstrap_index": int(boot_idx),
                "brier": brier_score(y_true_all[idx], y_proba_all[idx]),
                **cal,
            })
    return pd.DataFrame(rows)


def summarize_calibration_bootstrap(df_boot: pd.DataFrame) -> pd.DataFrame:
    required = ["model_name", "brier", "calibration_intercept", "calibration_slope"]
    missing = [c for c in required if c not in df_boot.columns]
    if missing:
        raise KeyError(f"Calibration bootstrap table missing columns: {missing}")

    rows: List[Dict] = []
    for model_name, g in df_boot.groupby("model_name"):
        row: Dict = {"model_name": model_name}
        for col in ["brier", "calibration_intercept", "calibration_slope"]:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = safe_nanmean(vals)
            row[f"{col}_sd"] = safe_nanstd(vals, ddof=1)
            row[f"{col}_ci_low"] = safe_nanpercentile(vals, 2.5)
            row[f"{col}_ci_high"] = safe_nanpercentile(vals, 97.5)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


# Decision-curve analysis
@dataclass(frozen=True)
class DecisionCurveResult:
    curves: pd.DataFrame
    summary: pd.DataFrame


def make_threshold_grid(
    start: float = 0.10,
    stop: float = 0.80,
    step: float = 0.01,
) -> np.ndarray:
    if not (0.0 < start < stop < 1.0):
        raise ValueError(f"Invalid threshold range: start={start}, stop={stop}")
    if step <= 0:
        raise ValueError(f"step must be > 0, got: {step}")
    grid = np.arange(start, stop + 1e-12, step, dtype=float)
    return grid[(grid > 0.0) & (grid < 1.0)]


def _net_benefit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    pred_pos = y_proba >= threshold
    tp = int(np.sum((y_true == 1) & pred_pos))
    fp = int(np.sum((y_true == 0) & pred_pos))
    n = int(len(y_true))
    odds = threshold / (1.0 - threshold)
    nb = (tp / n) - (fp / n) * odds
    prevalence = float(np.mean(y_true))
    snb = nb / prevalence if prevalence > 0 else np.nan
    return {
        "threshold": float(threshold),
        "net_benefit": float(nb),
        "standardized_net_benefit": float(snb),
        "tp": float(tp), "fp": float(fp),
        "n": float(n), "prevalence": prevalence,
    }


def _treat_all_nb(y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true))
    odds = threshold / (1.0 - threshold)
    nb = prevalence - (1.0 - prevalence) * odds
    snb = nb / prevalence if prevalence > 0 else np.nan
    return {"threshold": float(threshold), "net_benefit": float(nb),
            "standardized_net_benefit": float(snb), "prevalence": prevalence}


def _treat_none_nb(y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    prevalence = float(np.mean(np.asarray(y_true).astype(int)))
    return {"threshold": float(threshold), "net_benefit": 0.0,
            "standardized_net_benefit": 0.0 if prevalence > 0 else np.nan,
            "prevalence": prevalence}


def compute_decision_curves(
    df_pred: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
) -> DecisionCurveResult:
    required = ["model_name", "target", "pred_proba"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Prediction table missing columns: {missing}")

    if thresholds is None:
        thresholds = make_threshold_grid()
    thresholds = np.asarray(list(thresholds), dtype=float)

    curve_rows: List[Dict] = []
    for model_name, g in df_pred.groupby("model_name"):
        y_true = g["target"].to_numpy()
        y_proba = g["pred_proba"].to_numpy()
        for thr in thresholds:
            row = _net_benefit(y_true, y_proba, float(thr))
            row["model_name"] = model_name
            row["strategy"] = "model"
            curve_rows.append(row)

    y_ref = next(iter(df_pred.groupby("model_name")))[1]["target"].to_numpy()
    for thr in thresholds:
        r_none = _treat_none_nb(y_ref, float(thr))
        r_none["model_name"] = "treat_none"
        r_none["strategy"] = "reference"
        curve_rows.append(r_none)

        r_all = _treat_all_nb(y_ref, float(thr))
        r_all["model_name"] = "treat_all"
        r_all["strategy"] = "reference"
        curve_rows.append(r_all)

    curves = (
        pd.DataFrame(curve_rows)
        .sort_values(["strategy", "model_name", "threshold"])
        .reset_index(drop=True)
    )
    summary_rows: List[Dict] = []
    for model_name, g in curves.groupby("model_name"):
        if model_name in {"treat_all", "treat_none"}:
            continue
        summary_rows.append({
            "model_name": model_name,
            "n_thresholds": int(len(g)),
            "threshold_min": float(np.min(g["threshold"])),
            "threshold_max": float(np.max(g["threshold"])),
            "net_benefit_mean": safe_nanmean(g["net_benefit"].to_numpy(dtype=float)),
            "net_benefit_max": float(np.nanmax(g["net_benefit"])),
            "net_benefit_min": float(np.nanmin(g["net_benefit"])),
            "standardized_net_benefit_mean": safe_nanmean(g["standardized_net_benefit"].to_numpy(dtype=float)),
        })
    summary = pd.DataFrame(summary_rows).sort_values("model_name").reset_index(drop=True)
    return DecisionCurveResult(curves=curves, summary=summary)


def bootstrap_decision_curves(
    df_pred: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
    n_boot: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    required = ["model_name", "target", "pred_proba"]
    missing = [c for c in required if c not in df_pred.columns]
    if missing:
        raise KeyError(f"Prediction table missing columns: {missing}")

    if thresholds is None:
        thresholds = make_threshold_grid()
    thresholds = np.asarray(list(thresholds), dtype=float)

    rng = np.random.default_rng(random_state)
    rows: List[Dict] = []
    for model_name, g in df_pred.groupby("model_name"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        y_true_all = g["target"].to_numpy()
        y_proba_all = g["pred_proba"].to_numpy()
        for boot_idx in range(n_boot):
            idx = rng.integers(0, n, size=n)
            for thr in thresholds:
                out = _net_benefit(y_true_all[idx], y_proba_all[idx], float(thr))
                rows.append({
                    "model_name": model_name,
                    "bootstrap_index": int(boot_idx),
                    "threshold": float(thr),
                    "net_benefit": out["net_benefit"],
                    "standardized_net_benefit": out["standardized_net_benefit"],
                    "prevalence": out["prevalence"],
                    "n": out["n"],
                })
    return pd.DataFrame(rows)


def summarize_bootstrap_decision_curves(df_boot: pd.DataFrame) -> pd.DataFrame:
    required = ["model_name", "threshold", "net_benefit", "standardized_net_benefit"]
    missing = [c for c in required if c not in df_boot.columns]
    if missing:
        raise KeyError(f"Bootstrap DCA table missing columns: {missing}")

    rows: List[Dict] = []
    for (model_name, threshold), g in df_boot.groupby(["model_name", "threshold"]):
        nb = g["net_benefit"].to_numpy(dtype=float)
        snb = g["standardized_net_benefit"].to_numpy(dtype=float)
        rows.append({
            "model_name": model_name,
            "threshold": float(threshold),
            "net_benefit_mean": safe_nanmean(nb),
            "net_benefit_sd": safe_nanstd(nb),
            "net_benefit_ci_low": safe_nanpercentile(nb, 2.5),
            "net_benefit_ci_high": safe_nanpercentile(nb, 97.5),
            "standardized_net_benefit_mean": safe_nanmean(snb),
            "standardized_net_benefit_sd": safe_nanstd(snb),
            "standardized_net_benefit_ci_low": safe_nanpercentile(snb, 2.5),
            "standardized_net_benefit_ci_high": safe_nanpercentile(snb, 97.5),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "threshold"]).reset_index(drop=True)
