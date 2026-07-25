import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import config_extended as config


# DeLong test implementation
# Reference: DeLong et al. (1988) Comparing the areas under two or more
#            correlated receiver operating characteristic curves.


def _placement_values(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute placement values V10, V01 needed for DeLong variance."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # V10[i] = (1/n_neg) * sum_{j} psi(pos_i, neg_j)
    # V01[j] = (1/n_pos) * sum_{i} psi(pos_i, neg_j)
    # psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
    V10 = np.zeros(n_pos)
    V01 = np.zeros(n_neg)

    for i, p in enumerate(pos_scores):
        psi = (neg_scores < p).astype(float) + 0.5 * (neg_scores == p).astype(float)
        V10[i] = psi.mean()

    for j, n in enumerate(neg_scores):
        psi = (pos_scores > n).astype(float) + 0.5 * (pos_scores == n).astype(float)
        V01[j] = psi.mean()

    return V10, V01


def delong_auc_variance(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute AUC and its variance using the DeLong method.

    Returns
    -------
    auc : float
    var : float
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)

    V10, V01 = _placement_values(scores, labels)
    auc = V10.mean()

    var_V10 = np.var(V10, ddof=1) if n_pos > 1 else 0.0
    var_V01 = np.var(V01, ddof=1) if n_neg > 1 else 0.0

    variance = var_V10 / n_pos + var_V01 / n_neg
    return auc, variance


def delong_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Two-sided DeLong test comparing AUC of model A vs model B on shared labels.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Predicted probabilities / scores of the two models (same order as labels).
    labels : np.ndarray
        Ground-truth binary labels (0/1).

    Returns
    -------
    auc_a, auc_b, z_stat, p_value
    """
    # Remove rows where either score is NaN
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b) | np.isnan(labels.astype(float)))
    scores_a = scores_a[mask]
    scores_b = scores_b[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    auc_a, var_a = delong_auc_variance(scores_a, labels)
    auc_b, var_b = delong_auc_variance(scores_b, labels)

    V10_a, V01_a = _placement_values(scores_a, labels)
    V10_b, V01_b = _placement_values(scores_b, labels)

    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())

    # Covariance (DeLong eq. 9)
    cov_10 = np.cov(V10_a, V10_b)[0, 1] if n_pos > 1 else 0.0
    cov_01 = np.cov(V01_a, V01_b)[0, 1] if n_neg > 1 else 0.0
    cov = cov_10 / n_pos + cov_01 / n_neg

    var_diff = var_a + var_b - 2 * cov
    if var_diff <= 0:
        return auc_a, auc_b, np.nan, np.nan

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))  # two-sided
    return auc_a, auc_b, z, p


# Regression comparison: Wilcoxon signed-rank test on paired absolute errors
# Reference: Wilcoxon (1945); scipy.stats.wilcoxon


def wilcoxon_error_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Two-sided Wilcoxon signed-rank test comparing absolute prediction errors
    of two models on paired observations.

    Rationale
    ---------
    - Paired design: same patients evaluated by both models — the signed-rank
      test correctly exploits this pairing.
    - Non-parametric: no normality assumption on errors, appropriate for
      small samples (n=19–69 in this study).
    - Tests H0: the distribution of paired error differences
      |error_A| - |error_B| is symmetric around zero, i.e. neither model
      has systematically smaller absolute errors.
    - Directly analogous to DeLong for classification: both test whether one
      model is systematically better than the other on paired data.

    Parameters
    ----------
    preds_a, preds_b : np.ndarray
        Predicted continuous values of the two models (pred_value column).
    targets : np.ndarray
        Ground-truth continuous values.

    Returns
    -------
    rmse_a, rmse_b, stat, p_value
        rmse_a, rmse_b: RMSE of each model (for reporting)
        stat: Wilcoxon test statistic
        p_value: two-sided p-value
    """
    mask = ~(np.isnan(preds_a) | np.isnan(preds_b) | np.isnan(targets))
    preds_a = preds_a[mask]
    preds_b = preds_b[mask]
    targets = targets[mask]

    n = len(targets)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan

    rmse_a = float(np.sqrt(np.mean((preds_a - targets) ** 2)))
    rmse_b = float(np.sqrt(np.mean((preds_b - targets) ** 2)))

    errors_a = np.abs(preds_a - targets)
    errors_b = np.abs(preds_b - targets)

    diff = errors_a - errors_b

    # If all differences are zero (identical predictions), test is undefined
    if np.all(diff == 0):
        return rmse_a, rmse_b, np.nan, np.nan

    stat, p_value = wilcoxon(errors_a, errors_b, alternative="two-sided")
    return rmse_a, rmse_b, float(stat), float(p_value)


# Multiple comparison correction


def apply_correction(
    p_values: List[Optional[float]],
    method: str = config.CORRECTION_METHOD,
    alpha: float = config.ALPHA,
) -> List[Optional[float]]:
    """
    Apply multiple comparison correction to a list of p-values.

    Non-computable p-values (None / NaN) are passed through unchanged.

    Parameters
    ----------
    p_values : list of float or None
    method : str
        "fdr_bh", "bonferroni", or "none"
    alpha : float

    Returns
    -------
    list of corrected p-values (same length, None preserved).
    """
    if method == "none":
        return p_values

    # Indices where we actually have a p-value
    valid_idx = [i for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    if not valid_idx:
        return p_values

    raw = np.array([p_values[i] for i in valid_idx])
    _, corrected, _, _ = multipletests(raw, alpha=alpha, method=method)

    result = list(p_values)
    for i, idx in enumerate(valid_idx):
        result[idx] = float(corrected[i])
    return result


# High-level: compute all p-values for one task × subset block


def compute_pvalues_block(
    task: str,
    subset_key: str,
    predictions: Dict[str, Optional[pd.DataFrame]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute raw and corrected p-values for all comparison pairs in one block
    (one task × subset combination).

    Parameters
    ----------
    task : str  "cls" or "reg"
    subset_key : str  "train", "test_1", "test_2"
    predictions : dict  method_name -> predictions DataFrame (or None)
        Each DataFrame may contain multiple model_names per patient.
        We filter to the specific model_name before merging.

    Returns
    -------
    dict keyed by (new_method, baseline_method) ->
        {"raw": p_raw, "corrected": p_corrected}

    Notes
    -----
    Classification: uses pred_proba (calibrated probability in [0,1]) for the
        DeLong test. pred_proba preserves ranking and is the standard input
        for ROC/AUC-based comparisons.
    Regression: uses pred_value (predicted continuous value, e.g. kg for hand
        grip) for the Wilcoxon signed-rank test on paired absolute errors.
        This is non-parametric, exploits the paired structure, and is
        appropriate for the small sample sizes in this study (n=19–69).
    """
    # Score columns per task:
    #   cls -> pred_proba: calibrated probability [0,1], standard input for DeLong/AUC
    #   reg -> pred_value: predicted continuous value (e.g. kg for hand grip), used for RMSE
    SCORE_COL = {"cls": "pred_proba", "reg": "pred_value", "cls_2": "pred_proba"}
    score_col = SCORE_COL[task]

    raw_pvalues: List[Optional[float]] = []

    for new_method, baseline_method in config.COMPARISON_PAIRS:
        pred_new_df = predictions.get(new_method)
        pred_base_df = predictions.get(baseline_method)

        if pred_new_df is None or pred_base_df is None:
            raw_pvalues.append(None)
            continue

        # ── Filter to the specific model_name before merging ─────────────────
        # Each CSV contains rows for multiple models; without this filter the
        # merge on patient_id alone would produce a cartesian product.
        pred_new_filtered = pred_new_df[pred_new_df["model_name"] == new_method].copy()
        pred_base_filtered = pred_base_df[pred_base_df["model_name"] == baseline_method].copy()

        if pred_new_filtered.empty or pred_base_filtered.empty:
            print(
                f"  [WARNING] model_name not found in predictions: "
                f"'{new_method}' or '{baseline_method}'"
            )
            raw_pvalues.append(None)
            continue

        # ── Verify score column exists ────────────────────────────────────────
        # Fallback to pred_score if the expected column is absent
        def _resolve_score_col(df: pd.DataFrame, preferred: str, method: str) -> str:
            if preferred in df.columns:
                return preferred
            print(
                f"  [WARNING] Column '{preferred}' not found for '{method}', "
                f"falling back to 'pred_score'."
            )
            return "pred_score"

        score_col_new = _resolve_score_col(pred_new_filtered, score_col, new_method)
        score_col_base = _resolve_score_col(pred_base_filtered, score_col, baseline_method)

        # ── Merge on patient_id (one row per patient per model now) ───────────
        merged = pd.merge(
            pred_new_filtered[["patient_id", "target", score_col_new]].rename(
                columns={score_col_new: "score_new"}
            ),
            pred_base_filtered[["patient_id", score_col_base]].rename(
                columns={score_col_base: "score_base"}
            ),
            on="patient_id",
            how="inner",
        ).dropna(subset=["score_new", "score_base", "target"])

        if len(merged) < 5:
            print(
                f"  [WARNING] Too few aligned patients ({len(merged)}) for [{new_method}] vs [{baseline_method}]."
            )
            raw_pvalues.append(None)
            continue

        targets = merged["target"].values.astype(float)
        s_new = merged["score_new"].values.astype(float)
        s_base = merged["score_base"].values.astype(float)

        if task == "cls":
            _, _, _, p = delong_test(s_new, s_base, targets)
            raw_pvalues.append(p if p is not None and not np.isnan(p) else None)
        elif task == "cls_2":
            _, _, _, p = delong_test(s_new, s_base, targets)
            raw_pvalues.append(p if p is not None and not np.isnan(p) else None)
        else:
            _, _, _, p = wilcoxon_error_test(s_new, s_base, targets)
            raw_pvalues.append(p if p is not None and not np.isnan(p) else None)

    corrected_pvalues = apply_correction(raw_pvalues)

    result = {}
    for i, (new_method, baseline_method) in enumerate(config.COMPARISON_PAIRS):
        result[(new_method, baseline_method)] = {
            "raw": raw_pvalues[i],
            "corrected": corrected_pvalues[i],
        }
    return result
