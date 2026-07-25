from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from tqdm import tqdm

from models_base import (
    ModelSpec,
    _score_col,
    extract_coefficients_from_fitted,
    fit_direct,
    fit_multi_score,
    fit_score_plus_clinical,
    get_search_grid,
    predict_direct,
    predict_multi_score,
    predict_score_plus_clinical,
    _add_score_columns,
)
from stats import classification_metrics, compute_youden_threshold, regression_metrics


# Splitter factories
def make_outer_splitter(task_kind: str, n_splits: int, n_repeats: int, random_state: int):
    if task_kind == "classification":
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def make_inner_splitter(task_kind: str, n_splits: int, random_state: int):
    if task_kind == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# Inner selection result
@dataclass(frozen=True)
class InnerSelectionResult:
    best_params: Dict[str, float]
    best_score: float
    all_results: pd.DataFrame


# Threshold helpers
def _train_threshold(
    spec: ModelSpec, df_train: pd.DataFrame, pred_train: Dict, fallback: float
) -> float:
    if spec.task_kind != "classification":
        return np.nan
    return compute_youden_threshold(
        y_true=df_train["target"].to_numpy(),
        y_proba=pred_train["proba"],
        fallback_threshold=fallback,
    )


# Inner CV fold score for a direct spec
def _inner_fold_score(
    spec: ModelSpec,
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    params: Dict[str, float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> float:
    fitted = fit_direct(
        df_tr,
        spec,
        max_iter,
        convergence_tol,
        C=params.get("C"),
        alpha=params.get("alpha"),
        l1_ratio=params["l1_ratio"],
    )
    pred = predict_direct(fitted, df_val)
    if spec.task_kind == "classification":
        m = classification_metrics(
            df_val["target"].to_numpy(), pred["proba"], default_cls_threshold
        )
        return m["auc"]
    m = regression_metrics(df_val["target"].to_numpy(), pred["pred"], len(spec.base_features))
    return m["rmse"]


def _is_better(task_kind: str, score: float, best: float) -> bool:
    if np.isnan(score):
        return False
    return score > best if task_kind == "classification" else score < best


def _init_best(task_kind: str) -> float:
    return -np.inf if task_kind == "classification" else np.inf


# Parameter selection — direct model
def select_best_direct_params(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> InnerSelectionResult:
    grid = get_search_grid(spec.task_kind, l1_ratios, logistic_cs, linear_alphas)
    metric = "auc" if spec.task_kind == "classification" else "rmse"
    X = df_outer_train.loc[:, list(spec.base_features)].copy()
    y = df_outer_train["target"].to_numpy()

    rows: List[Dict] = []
    best_score = _init_best(spec.task_kind)
    best_params: Optional[Dict] = None

    for params in grid:
        fold_scores: List[float] = []
        for tr_idx, val_idx in inner_splitter.split(X, y):
            score = _inner_fold_score(
                spec,
                df_outer_train.iloc[tr_idx].reset_index(drop=True),
                df_outer_train.iloc[val_idx].reset_index(drop=True),
                params,
                max_iter,
                convergence_tol,
                default_cls_threshold,
            )
            fold_scores.append(score)
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else np.nan
        rows.append({**params, f"mean_inner_{metric}": mean_score})
        if best_params is None or _is_better(spec.task_kind, mean_score, best_score):
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError(f"Could not select params for model: {spec.name}")

    all_results = (
        pd.DataFrame(rows)
        .sort_values(f"mean_inner_{metric}", ascending=(spec.task_kind == "regression"))
        .reset_index(drop=True)
    )
    return InnerSelectionResult(
        best_params=best_params, best_score=float(best_score), all_results=all_results
    )


# Parameter selection — score+clinical (two-stage)
def _oof_scores_for_stage2(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    score_params: Dict[str, float],
    inner_splitter,
    max_iter: int,
    convergence_tol: float,
) -> pd.DataFrame:
    """Generate OOF scores for stage-2 without leakage."""
    score_spec = ModelSpec(
        name=f"{spec.name}__stage1",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=spec.base_features,
        clinical_features=spec.clinical_features,
    )
    X = df_outer_train.loc[:, list(spec.base_features)].copy()
    y = df_outer_train["target"].to_numpy()
    oof_score = np.full(len(df_outer_train), np.nan, dtype=float)

    for tr_idx, val_idx in inner_splitter.split(X, y):
        fitted = fit_direct(
            df_outer_train.iloc[tr_idx].reset_index(drop=True),
            score_spec,
            max_iter,
            convergence_tol,
            C=score_params.get("C"),
            alpha=score_params.get("alpha"),
            l1_ratio=score_params["l1_ratio"],
        )
        pred = predict_direct(fitted, df_outer_train.iloc[val_idx].reset_index(drop=True))
        oof_score[val_idx] = pred["score"]

    if np.isnan(oof_score).any():
        raise RuntimeError(f"OOF stage-1 score has NaNs for: {spec.name}")
    return _add_score_columns(df_outer_train, {"score": oof_score})


def select_best_score_stage_params(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> InnerSelectionResult:
    score_spec = ModelSpec(
        name=f"{spec.name}__stage1",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=spec.base_features,
        clinical_features=spec.clinical_features,
    )
    return select_best_direct_params(
        df_outer_train,
        score_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )


def select_best_combined_stage_params(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    score_params: Dict[str, float],
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> InnerSelectionResult:
    df_s2 = _oof_scores_for_stage2(
        df_outer_train, spec, score_params, inner_splitter, max_iter, convergence_tol
    )
    combined_spec = ModelSpec(
        name=f"{spec.name}__stage2",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=("score",) + spec.clinical_features,
        clinical_features=spec.clinical_features,
    )
    return select_best_direct_params(
        df_s2,
        combined_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )


# Parameter selection — multi-score (one param set per group + combined)
def select_best_multi_score_params(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> Tuple[Dict[str, InnerSelectionResult], InnerSelectionResult]:
    """Select params for each score group independently, then for combined stage."""
    group_results: Dict[str, InnerSelectionResult] = {}
    for group_name, feature_cols in spec.score_groups:
        group_spec = ModelSpec(
            name=f"{spec.name}__{group_name}",
            kind="direct",
            task_kind=spec.task_kind,
            base_features=feature_cols,
            clinical_features=spec.clinical_features,
        )
        group_results[group_name] = select_best_direct_params(
            df_outer_train,
            group_spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )

    # Build OOF scores for combined stage
    y = df_outer_train["target"].to_numpy()
    oof_scores_map: Dict[str, np.ndarray] = {}
    for group_name, feature_cols in spec.score_groups:
        X_group = df_outer_train.loc[:, list(feature_cols)].copy()
        oof_score = np.full(len(df_outer_train), np.nan, dtype=float)
        bp = group_results[group_name].best_params
        for tr_idx, val_idx in inner_splitter.split(X_group, y):
            g_spec = ModelSpec(
                name=f"{spec.name}__{group_name}",
                kind="direct",
                task_kind=spec.task_kind,
                base_features=feature_cols,
                clinical_features=spec.clinical_features,
            )
            fitted = fit_direct(
                df_outer_train.iloc[tr_idx].reset_index(drop=True),
                g_spec,
                max_iter,
                convergence_tol,
                C=bp.get("C"),
                alpha=bp.get("alpha"),
                l1_ratio=bp["l1_ratio"],
            )
            pred = predict_direct(fitted, df_outer_train.iloc[val_idx].reset_index(drop=True))
            oof_score[val_idx] = pred["score"]
        oof_scores_map[_score_col(group_name)] = oof_score

    df_s2 = _add_score_columns(df_outer_train, oof_scores_map)
    combined_spec = ModelSpec(
        name=f"{spec.name}__combined",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=spec.all_features,
        clinical_features=spec.clinical_features,
    )
    combined_sel = select_best_direct_params(
        df_s2,
        combined_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )
    return group_results, combined_sel


# Build prediction dataframe
def _build_pred_df(
    spec: ModelSpec,
    df_eval: pd.DataFrame,
    pred: Dict,
    selected_threshold: float,
    extra_cols: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    base_cols = [c for c in ["patient_id", "cohort", "split", "target"] if c in df_eval.columns]
    out = df_eval[base_cols].copy()
    out["model_name"] = spec.name
    out["pred_score"] = pred["score"]
    if spec.task_kind == "classification":
        out["pred_proba"] = pred["proba"]
        out["pred_label"] = (pred["proba"] >= selected_threshold).astype(int)
        out["selected_threshold"] = selected_threshold
    else:
        out["pred_value"] = pred["pred"]
    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v
    return out


# One outer fold
def _run_one_fold(
    *,
    df_outer_train: pd.DataFrame,
    df_outer_valid: pd.DataFrame,
    spec: ModelSpec,
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
    repeat_idx: int,
    fold_idx: int,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Run one outer fold for any model kind. Returns (metric_row, pred_df, tuning_df)."""

    # ---- hyper-parameter selection ----
    if spec.kind == "direct":
        inner_sel = select_best_direct_params(
            df_outer_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        bp = inner_sel.best_params
        fitted = fit_direct(
            df_outer_train,
            spec,
            max_iter,
            convergence_tol,
            C=bp.get("C"),
            alpha=bp.get("alpha"),
            l1_ratio=bp["l1_ratio"],
        )
        pred_tr = predict_direct(fitted, df_outer_train)
        pred_val = predict_direct(fitted, df_outer_valid)
        thr = _train_threshold(spec, df_outer_train, pred_tr, default_cls_threshold)
        best_params_out = dict(bp)
        tuning_df = inner_sel.all_results.assign(
            model_name=spec.name,
            repeat_index=repeat_idx,
            fold_index=fold_idx,
            stage="single_stage",
        )

    elif spec.kind == "score_plus_clinical":
        score_sel = select_best_score_stage_params(
            df_outer_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        comb_sel = select_best_combined_stage_params(
            df_outer_train,
            spec,
            score_sel.best_params,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        fitted = fit_score_plus_clinical(
            df_outer_train,
            spec,
            max_iter,
            convergence_tol,
            score_C=score_sel.best_params.get("C"),
            score_alpha=score_sel.best_params.get("alpha"),
            score_l1_ratio=score_sel.best_params["l1_ratio"],
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_score_plus_clinical(fitted, df_outer_train)
        pred_val = predict_score_plus_clinical(fitted, df_outer_valid)
        thr = _train_threshold(spec, df_outer_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "score_C": score_sel.best_params.get("C", np.nan),
            "score_alpha": score_sel.best_params.get("alpha", np.nan),
            "score_l1_ratio": score_sel.best_params["l1_ratio"],
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_alpha": comb_sel.best_params.get("alpha", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        tuning_df = pd.concat(
            [
                score_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage="score_stage",
                ),
                comb_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage="combined_stage",
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    elif spec.kind in ("multi_score", "multi_score_plus_clinical"):
        group_sels, comb_sel = select_best_multi_score_params(
            df_outer_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        score_params_by_group = {g: sel.best_params for g, sel in group_sels.items()}
        fitted = fit_multi_score(
            df_outer_train,
            spec,
            max_iter,
            convergence_tol,
            score_params_by_group=score_params_by_group,
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_multi_score(fitted, df_outer_train)
        pred_val = predict_multi_score(fitted, df_outer_valid)
        thr = _train_threshold(spec, df_outer_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_alpha": comb_sel.best_params.get("alpha", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        tuning_parts = []
        for g_name, g_sel in group_sels.items():
            tuning_parts.append(
                g_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage=f"score_stage__{g_name}",
                )
            )
        tuning_parts.append(
            comb_sel.all_results.assign(
                model_name=spec.name,
                repeat_index=repeat_idx,
                fold_index=fold_idx,
                stage="combined_stage",
            )
        )
        tuning_df = pd.concat(tuning_parts, axis=0, ignore_index=True)

    elif spec.kind == "two_scores_plus_clinical":
        from models_base import fit_two_scores_plus_clinical, predict_two_scores_plus_clinical

        ct_sel, mf_sel, comb_sel = select_best_two_score_params(
            df_outer_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        fitted = fit_two_scores_plus_clinical(
            df_outer_train,
            spec,
            max_iter,
            convergence_tol,
            ct_score_C=ct_sel.best_params.get("C"),
            ct_score_alpha=ct_sel.best_params.get("alpha"),
            ct_score_l1_ratio=ct_sel.best_params["l1_ratio"],
            mf_score_C=mf_sel.best_params.get("C"),
            mf_score_alpha=mf_sel.best_params.get("alpha"),
            mf_score_l1_ratio=mf_sel.best_params["l1_ratio"],
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_two_scores_plus_clinical(fitted, df_outer_train)
        pred_val = predict_two_scores_plus_clinical(fitted, df_outer_valid)
        thr = _train_threshold(spec, df_outer_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "ct_score_C": ct_sel.best_params.get("C", np.nan),
            "ct_score_l1_ratio": ct_sel.best_params["l1_ratio"],
            "mf_score_C": mf_sel.best_params.get("C", np.nan),
            "mf_score_l1_ratio": mf_sel.best_params["l1_ratio"],
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        tuning_df = pd.concat(
            [
                ct_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage="ct_score_stage",
                ),
                mf_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage="mf_score_stage",
                ),
                comb_sel.all_results.assign(
                    model_name=spec.name,
                    repeat_index=repeat_idx,
                    fold_index=fold_idx,
                    stage="combined_stage",
                ),
            ],
            axis=0,
            ignore_index=True,
        )
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")

    # ---- metrics ----
    if spec.task_kind == "classification":
        m = classification_metrics(df_outer_valid["target"].to_numpy(), pred_val["proba"], thr)
    else:
        m = regression_metrics(
            df_outer_valid["target"].to_numpy(), pred_val["pred"], spec.n_features
        )

    pred_df = _build_pred_df(
        spec,
        df_outer_valid,
        pred_val,
        thr,
        extra_cols={"repeat_index": repeat_idx, "fold_index": fold_idx},
    )

    metric_row = {
        "model_name": spec.name,
        "task_kind": spec.task_kind,
        "repeat_index": repeat_idx,
        "fold_index": fold_idx,
        "n_train": len(df_outer_train),
        "n_valid": len(df_outer_valid),
        **best_params_out,
        **m,
    }
    if spec.task_kind == "classification":
        metric_row["selected_threshold"] = thr

    return metric_row, pred_df, tuning_df


# Outer CV for one model
def run_outer_cv_for_model(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    *,
    n_splits_outer: int,
    n_repeats_outer: int,
    n_splits_inner: int,
    random_state: int,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> Dict:
    outer_splitter = make_outer_splitter(
        spec.task_kind, n_splits_outer, n_repeats_outer, random_state
    )
    inner_splitter = make_inner_splitter(spec.task_kind, n_splits_inner, random_state)

    X_ref = df_train.loc[
        :, list(spec.base_features if spec.base_features else spec.clinical_features)
    ].copy()
    y_ref = df_train["target"].to_numpy()

    metric_rows: List[Dict] = []
    pred_rows: List[pd.DataFrame] = []
    tuning_rows: List[pd.DataFrame] = []

    for split_idx, (tr_idx, val_idx) in enumerate(outer_splitter.split(X_ref, y_ref)):
        rep_idx = split_idx // n_splits_outer
        fold_idx = split_idx % n_splits_outer
        metric_row, pred_df, tuning_df = _run_one_fold(
            df_outer_train=df_train.iloc[tr_idx].reset_index(drop=True),
            df_outer_valid=df_train.iloc[val_idx].reset_index(drop=True),
            spec=spec,
            inner_splitter=inner_splitter,
            l1_ratios=l1_ratios,
            logistic_cs=logistic_cs,
            linear_alphas=linear_alphas,
            max_iter=max_iter,
            convergence_tol=convergence_tol,
            default_cls_threshold=default_cls_threshold,
            repeat_idx=rep_idx,
            fold_idx=fold_idx,
        )
        metric_rows.append(metric_row)
        pred_rows.append(pred_df)
        tuning_rows.append(tuning_df)

    return {
        "cv_fold_metrics": pd.DataFrame(metric_rows),
        "oof_predictions": pd.concat(pred_rows, axis=0, ignore_index=True),
        "inner_tuning": pd.concat(tuning_rows, axis=0, ignore_index=True),
    }


# Refit on full training set and evaluate test set
def refit_and_evaluate_test(
    df: pd.DataFrame,
    spec: ModelSpec,
    *,
    train_split_value: str,
    test_split_value: str,
    n_splits_inner: int,
    random_state: int,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> Dict:
    df_train = df[df["split"] == train_split_value].reset_index(drop=True)
    df_test = df[df["split"] == test_split_value].reset_index(drop=True)
    inner_splitter = make_inner_splitter(spec.task_kind, n_splits_inner, random_state)

    if spec.kind == "direct":
        inner_sel = select_best_direct_params(
            df_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        bp = inner_sel.best_params
        fitted = fit_direct(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            C=bp.get("C"),
            alpha=bp.get("alpha"),
            l1_ratio=bp["l1_ratio"],
        )
        pred_tr = predict_direct(fitted, df_train)
        pred_test = predict_direct(fitted, df_test)
        thr = _train_threshold(spec, df_train, pred_tr, default_cls_threshold)
        best_params_out = dict(bp)
        inner_tuning = inner_sel.all_results.assign(model_name=spec.name, stage="single_stage")

    elif spec.kind == "score_plus_clinical":
        score_sel = select_best_score_stage_params(
            df_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        comb_sel = select_best_combined_stage_params(
            df_train,
            spec,
            score_sel.best_params,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        fitted = fit_score_plus_clinical(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            score_C=score_sel.best_params.get("C"),
            score_alpha=score_sel.best_params.get("alpha"),
            score_l1_ratio=score_sel.best_params["l1_ratio"],
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_score_plus_clinical(fitted, df_train)
        pred_test = predict_score_plus_clinical(fitted, df_test)
        thr = _train_threshold(spec, df_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "score_C": score_sel.best_params.get("C", np.nan),
            "score_alpha": score_sel.best_params.get("alpha", np.nan),
            "score_l1_ratio": score_sel.best_params["l1_ratio"],
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_alpha": comb_sel.best_params.get("alpha", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        inner_tuning = pd.concat(
            [
                score_sel.all_results.assign(model_name=spec.name, stage="score_stage"),
                comb_sel.all_results.assign(model_name=spec.name, stage="combined_stage"),
            ],
            axis=0,
            ignore_index=True,
        )

    elif spec.kind in ("multi_score", "multi_score_plus_clinical"):
        group_sels, comb_sel = select_best_multi_score_params(
            df_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        score_params_by_group = {g: sel.best_params for g, sel in group_sels.items()}
        fitted = fit_multi_score(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            score_params_by_group=score_params_by_group,
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_multi_score(fitted, df_train)
        pred_test = predict_multi_score(fitted, df_test)
        thr = _train_threshold(spec, df_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_alpha": comb_sel.best_params.get("alpha", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        inner_parts = []
        for g_name, g_sel in group_sels.items():
            inner_parts.append(
                g_sel.all_results.assign(model_name=spec.name, stage=f"score_stage__{g_name}")
            )
        inner_parts.append(
            comb_sel.all_results.assign(model_name=spec.name, stage="combined_stage")
        )
        inner_tuning = pd.concat(inner_parts, axis=0, ignore_index=True)

    elif spec.kind == "two_scores_plus_clinical":
        from models_base import fit_two_scores_plus_clinical, predict_two_scores_plus_clinical

        ct_sel, mf_sel, comb_sel = select_best_two_score_params(
            df_train,
            spec,
            inner_splitter,
            l1_ratios,
            logistic_cs,
            linear_alphas,
            max_iter,
            convergence_tol,
            default_cls_threshold,
        )
        fitted = fit_two_scores_plus_clinical(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            ct_score_C=ct_sel.best_params.get("C"),
            ct_score_alpha=ct_sel.best_params.get("alpha"),
            ct_score_l1_ratio=ct_sel.best_params["l1_ratio"],
            mf_score_C=mf_sel.best_params.get("C"),
            mf_score_alpha=mf_sel.best_params.get("alpha"),
            mf_score_l1_ratio=mf_sel.best_params["l1_ratio"],
            combined_C=comb_sel.best_params.get("C"),
            combined_alpha=comb_sel.best_params.get("alpha"),
            combined_l1_ratio=comb_sel.best_params["l1_ratio"],
        )
        pred_tr = predict_two_scores_plus_clinical(fitted, df_train)
        pred_test = predict_two_scores_plus_clinical(fitted, df_test)
        thr = _train_threshold(spec, df_train, pred_tr, default_cls_threshold)
        best_params_out = {
            "ct_score_C": ct_sel.best_params.get("C", np.nan),
            "ct_score_l1_ratio": ct_sel.best_params["l1_ratio"],
            "mf_score_C": mf_sel.best_params.get("C", np.nan),
            "mf_score_l1_ratio": mf_sel.best_params["l1_ratio"],
            "combined_C": comb_sel.best_params.get("C", np.nan),
            "combined_l1_ratio": comb_sel.best_params["l1_ratio"],
        }
        inner_tuning = pd.concat(
            [
                ct_sel.all_results.assign(model_name=spec.name, stage="ct_score_stage"),
                mf_sel.all_results.assign(model_name=spec.name, stage="mf_score_stage"),
                comb_sel.all_results.assign(model_name=spec.name, stage="combined_stage"),
            ],
            axis=0,
            ignore_index=True,
        )
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")

    if spec.task_kind == "classification":
        best_params_out["selected_threshold"] = thr
        fitted["selected_threshold"] = thr
        m = classification_metrics(df_test["target"].to_numpy(), pred_test["proba"], thr)
    else:
        m = regression_metrics(df_test["target"].to_numpy(), pred_test["pred"], spec.n_features)

    test_metrics = pd.DataFrame([{"model_name": spec.name, **best_params_out, **m}])
    coef_df = extract_coefficients_from_fitted(fitted)
    pred_df = _build_pred_df(spec, df_test, pred_test, thr)

    return {
        "best_params": best_params_out,
        "test_metrics": test_metrics,
        "test_predictions": pred_df,
        "coefficients": coef_df,
        "inner_tuning": inner_tuning,
        "fitted": fitted,
    }


# Aggregation helpers
def _safe_nanmean(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if v.size > 0 and not np.all(np.isnan(v)) else np.nan


def _safe_nanstd(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    finite = v[~np.isnan(v)]
    return float(np.nanstd(v, ddof=1)) if finite.size > 1 else np.nan


def summarize_cv_metrics(df_fold: pd.DataFrame, task_kind: str) -> pd.DataFrame:
    if df_fold.empty:
        return pd.DataFrame()
    metric_cols = (
        [
            "auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "brier",
            "selected_threshold",
        ]
        if task_kind == "classification"
        else ["r2", "adjusted_r2", "mae", "rmse", "spearman_r"]
    )
    rows: List[Dict] = []
    for model_name, g in df_fold.groupby("model_name"):
        row: Dict = {"model_name": model_name}
        for col in metric_cols:
            if col in g.columns:
                vals = g[col].to_numpy(dtype=float)
                row[f"{col}_mean"] = _safe_nanmean(vals)
                row[f"{col}_sd"] = _safe_nanstd(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def aggregate_oof_predictions(
    df_pred: pd.DataFrame,
    task_kind: str,
    default_threshold: float = 0.5,
) -> pd.DataFrame:
    if df_pred.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "patient_id", "target"]

    if task_kind == "classification":
        agg_dict = dict(
            pred_score_mean=("pred_score", "mean"),
            pred_score_sd=("pred_score", "std"),
            pred_proba_mean=("pred_proba", "mean"),
            pred_proba_sd=("pred_proba", "std"),
            n_predictions=("pred_proba", "size"),
        )
        if "selected_threshold" in df_pred.columns:
            agg_dict["selected_threshold_mean"] = ("selected_threshold", "mean")
        agg = df_pred.groupby(group_cols, as_index=False).agg(**agg_dict)
        agg["pred_proba"] = agg["pred_proba_mean"]
        agg["pred_score"] = agg["pred_score_mean"]
        thr_col = "selected_threshold_mean" if "selected_threshold_mean" in agg.columns else None
        agg["selected_threshold"] = agg[thr_col] if thr_col else default_threshold
        agg["pred_label"] = (agg["pred_proba"] >= agg["selected_threshold"]).astype(int)
        return agg

    agg = df_pred.groupby(group_cols, as_index=False).agg(
        pred_score_mean=("pred_score", "mean"),
        pred_score_sd=("pred_score", "std"),
        pred_value_mean=("pred_value", "mean"),
        pred_value_sd=("pred_value", "std"),
        n_predictions=("pred_value", "size"),
    )
    agg["pred_value"] = agg["pred_value_mean"]
    agg["pred_score"] = agg["pred_score_mean"]
    return agg


# Top-level entry point
def run_cv_for_specs(
    df: pd.DataFrame,
    specs: Sequence[ModelSpec],
    *,
    train_split_value: str = "train",
    test_split_value: str = "test",
    n_splits_outer: int = 5,
    n_repeats_outer: int = 5,
    n_splits_inner: int = 5,
    random_state: int = 42,
    l1_ratios: Sequence[float] = (0.0, 0.5, 1.0),
    logistic_cs: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
    linear_alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
    max_iter: int = 20000,
    convergence_tol: float = 1e-4,
    default_cls_threshold: float = 0.5,
) -> Dict:
    if "split" not in df.columns:
        raise KeyError("Input dataframe must contain 'split' column.")
    df_train = df[df["split"] == train_split_value].reset_index(drop=True)

    cv_metrics: List[pd.DataFrame] = []
    oof_preds: List[pd.DataFrame] = []
    inner_tunings: List[pd.DataFrame] = []
    test_metrics: List[pd.DataFrame] = []
    test_preds: List[pd.DataFrame] = []
    coef_dfs: List[pd.DataFrame] = []
    refit_tunings: List[pd.DataFrame] = []
    model_info_rows: List[Dict] = []
    error_rows: List[Dict] = []
    refit_models: Dict = {}

    for spec in tqdm(specs, desc="CV"):
        model_info_rows.append(
            {
                "model_name": spec.name,
                "task_kind": spec.task_kind,
                "model_kind": spec.kind,
                "n_features": spec.n_features,
            }
        )
        try:
            cv_out = run_outer_cv_for_model(
                df_train,
                spec,
                n_splits_outer=n_splits_outer,
                n_repeats_outer=n_repeats_outer,
                n_splits_inner=n_splits_inner,
                random_state=random_state,
                l1_ratios=l1_ratios,
                logistic_cs=logistic_cs,
                linear_alphas=linear_alphas,
                max_iter=max_iter,
                convergence_tol=convergence_tol,
                default_cls_threshold=default_cls_threshold,
            )
            cv_metrics.append(cv_out["cv_fold_metrics"])
            oof_preds.append(cv_out["oof_predictions"])
            inner_tunings.append(cv_out["inner_tuning"])

            test_out = refit_and_evaluate_test(
                df,
                spec,
                train_split_value=train_split_value,
                test_split_value=test_split_value,
                n_splits_inner=n_splits_inner,
                random_state=random_state,
                l1_ratios=l1_ratios,
                logistic_cs=logistic_cs,
                linear_alphas=linear_alphas,
                max_iter=max_iter,
                convergence_tol=convergence_tol,
                default_cls_threshold=default_cls_threshold,
            )
            test_metrics.append(test_out["test_metrics"])
            test_preds.append(test_out["test_predictions"])
            coef_dfs.append(test_out["coefficients"])
            refit_tunings.append(test_out["inner_tuning"])
            refit_models[spec.name] = test_out["fitted"]
        except Exception as exc:
            error_rows.append(
                {
                    "model_name": spec.name,
                    "task_kind": spec.task_kind,
                    "model_kind": spec.kind,
                    "error": repr(exc),
                }
            )

    def _c(tables):
        return pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()

    task_kind = specs[0].task_kind if specs else "classification"
    cv_fold_metrics = _c(cv_metrics)
    oof_predictions = _c(oof_preds)
    oof_predictions_agg = aggregate_oof_predictions(
        oof_predictions, task_kind, default_cls_threshold
    )

    return {
        "cv_fold_metrics": cv_fold_metrics,
        "cv_summary": summarize_cv_metrics(cv_fold_metrics, task_kind),
        "oof_predictions": oof_predictions,
        "oof_predictions_aggregated": oof_predictions_agg,
        "inner_tuning_cv": _c(inner_tunings),
        "test_metrics": _c(test_metrics),
        "test_predictions": _c(test_preds),
        "coefficients": _c(coef_dfs),
        "inner_tuning_refit": _c(refit_tunings),
        "model_info": pd.DataFrame(model_info_rows),
        "errors": pd.DataFrame(error_rows),
        "refit_models": refit_models,
    }


# Two-scores-plus-clinical inner CV selection and fold helpers
# (used by 06_03_deep_radiomics)
def select_best_two_score_params(
    df_outer_train: pd.DataFrame,
    spec: ModelSpec,
    inner_splitter,
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
    max_iter: int,
    convergence_tol: float,
    default_cls_threshold: float,
) -> Tuple[InnerSelectionResult, InnerSelectionResult, InnerSelectionResult]:
    """Select params for CT stage, MF stage, and combined stage independently."""
    from models_base import ModelSpec as _MS

    # MuscleFat features are stored in score_groups[0][1]
    mf_features = spec.score_groups[0][1] if spec.score_groups else spec.base_features

    ct_spec = _MS(
        name=f"{spec.name}__ct",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=spec.base_features,
        clinical_features=spec.clinical_features,
    )
    mf_spec = _MS(
        name=f"{spec.name}__mf",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=mf_features,
        clinical_features=spec.clinical_features,
    )

    ct_sel = select_best_direct_params(
        df_outer_train,
        ct_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )
    mf_sel = select_best_direct_params(
        df_outer_train,
        mf_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )

    # Build OOF scores for combined stage
    y = df_outer_train["target"].to_numpy()
    X_ct = df_outer_train.loc[:, list(spec.base_features)].copy()
    X_mf = df_outer_train.loc[:, list(mf_features)].copy()
    oof_ct = np.full(len(df_outer_train), np.nan, dtype=float)
    oof_mf = np.full(len(df_outer_train), np.nan, dtype=float)

    for tr_idx, val_idx in inner_splitter.split(X_ct, y):
        fitted_ct = fit_direct(
            df_outer_train.iloc[tr_idx].reset_index(drop=True),
            ct_spec,
            max_iter,
            convergence_tol,
            C=ct_sel.best_params.get("C"),
            alpha=ct_sel.best_params.get("alpha"),
            l1_ratio=ct_sel.best_params["l1_ratio"],
        )
        pred_ct = predict_direct(fitted_ct, df_outer_train.iloc[val_idx].reset_index(drop=True))
        oof_ct[val_idx] = pred_ct["score"]

        fitted_mf = fit_direct(
            df_outer_train.iloc[tr_idx].reset_index(drop=True),
            mf_spec,
            max_iter,
            convergence_tol,
            C=mf_sel.best_params.get("C"),
            alpha=mf_sel.best_params.get("alpha"),
            l1_ratio=mf_sel.best_params["l1_ratio"],
        )
        pred_mf = predict_direct(fitted_mf, df_outer_train.iloc[val_idx].reset_index(drop=True))
        oof_mf[val_idx] = pred_mf["score"]

    df_s2 = _add_score_columns(df_outer_train, {"score_ct": oof_ct, "score_musclefat": oof_mf})
    combined_spec = _MS(
        name=f"{spec.name}__combined",
        kind="direct",
        task_kind=spec.task_kind,
        base_features=spec.all_features,
        clinical_features=spec.clinical_features,
    )
    comb_sel = select_best_direct_params(
        df_s2,
        combined_spec,
        inner_splitter,
        l1_ratios,
        logistic_cs,
        linear_alphas,
        max_iter,
        convergence_tol,
        default_cls_threshold,
    )
    return ct_sel, mf_sel, comb_sel
