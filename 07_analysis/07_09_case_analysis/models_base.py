from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Type aliases
ModelKind = Literal[
    "direct",
    "score_plus_clinical",
    "multi_score",
    "multi_score_plus_clinical",
    "two_scores_plus_clinical",
]

ScoreGroup = Tuple[str, Tuple[str, ...]]  # (group_name, feature_columns)


# ModelSpec
@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: ModelKind
    task_kind: Literal["classification", "regression"]
    base_features: Tuple[str, ...]
    clinical_features: Tuple[str, ...]
    use_score: bool = False
    score_groups: Tuple[ScoreGroup, ...] = field(default_factory=tuple)

    @property
    def all_features(self) -> Tuple[str, ...]:
        if self.kind == "direct":
            return self.base_features
        if self.kind == "score_plus_clinical":
            return ("score",) + self.clinical_features
        if self.kind == "multi_score":
            return tuple(_score_col(name) for name, _ in self.score_groups)
        if self.kind == "multi_score_plus_clinical":
            return tuple(_score_col(name) for name, _ in self.score_groups) + self.clinical_features
        if self.kind == "two_scores_plus_clinical":
            return ("score_ct", "score_musclefat") + self.clinical_features
        raise ValueError(f"Unknown model kind: {self.kind}")

    @property
    def n_features(self) -> int:
        if self.kind == "direct":
            return len(self.base_features)
        if self.kind == "score_plus_clinical":
            return 1 + len(self.clinical_features)
        if self.kind == "multi_score":
            return len(self.score_groups)
        if self.kind == "multi_score_plus_clinical":
            return len(self.score_groups) + len(self.clinical_features)
        if self.kind == "two_scores_plus_clinical":
            return 2 + len(self.clinical_features)
        raise ValueError(f"Unknown model kind: {self.kind}")


def _score_col(group_name: str) -> str:
    return f"score__{group_name}"


# Feature-group resolution (used by spectral radiomics pipelines)
# Sentinel prefix: base_features = ("__GROUP__:key",) means resolve at runtime
_GROUP_SENTINEL = "__GROUP__:"


def resolve_model_specs(
    specs: Sequence[ModelSpec],
    group_lookup: Dict[str, Tuple[str, ...]],
) -> List[ModelSpec]:
    """Replace __GROUP__ sentinels and multi_score group references with actual columns."""
    resolved: List[ModelSpec] = []
    for spec in specs:
        if spec.kind in ("multi_score", "multi_score_plus_clinical"):
            groups: List[ScoreGroup] = []
            for group_name, _ in spec.score_groups:
                if group_name not in group_lookup:
                    raise ValueError(f"Could not resolve feature group: {group_name}")
                groups.append((group_name, group_lookup[group_name]))
            resolved.append(
                ModelSpec(
                    name=spec.name,
                    kind=spec.kind,
                    task_kind=spec.task_kind,
                    base_features=tuple(),
                    clinical_features=spec.clinical_features,
                    use_score=spec.use_score,
                    score_groups=tuple(groups),
                )
            )
            continue

        if len(spec.base_features) == 1 and spec.base_features[0].startswith(_GROUP_SENTINEL):
            key = spec.base_features[0][len(_GROUP_SENTINEL) :]
            if key not in group_lookup:
                raise ValueError(f"Could not resolve feature group: {key}")
            resolved.append(
                ModelSpec(
                    name=spec.name,
                    kind=spec.kind,
                    task_kind=spec.task_kind,
                    base_features=group_lookup[key],
                    clinical_features=spec.clinical_features,
                    use_score=spec.use_score,
                    score_groups=spec.score_groups,
                )
            )
        else:
            resolved.append(spec)
    return resolved


# Pipeline factories
def _logistic_pipeline(C: float, l1_ratio: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    class_weight="balanced",
                    C=float(C),
                    l1_ratio=float(l1_ratio),
                    max_iter=int(max_iter),
                    random_state=42,
                ),
            ),
        ]
    )


def _linear_pipeline(alpha: float, l1_ratio: float, max_iter: int, tol: float) -> Pipeline:
    if float(l1_ratio) == 0.0:
        model = Ridge(alpha=float(alpha), tol=float(tol), random_state=42)
    else:
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=42,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_estimator(
    task_kind: Literal["classification", "regression"],
    max_iter: int,
    convergence_tol: float,
    *,
    C: Optional[float] = None,
    alpha: Optional[float] = None,
    l1_ratio: float = 0.5,
) -> Pipeline:
    if task_kind == "classification":
        if C is None:
            raise ValueError("C must be provided for classification.")
        return _logistic_pipeline(C=C, l1_ratio=l1_ratio, max_iter=max_iter)
    if task_kind == "regression":
        if alpha is None:
            raise ValueError("alpha must be provided for regression.")
        return _linear_pipeline(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=convergence_tol
        )
    raise ValueError(f"Unknown task_kind: {task_kind}")


# Hyper-parameter search grid
def get_search_grid(
    task_kind: Literal["classification", "regression"],
    l1_ratios: Sequence[float],
    logistic_cs: Sequence[float],
    linear_alphas: Sequence[float],
) -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []
    if task_kind == "classification":
        for l1 in l1_ratios:
            for C in logistic_cs:
                grid.append({"l1_ratio": float(l1), "C": float(C)})
    elif task_kind == "regression":
        for l1 in l1_ratios:
            for alpha in linear_alphas:
                grid.append({"l1_ratio": float(l1), "alpha": float(alpha)})
    else:
        raise ValueError(f"Unknown task_kind: {task_kind}")
    return grid


# Data helpers
def get_X_y(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "target",
) -> Tuple[pd.DataFrame, np.ndarray]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    return df.loc[:, list(feature_cols)].copy(), df[target_col].to_numpy()


def split_train_test(
    df: pd.DataFrame,
    train_split_value: str = "train",
    test_split_value: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in df.columns:
        raise KeyError("DataFrame must contain 'split' column.")
    return (
        df[df["split"] == train_split_value].reset_index(drop=True),
        df[df["split"] == test_split_value].reset_index(drop=True),
    )


def _add_score_columns(df: pd.DataFrame, score_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    for col, values in score_map.items():
        out[col] = np.asarray(values, dtype=float)
    return out


# Decision score / probability extraction
def predict_score(
    estimator: Pipeline,
    X: pd.DataFrame,
    task_kind: Literal["classification", "regression"],
) -> np.ndarray:
    if task_kind == "classification":
        model = estimator.named_steps["model"]
        if hasattr(model, "decision_function"):
            Xt = estimator[:-1].transform(X)
            return model.decision_function(Xt)
        raise AttributeError("Classification estimator has no decision_function.")
    return estimator.predict(X)


# Direct model
def fit_direct(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    max_iter: int,
    convergence_tol: float,
    *,
    C: Optional[float] = None,
    alpha: Optional[float] = None,
    l1_ratio: float = 0.5,
) -> Dict:
    X, y = get_X_y(df_train, spec.base_features)
    est = build_estimator(
        spec.task_kind, max_iter, convergence_tol, C=C, alpha=alpha, l1_ratio=l1_ratio
    )
    est.fit(X, y)
    return {
        "spec": spec,
        "estimator": est,
        "features": spec.base_features,
        "task_kind": spec.task_kind,
        "params": {"C": C, "alpha": alpha, "l1_ratio": l1_ratio},
    }


def predict_direct(fitted: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    spec: ModelSpec = fitted["spec"]
    est: Pipeline = fitted["estimator"]
    X, _ = get_X_y(df, spec.base_features)
    out: Dict[str, np.ndarray] = {"score": predict_score(est, X, spec.task_kind)}
    if spec.task_kind == "classification":
        out["proba"] = est.predict_proba(X)[:, 1]
        out["pred"] = est.predict(X)
    else:
        out["pred"] = est.predict(X)
    return out


# Score-plus-clinical model
def fit_score_plus_clinical(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    max_iter: int,
    convergence_tol: float,
    *,
    score_C: Optional[float] = None,
    score_alpha: Optional[float] = None,
    score_l1_ratio: float = 0.5,
    combined_C: Optional[float] = None,
    combined_alpha: Optional[float] = None,
    combined_l1_ratio: float = 0.5,
) -> Dict:
    X_score, y = get_X_y(df_train, spec.base_features)
    score_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=score_C,
        alpha=score_alpha,
        l1_ratio=score_l1_ratio,
    )
    score_est.fit(X_score, y)
    train_score = predict_score(score_est, X_score, spec.task_kind)

    df_s2 = _add_score_columns(df_train, {"score": train_score})
    X_comb, _ = get_X_y(df_s2, spec.all_features)
    comb_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=combined_C,
        alpha=combined_alpha,
        l1_ratio=combined_l1_ratio,
    )
    comb_est.fit(X_comb, y)
    return {
        "spec": spec,
        "score_estimator": score_est,
        "combined_estimator": comb_est,
        "score_features": spec.base_features,
        "combined_features": spec.all_features,
        "task_kind": spec.task_kind,
        "params": {
            "score_C": score_C,
            "score_alpha": score_alpha,
            "score_l1_ratio": score_l1_ratio,
            "combined_C": combined_C,
            "combined_alpha": combined_alpha,
            "combined_l1_ratio": combined_l1_ratio,
        },
    }


def predict_score_plus_clinical(fitted: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    spec: ModelSpec = fitted["spec"]
    score_est: Pipeline = fitted["score_estimator"]
    comb_est: Pipeline = fitted["combined_estimator"]

    X_score, _ = get_X_y(df, fitted["score_features"])
    score = predict_score(score_est, X_score, spec.task_kind)
    df_s2 = _add_score_columns(df, {"score": score})
    X_comb, _ = get_X_y(df_s2, fitted["combined_features"])

    out: Dict[str, np.ndarray] = {
        "score": predict_score(comb_est, X_comb, spec.task_kind),
        "stage1_score": score,
    }
    if spec.task_kind == "classification":
        out["proba"] = comb_est.predict_proba(X_comb)[:, 1]
        out["pred"] = comb_est.predict(X_comb)
    else:
        out["pred"] = comb_est.predict(X_comb)
    return out


# Multi-score model (one estimator per feature group, combined in stage 2)
def fit_multi_score(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    max_iter: int,
    convergence_tol: float,
    *,
    score_params_by_group: Dict[str, Dict[str, float]],
    combined_C: Optional[float] = None,
    combined_alpha: Optional[float] = None,
    combined_l1_ratio: float = 0.5,
) -> Dict:
    group_estimators: Dict[str, Pipeline] = {}
    train_scores: Dict[str, np.ndarray] = {}
    y = df_train["target"].to_numpy()

    for group_name, feature_cols in spec.score_groups:
        X_group, _ = get_X_y(df_train, feature_cols)
        params = score_params_by_group[group_name]
        est = build_estimator(
            spec.task_kind,
            max_iter,
            convergence_tol,
            C=params.get("C"),
            alpha=params.get("alpha"),
            l1_ratio=params["l1_ratio"],
        )
        est.fit(X_group, y)
        group_estimators[group_name] = est
        train_scores[_score_col(group_name)] = predict_score(est, X_group, spec.task_kind)

    df_s2 = _add_score_columns(df_train, train_scores)
    X_comb, _ = get_X_y(df_s2, spec.all_features)
    comb_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=combined_C,
        alpha=combined_alpha,
        l1_ratio=combined_l1_ratio,
    )
    comb_est.fit(X_comb, y)

    return {
        "spec": spec,
        "group_estimators": group_estimators,
        "group_features": {name: cols for name, cols in spec.score_groups},
        "combined_estimator": comb_est,
        "combined_features": spec.all_features,
        "task_kind": spec.task_kind,
        "params": {
            "score_params_by_group": score_params_by_group,
            "combined_C": combined_C,
            "combined_alpha": combined_alpha,
            "combined_l1_ratio": combined_l1_ratio,
        },
    }


def predict_multi_score(fitted: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    spec: ModelSpec = fitted["spec"]
    score_cols: Dict[str, np.ndarray] = {}
    for group_name, feature_cols in fitted["group_features"].items():
        est: Pipeline = fitted["group_estimators"][group_name]
        X, _ = get_X_y(df, feature_cols)
        score_cols[_score_col(group_name)] = predict_score(est, X, spec.task_kind)

    df_s2 = _add_score_columns(df, score_cols)
    X_comb, _ = get_X_y(df_s2, fitted["combined_features"])
    comb_est: Pipeline = fitted["combined_estimator"]

    out: Dict[str, np.ndarray] = {
        "score": predict_score(comb_est, X_comb, spec.task_kind),
        **score_cols,
    }
    if spec.task_kind == "classification":
        out["proba"] = comb_est.predict_proba(X_comb)[:, 1]
        out["pred"] = comb_est.predict(X_comb)
    else:
        out["pred"] = comb_est.predict(X_comb)
    return out


# Generic dispatch helpers (used by cv.py)
def fit_model(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    max_iter: int,
    convergence_tol: float,
    *,
    params: Dict,
) -> Dict:
    """Dispatch fit to the correct function based on spec.kind."""
    if spec.kind == "direct":
        return fit_direct(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            C=params.get("C"),
            alpha=params.get("alpha"),
            l1_ratio=params.get("l1_ratio", 0.5),
        )
    if spec.kind == "score_plus_clinical":
        return fit_score_plus_clinical(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            score_C=params.get("score_C"),
            score_alpha=params.get("score_alpha"),
            score_l1_ratio=params.get("score_l1_ratio", 0.5),
            combined_C=params.get("combined_C"),
            combined_alpha=params.get("combined_alpha"),
            combined_l1_ratio=params.get("combined_l1_ratio", 0.5),
        )
    if spec.kind in ("multi_score", "multi_score_plus_clinical"):
        return fit_multi_score(
            df_train,
            spec,
            max_iter,
            convergence_tol,
            score_params_by_group=params["score_params_by_group"],
            combined_C=params.get("combined_C"),
            combined_alpha=params.get("combined_alpha"),
            combined_l1_ratio=params.get("combined_l1_ratio", 0.5),
        )
    raise ValueError(f"Unknown model kind: {spec.kind}")


def predict_model(fitted: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Dispatch predict to the correct function based on spec.kind."""
    spec: ModelSpec = fitted["spec"]
    if spec.kind == "direct":
        return predict_direct(fitted, df)
    if spec.kind == "score_plus_clinical":
        return predict_score_plus_clinical(fitted, df)
    if spec.kind in ("multi_score", "multi_score_plus_clinical"):
        return predict_multi_score(fitted, df)
    if spec.kind == "two_scores_plus_clinical":
        # imported at call time to avoid forward reference at module level
        from models_base import predict_two_scores_plus_clinical

        return predict_two_scores_plus_clinical(fitted, df)
    raise ValueError(f"Unknown model kind: {spec.kind}")


# Coefficient extraction
def extract_coefficients(estimator: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    model = estimator.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise AttributeError("Model does not expose coef_.")
    coef = np.asarray(model.coef_).reshape(-1)
    if len(coef) != len(feature_names):
        raise ValueError(f"Coefficient length mismatch: {len(coef)} vs {len(feature_names)}")
    intercept = float(np.asarray(getattr(model, "intercept_", [np.nan])).reshape(-1)[0])
    return pd.DataFrame(
        {"feature": list(feature_names), "coefficient": coef.astype(float), "intercept": intercept}
    )


def extract_coefficients_from_fitted(fitted: Dict) -> pd.DataFrame:
    """Extract a single merged coefficients table from any fitted model dict."""
    spec: ModelSpec = fitted["spec"]
    rows: List[pd.DataFrame] = []

    if spec.kind == "direct":
        df = extract_coefficients(fitted["estimator"], fitted["features"])
        df["stage"] = "single_stage"
        rows.append(df)

    elif spec.kind == "score_plus_clinical":
        s = extract_coefficients(fitted["score_estimator"], fitted["score_features"])
        s["stage"] = "score_stage"
        c = extract_coefficients(fitted["combined_estimator"], fitted["combined_features"])
        c["stage"] = "combined_stage"
        rows.extend([s, c])

    elif spec.kind in ("multi_score", "multi_score_plus_clinical"):
        for group_name, est in fitted["group_estimators"].items():
            s = extract_coefficients(est, fitted["group_features"][group_name])
            s["stage"] = f"score_stage__{group_name}"
            rows.append(s)
        c = extract_coefficients(fitted["combined_estimator"], fitted["combined_features"])
        c["stage"] = "combined_stage"
        rows.append(c)

    elif spec.kind == "two_scores_plus_clinical":
        ct = extract_coefficients(fitted["ct_score_estimator"], fitted["ct_score_features"])
        ct["stage"] = "ct_score_stage"
        rows.append(ct)
        mf = extract_coefficients(fitted["mf_score_estimator"], fitted["mf_score_features"])
        mf["stage"] = "mf_score_stage"
        rows.append(mf)
        c = extract_coefficients(fitted["combined_estimator"], fitted["combined_features"])
        c["stage"] = "combined_stage"
        rows.append(c)

    result = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    result["model_name"] = spec.name
    return result


# Two-scores-plus-clinical model (deep radiomics CT + MuscleFat)
# This is a specialised multi_score variant with exactly two named score groups
# (score_ct, score_musclefat) that is kept separate from the generic multi_score
# because it has dedicated named fields (ct_score_features, mf_score_features)
# and corresponding named output keys that external code depends on.
def fit_two_scores_plus_clinical(
    df_train: pd.DataFrame,
    spec: ModelSpec,
    max_iter: int,
    convergence_tol: float,
    *,
    ct_score_C: Optional[float] = None,
    ct_score_alpha: Optional[float] = None,
    ct_score_l1_ratio: float = 0.5,
    mf_score_C: Optional[float] = None,
    mf_score_alpha: Optional[float] = None,
    mf_score_l1_ratio: float = 0.5,
    combined_C: Optional[float] = None,
    combined_alpha: Optional[float] = None,
    combined_l1_ratio: float = 0.5,
) -> Dict:
    """Fit CT score + MuscleFat score + clinical combined model."""
    if spec.kind != "two_scores_plus_clinical":
        raise ValueError(f"Expected two_scores_plus_clinical spec, got: {spec.kind}")

    y = df_train["target"].to_numpy()

    # Stage 1a: CT estimator
    X_ct, _ = get_X_y(df_train, spec.base_features)
    ct_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=ct_score_C,
        alpha=ct_score_alpha,
        l1_ratio=ct_score_l1_ratio,
    )
    ct_est.fit(X_ct, y)
    score_ct = predict_score(ct_est, X_ct, spec.task_kind)

    # Stage 1b: MuscleFat estimator
    # aux_features stored in score_groups for compatibility
    mf_features = spec.score_groups[0][1] if spec.score_groups else spec.clinical_features
    X_mf, _ = get_X_y(df_train, mf_features)
    mf_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=mf_score_C,
        alpha=mf_score_alpha,
        l1_ratio=mf_score_l1_ratio,
    )
    mf_est.fit(X_mf, y)
    score_mf = predict_score(mf_est, X_mf, spec.task_kind)

    # Stage 2: combined (score_ct, score_musclefat, clinical)
    df_s2 = _add_score_columns(df_train, {"score_ct": score_ct, "score_musclefat": score_mf})
    X_comb, _ = get_X_y(df_s2, spec.all_features)
    comb_est = build_estimator(
        spec.task_kind,
        max_iter,
        convergence_tol,
        C=combined_C,
        alpha=combined_alpha,
        l1_ratio=combined_l1_ratio,
    )
    comb_est.fit(X_comb, y)

    return {
        "spec": spec,
        "ct_score_estimator": ct_est,
        "mf_score_estimator": mf_est,
        "combined_estimator": comb_est,
        "ct_score_features": spec.base_features,
        "mf_score_features": mf_features,
        "combined_features": spec.all_features,
        "task_kind": spec.task_kind,
        "params": {
            "ct_score_C": ct_score_C,
            "ct_score_alpha": ct_score_alpha,
            "ct_score_l1_ratio": ct_score_l1_ratio,
            "mf_score_C": mf_score_C,
            "mf_score_alpha": mf_score_alpha,
            "mf_score_l1_ratio": mf_score_l1_ratio,
            "combined_C": combined_C,
            "combined_alpha": combined_alpha,
            "combined_l1_ratio": combined_l1_ratio,
        },
    }


def predict_two_scores_plus_clinical(fitted: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    spec: ModelSpec = fitted["spec"]
    ct_est: Pipeline = fitted["ct_score_estimator"]
    mf_est: Pipeline = fitted["mf_score_estimator"]
    comb_est: Pipeline = fitted["combined_estimator"]

    X_ct, _ = get_X_y(df, fitted["ct_score_features"])
    score_ct = predict_score(ct_est, X_ct, spec.task_kind)

    X_mf, _ = get_X_y(df, fitted["mf_score_features"])
    score_mf = predict_score(mf_est, X_mf, spec.task_kind)

    df_s2 = _add_score_columns(df, {"score_ct": score_ct, "score_musclefat": score_mf})
    X_comb, _ = get_X_y(df_s2, fitted["combined_features"])

    out: Dict[str, np.ndarray] = {
        "score_ct": score_ct,
        "score_musclefat": score_mf,
        "score": predict_score(comb_est, X_comb, spec.task_kind),
    }
    if spec.task_kind == "classification":
        out["proba"] = comb_est.predict_proba(X_comb)[:, 1]
        out["pred"] = comb_est.predict(X_comb)
    else:
        out["pred"] = comb_est.predict(X_comb)
    return out
