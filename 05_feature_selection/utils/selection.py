from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SpectralSignatureConfig


@dataclass
class PreparedSelectionData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols_before: List[str]
    feature_cols_after: List[str]
    dropped_all_missing: List[str]
    dropped_low_coverage: List[str]
    dropped_non_numeric: List[str]
    dropped_near_constant: List[str]
    X_train: pd.DataFrame
    y_train: pd.Series


def _coerce_numeric(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    dropped = []
    for col in feature_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == 0:
            dropped.append(col)
        else:
            df[col] = converted
    return df, dropped


def prepare_data_for_selection(
    full_df: pd.DataFrame,
    cfg: SpectralSignatureConfig,
) -> PreparedSelectionData:
    train_df = full_df[full_df[cfg.split_col] == "train"].copy()
    test_df = full_df[full_df[cfg.split_col] == "test"].copy()
    if train_df.empty:
        raise ValueError("No training rows available.")
    if train_df["task_target"].isna().any():
        raise ValueError(f"Training target contains {int(train_df['task_target'].isna().sum())} missing values.")

    excluded = {cfg.patient_id_col, cfg.cohort_col, cfg.split_col, cfg.temporal_flag_col, "task_target"}
    feature_cols_before = [c for c in full_df.columns if c not in excluded]

    full_df_num, dropped_non_numeric = _coerce_numeric(full_df, feature_cols_before)
    train_df = full_df_num[full_df_num[cfg.split_col] == "train"].copy()
    test_df = full_df_num[full_df_num[cfg.split_col] == "test"].copy()

    candidate_cols = [c for c in feature_cols_before if c not in set(dropped_non_numeric)]
    dropped_all_missing = [c for c in candidate_cols if train_df[c].notna().sum() == 0]
    candidate_cols = [c for c in candidate_cols if c not in set(dropped_all_missing)]

    dropped_low_coverage = [
        c for c in candidate_cols
        if train_df[c].notna().mean() < cfg.min_non_missing_fraction
    ]
    candidate_cols = [c for c in candidate_cols if c not in set(dropped_low_coverage)]

    dropped_near_constant = []
    for col in candidate_cols:
        vals = train_df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0 or np.nanstd(vals) <= cfg.near_constant_threshold:
            dropped_near_constant.append(col)

    feature_cols_after = [c for c in candidate_cols if c not in set(dropped_near_constant)]
    if not feature_cols_after:
        raise ValueError("No usable features remain after preprocessing.")

    medians = train_df[feature_cols_after].median()
    train_df[feature_cols_after] = train_df[feature_cols_after].fillna(medians)
    if not test_df.empty:
        test_df[feature_cols_after] = test_df[feature_cols_after].fillna(medians)

    X_train = train_df[feature_cols_after].copy()
    y_train = train_df["task_target"].astype(int).copy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training target has fewer than two classes.")

    return PreparedSelectionData(
        train_df=train_df,
        test_df=test_df,
        feature_cols_before=feature_cols_before,
        feature_cols_after=feature_cols_after,
        dropped_all_missing=dropped_all_missing,
        dropped_low_coverage=dropped_low_coverage,
        dropped_non_numeric=dropped_non_numeric,
        dropped_near_constant=dropped_near_constant,
        X_train=X_train,
        y_train=y_train,
    )


def build_l1_logistic_pipeline(cfg: SpectralSignatureConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    class_weight="balanced",
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def tune_l1_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: SpectralSignatureConfig,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    search = GridSearchCV(
        estimator=build_l1_logistic_pipeline(cfg),
        param_grid={"model__C": list(cfg.c_grid)},
        scoring=cfg.scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def extract_selected_coefficients(search: GridSearchCV, feature_cols: List[str]) -> pd.DataFrame:
    coefs = search.best_estimator_.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["selected"] = coef_df["coefficient"] != 0
    return coef_df.sort_values(["selected", "abs_coefficient"], ascending=[False, False]).reset_index(drop=True)


def compute_coefficient_paths(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    cfg: SpectralSignatureConfig,
) -> pd.DataFrame:
    rows = []
    base = build_l1_logistic_pipeline(cfg)
    for c_val in cfg.c_grid:
        est = clone(base)
        est.set_params(model__C=float(c_val))
        est.fit(X_train, y_train)
        for feature, coef in zip(feature_cols, est.named_steps["model"].coef_.ravel()):
            rows.append({"C": float(c_val), "feature": feature, "coefficient": float(coef)})
    return pd.DataFrame(rows)


def _draw_bootstrap_indices(y_train: pd.Series, rng: np.random.RandomState, stratified: bool) -> np.ndarray:
    y = pd.Series(y_train).reset_index(drop=True)
    n = len(y)
    if not stratified:
        return rng.choice(np.arange(n), size=n, replace=True)

    sampled = []
    for cls, n_cls in y.value_counts(sort=False).to_dict().items():
        cls_idx = np.flatnonzero(y.to_numpy() == cls)
        sampled.append(rng.choice(cls_idx, size=int(n_cls), replace=True))
    idx = np.concatenate(sampled)
    rng.shuffle(idx)
    return idx


def _selection_frequency_bootstrap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    best_c: float,
    cfg: SpectralSignatureConfig,
) -> pd.DataFrame:
    rng = np.random.RandomState(cfg.random_state)
    base = build_l1_logistic_pipeline(cfg)
    base.set_params(model__C=float(best_c))

    selected_counts = np.zeros(len(feature_cols), dtype=int)
    coef_sums = np.zeros(len(feature_cols), dtype=float)
    abs_coef_sums = np.zeros(len(feature_cols), dtype=float)
    total_fits = 0

    for _ in range(int(cfg.selection_frequency_repeats)):
        train_idx = _draw_bootstrap_indices(y_train, rng, bool(cfg.bootstrap_stratified))
        X_boot = X_train.iloc[train_idx]
        y_boot = y_train.iloc[train_idx]
        if len(np.unique(y_boot)) < 2:
            continue
        est = clone(base)
        est.fit(X_boot, y_boot)
        coef = est.named_steps["model"].coef_.ravel()
        selected_counts += (coef != 0).astype(int)
        coef_sums += coef
        abs_coef_sums += np.abs(coef)
        total_fits += 1

    if total_fits == 0:
        raise RuntimeError("No bootstrap fits were completed.")

    return pd.DataFrame({
        "feature": feature_cols,
        "selection_count": selected_counts,
        "n_repeats": total_fits,
        "selection_frequency": selected_counts / total_fits,
        "mean_coefficient": coef_sums / total_fits,
        "abs_mean_coefficient": abs_coef_sums / total_fits,
        "stability_method": "bootstrap",
    }).sort_values(["selection_frequency", "abs_mean_coefficient"], ascending=[False, False]).reset_index(drop=True)


def _selection_frequency_repeated_kfold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    best_c: float,
    cfg: SpectralSignatureConfig,
) -> pd.DataFrame:
    n_repeats = max(1, int(np.ceil(cfg.selection_frequency_repeats / cfg.n_splits)))
    rkf = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=n_repeats,
        random_state=cfg.random_state,
    )
    base = build_l1_logistic_pipeline(cfg)
    base.set_params(model__C=float(best_c))

    selected_counts = np.zeros(len(feature_cols), dtype=int)
    coef_sums = np.zeros(len(feature_cols), dtype=float)
    abs_coef_sums = np.zeros(len(feature_cols), dtype=float)
    total_fits = 0

    for train_idx, _ in rkf.split(X_train, y_train):
        if total_fits >= cfg.selection_frequency_repeats:
            break
        est = clone(base)
        est.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        coef = est.named_steps["model"].coef_.ravel()
        selected_counts += (coef != 0).astype(int)
        coef_sums += coef
        abs_coef_sums += np.abs(coef)
        total_fits += 1

    return pd.DataFrame({
        "feature": feature_cols,
        "selection_count": selected_counts,
        "n_repeats": total_fits,
        "selection_frequency": selected_counts / total_fits,
        "mean_coefficient": coef_sums / total_fits,
        "abs_mean_coefficient": abs_coef_sums / total_fits,
        "stability_method": "repeated_stratified_kfold",
    }).sort_values(["selection_frequency", "abs_mean_coefficient"], ascending=[False, False]).reset_index(drop=True)


def compute_selection_frequency(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    best_c: float,
    cfg: SpectralSignatureConfig,
) -> pd.DataFrame:
    method = cfg.stability_resampling_method.strip().lower()
    if method == "bootstrap":
        return _selection_frequency_bootstrap(X_train, y_train, feature_cols, best_c, cfg)
    if method == "repeated_stratified_kfold":
        return _selection_frequency_repeated_kfold(X_train, y_train, feature_cols, best_c, cfg)
    raise ValueError("Unsupported stability_resampling_method. Use 'bootstrap' or 'repeated_stratified_kfold'.")


def summarize_selection(
    prepared: PreparedSelectionData,
    search: GridSearchCV,
    coef_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    cfg: SpectralSignatureConfig,
    map_name: str,
) -> dict:
    selected = coef_df.loc[coef_df["selected"], "feature"].tolist()
    return {
        "map_name": map_name,
        "target_col": cfg.target_col,
        "n_train": int(len(prepared.train_df)),
        "n_test": int(len(prepared.test_df)),
        "n_features_initial": int(len(prepared.feature_cols_before)),
        "n_features_after_preprocessing": int(len(prepared.feature_cols_after)),
        "n_features_selected": int(len(selected)),
        "selected_features": selected,
        "best_C": float(search.best_params_["model__C"]),
        f"best_cv_{cfg.scoring}": float(search.best_score_),
        "dropped_all_missing": prepared.dropped_all_missing,
        "dropped_low_coverage": prepared.dropped_low_coverage,
        "dropped_non_numeric": prepared.dropped_non_numeric,
        "dropped_near_constant": prepared.dropped_near_constant,
        "selection_frequency_threshold": float(cfg.selection_frequency_threshold),
        "selection_frequency_table_rows": int(len(freq_df)),
    }
