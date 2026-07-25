from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class AnalysisResult:
    df_all: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    selected_features: List[str]
    removed_features: List[str]
    variance_summary: pd.DataFrame
    correlation_summary: pd.DataFrame
    corr_matrix_train: pd.DataFrame
    overall_summary: pd.DataFrame


def safe_variance(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    return 0.0 if len(vals) <= 1 else float(vals.var(ddof=1))


def safe_nunique(x: pd.Series) -> int:
    return int(pd.to_numeric(x, errors="coerce").dropna().nunique())


def assess_variance(X_train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X_train.columns:
        vals = pd.to_numeric(X_train[col], errors="coerce").dropna()
        rows.append(
            {
                "feature": col,
                "variance_train": safe_variance(X_train[col]),
                "std_train": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "mean_train": float(vals.mean()) if len(vals) else np.nan,
                "median_train": float(vals.median()) if len(vals) else np.nan,
                "min_train": float(vals.min()) if len(vals) else np.nan,
                "max_train": float(vals.max()) if len(vals) else np.nan,
                "n_unique_train": safe_nunique(X_train[col]),
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def near_zero_variance_filter(
    X_train: pd.DataFrame,
    variance_threshold: float,
    min_unique_values: int,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    rows = []
    for col in X_train.columns:
        variance = safe_variance(X_train[col])
        nunique = safe_nunique(X_train[col])
        keep = bool((variance > variance_threshold) and (nunique >= min_unique_values))
        rows.append(
            {
                "feature": col,
                "variance_train": variance,
                "n_unique_train": nunique,
                "keep_after_nzv": keep,
                "drop_reason_nzv": (
                    "low_variance"
                    if variance <= variance_threshold
                    else "low_unique_values" if nunique < min_unique_values else ""
                ),
            }
        )
    summary = (
        pd.DataFrame(rows)
        .sort_values(
            by=["keep_after_nzv", "variance_train", "n_unique_train", "feature"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )
    kept = summary.loc[summary["keep_after_nzv"], "feature"].tolist()
    removed = summary.loc[~summary["keep_after_nzv"], "feature"].tolist()
    return kept, removed, summary


def correlation_clustering_summary(
    X_train: pd.DataFrame,
    rho_threshold: float,
    df_stability: Optional[pd.DataFrame] = None,
    use_representatives: bool = False,
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    features = X_train.columns.tolist()
    if len(features) == 0:
        return [], [], pd.DataFrame(), pd.DataFrame()

    corr = X_train.corr(method="spearman").fillna(0.0)
    corr_abs = corr.abs().clip(lower=0.0, upper=1.0)

    if len(features) == 1:
        summary = pd.DataFrame(
            [
                {
                    "feature": features[0],
                    "cluster_id": 1,
                    "cluster_size": 1,
                    "mean_abs_spearman_to_others": 1.0,
                    "is_representative": True,
                    "representative_feature": features[0],
                    "abs_spearman_to_representative": 1.0,
                }
            ]
        )
        return features, [], summary, corr_abs

    dist = 1.0 - corr_abs
    np.fill_diagonal(dist.values, 0.0)
    Z = linkage(squareform(dist.values, checks=False), method="average")
    cluster_ids = fcluster(Z, t=1.0 - rho_threshold, criterion="distance")

    base = pd.DataFrame({"feature": features, "cluster_id": cluster_ids.astype(int)})
    cluster_sizes = base.groupby("cluster_id")["feature"].count().to_dict()
    base["cluster_size"] = base["cluster_id"].map(cluster_sizes).astype(int)
    base["mean_abs_spearman_to_others"] = [
        float(corr_abs.loc[f, [o for o in features if o != f]].mean()) for f in features
    ]

    if not use_representatives:
        return (
            features,
            [],
            base.sort_values(["cluster_id", "feature"]).reset_index(drop=True),
            corr_abs,
        )

    icc_map = {}
    if df_stability is not None and {"feature", "mean_train_icc"}.issubset(df_stability.columns):
        icc_map = df_stability.set_index("feature")["mean_train_icc"].to_dict()

    base["mean_train_icc"] = base["feature"].map(icc_map).astype(float)
    base["variance_train"] = [safe_variance(X_train[f]) for f in base["feature"]]

    selected = []
    rows = []
    for cluster_id, df_cluster in base.groupby("cluster_id", sort=True):
        ranked = df_cluster.copy()
        ranked["icc_rank"] = ranked["mean_train_icc"].fillna(-np.inf)
        ranked = ranked.sort_values(
            by=["icc_rank", "variance_train", "feature"],
            ascending=[False, False, True],
        )
        representative = str(ranked.iloc[0]["feature"])
        selected.append(representative)

        for _, row in df_cluster.iterrows():
            feat = row["feature"]
            rows.append(
                {
                    "feature": feat,
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(row["cluster_size"]),
                    "mean_abs_spearman_to_others": float(row["mean_abs_spearman_to_others"]),
                    "mean_train_icc": (
                        float(row["mean_train_icc"]) if pd.notna(row["mean_train_icc"]) else np.nan
                    ),
                    "variance_train": float(row["variance_train"]),
                    "is_representative": bool(feat == representative),
                    "representative_feature": representative,
                    "abs_spearman_to_representative": float(corr_abs.loc[feat, representative]),
                }
            )

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            by=["cluster_id", "is_representative", "feature"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    removed = [f for f in features if f not in selected]
    return selected, removed, summary, corr_abs


def subset_with_ids(
    df: pd.DataFrame, id_columns: List[str], feature_cols: List[str]
) -> pd.DataFrame:
    return df.loc[:, [c for c in id_columns if c in df.columns] + feature_cols].copy()


def run_analysis_only_pipeline(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    id_columns: List[str],
    rho_threshold: float,
    dataset_name: str,
    split_col: str = "split",
) -> AnalysisResult:
    df_train = (
        df_all[df_all[split_col] == "train"].reset_index(drop=True)
        if split_col in df_all.columns
        else pd.DataFrame()
    )
    df_test = (
        df_all[df_all[split_col] == "test"].reset_index(drop=True)
        if split_col in df_all.columns
        else pd.DataFrame()
    )
    analysis_df = df_train if not df_train.empty else df_all
    analysis_split = "train" if not df_train.empty else "all"

    X_train = analysis_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    variance_summary = assess_variance(X_train)
    selected, removed, corr_summary, corr_matrix = correlation_clustering_summary(
        X_train=X_train,
        rho_threshold=rho_threshold,
        use_representatives=False,
    )

    overall = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "analysis_split": analysis_split,
                "n_rows_train": int(len(df_train)),
                "n_rows_test": int(len(df_test)),
                "n_initial_features": int(len(feature_cols)),
                "n_after_analysis": int(len(feature_cols)),
                "n_removed_features": 0,
                "spearman_rho_threshold_for_clustering": float(rho_threshold),
                "n_correlation_clusters": (
                    int(corr_summary["cluster_id"].nunique()) if not corr_summary.empty else 0
                ),
            }
        ]
    )

    empty_cols = [*id_columns, *feature_cols]
    return AnalysisResult(
        df_all=subset_with_ids(df_all, id_columns, feature_cols),
        df_train=(
            subset_with_ids(df_train, id_columns, feature_cols)
            if not df_train.empty
            else pd.DataFrame(columns=empty_cols)
        ),
        df_test=(
            subset_with_ids(df_test, id_columns, feature_cols)
            if not df_test.empty
            else pd.DataFrame(columns=empty_cols)
        ),
        selected_features=feature_cols,
        removed_features=[],
        variance_summary=variance_summary,
        correlation_summary=corr_summary,
        corr_matrix_train=corr_matrix,
        overall_summary=overall,
    )


def run_spectral_filtering_pipeline(
    df_all: pd.DataFrame,
    feature_cols: List[str],
    id_columns: List[str],
    df_stability: pd.DataFrame,
    map_name: str,
    variance_threshold: float,
    min_unique_values: int,
    rho_threshold: float,
    split_col: str = "split",
) -> AnalysisResult:
    df_train = df_all[df_all[split_col] == "train"].copy()
    if df_train.empty:
        raise ValueError(
            f"No training cases found for map '{map_name}'. Filtering must be learned on cohort 1 training data."
        )
    df_test = df_all[df_all[split_col] == "test"].copy()
    X_train = df_train[feature_cols].apply(pd.to_numeric, errors="coerce")

    kept_nzv, removed_nzv, nzv_summary = near_zero_variance_filter(
        X_train=X_train,
        variance_threshold=variance_threshold,
        min_unique_values=min_unique_values,
    )
    selected, removed_corr, corr_summary, corr_matrix = correlation_clustering_summary(
        X_train=X_train[kept_nzv],
        rho_threshold=rho_threshold,
        df_stability=df_stability,
        use_representatives=True,
    )

    overall = pd.DataFrame(
        [
            {
                "map_name": map_name,
                "n_train_cases_used_for_filtering": int(len(df_train)),
                "n_initial_stable_features": int(len(feature_cols)),
                "n_after_nzv": int(len(kept_nzv)),
                "n_after_correlation_filtering": int(len(selected)),
                "n_removed_nzv": int(len(removed_nzv)),
                "n_removed_correlation": int(len(removed_corr)),
                "variance_threshold": float(variance_threshold),
                "min_unique_values": int(min_unique_values),
                "spearman_rho_threshold": float(rho_threshold),
            }
        ]
    )

    return AnalysisResult(
        df_all=subset_with_ids(df_all, id_columns, selected),
        df_train=subset_with_ids(df_train.reset_index(drop=True), id_columns, selected),
        df_test=subset_with_ids(df_test.reset_index(drop=True), id_columns, selected),
        selected_features=selected,
        removed_features=removed_nzv + removed_corr,
        variance_summary=nzv_summary,
        correlation_summary=corr_summary,
        corr_matrix_train=corr_matrix,
        overall_summary=overall,
    )
