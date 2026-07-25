from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


def plot_variance_distribution(
    variance_summary: pd.DataFrame,
    out_path: Path,
    title_name: str,
    variance_column: str = "variance_train",
    variance_threshold: float | None = None,
) -> None:
    if variance_summary.empty or variance_column not in variance_summary.columns:
        return
    vals = variance_summary[variance_column].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    if len(vals) <= 20:
        labels = (
            variance_summary.loc[variance_summary[variance_column].notna(), "feature"]
            .astype(str)
            .tolist()
        )
        plt.bar(labels, vals)
        plt.xticks(rotation=90, fontsize=7)
        plt.ylabel("Training variance")
    else:
        plot_vals = np.clip(vals, a_min=1e-16, a_max=None)
        plt.hist(np.log10(plot_vals), bins=40)
        if variance_threshold is not None:
            plt.axvline(np.log10(max(variance_threshold, 1e-16)), linestyle="--")
        plt.xlabel("log10(training variance)")
        plt.ylabel("Number of features")
    plt.title(f"Variance distribution — {title_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_counts(
    counts: list[int],
    labels: list[str],
    out_path: Path,
    title_name: str,
) -> None:
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    plt.ylabel("Number of features")
    plt.title(f"Feature filtering summary — {title_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _compute_linkage(corr_abs: pd.DataFrame) -> np.ndarray:
    dist = 1.0 - corr_abs.copy()
    np.fill_diagonal(dist.values, 0.0)
    return linkage(squareform(dist.values, checks=False), method="average")


def compute_feature_clustering(
    corr_abs: pd.DataFrame, rho_threshold: float
) -> Tuple[np.ndarray, list[str], np.ndarray]:
    if corr_abs.empty:
        return np.empty((0, 4)), [], np.array([])
    features = corr_abs.index.tolist()
    if len(features) == 1:
        return np.empty((0, 4)), features, np.array([1])
    z = _compute_linkage(corr_abs)
    dendro = dendrogram(z, no_plot=True, labels=features)
    cluster_ids = fcluster(z, t=1.0 - rho_threshold, criterion="distance")
    return z, dendro["ivl"], cluster_ids


def plot_spearman_dendrogram(
    corr_abs: pd.DataFrame,
    rho_threshold: float,
    out_path: Path,
    title_name: str,
    force_labels: bool = True,
) -> Optional[pd.DataFrame]:
    if corr_abs.empty:
        return None
    features = corr_abs.index.tolist()
    if len(features) == 1:
        out = pd.DataFrame({"leaf_order": [1], "feature": features, "cluster_id": [1]})
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, features[0], ha="center", va="center")
        plt.axis("off")
        plt.title(f"Spearman feature dendrogram — {title_name}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out

    z, ordered, cluster_ids = compute_feature_clustering(corr_abs, rho_threshold)
    cluster_map = dict(zip(features, cluster_ids))
    leaf_table = pd.DataFrame(
        {
            "leaf_order": np.arange(1, len(ordered) + 1),
            "feature": ordered,
            "cluster_id": [int(cluster_map[f]) for f in ordered],
        }
    )

    n = len(features)
    labels = features if force_labels else (features if n <= 120 else None)
    font_size = 7 if n <= 50 else 5 if n <= 120 else 3
    plt.figure(figsize=(max(10, min(40, n * 0.16)), 8))
    dendrogram(
        z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=font_size,
        color_threshold=1.0 - rho_threshold,
        above_threshold_color="black",
    )
    plt.axhline(
        1.0 - rho_threshold,
        linestyle="--",
        linewidth=1.2,
        label=f"Cut: |rho| >= {rho_threshold:.2f}",
    )
    plt.ylabel("Distance = 1 - |Spearman rho|")
    plt.xlabel("Features")
    plt.title(f"Spearman feature dendrogram — {title_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return leaf_table


def plot_clustered_abs_spearman_heatmap(
    corr_abs: pd.DataFrame,
    rho_threshold: float,
    out_path: Path,
    title_name: str,
    max_features_for_plot: int = 150,
) -> Optional[pd.DataFrame]:
    if corr_abs.empty:
        return None
    features = corr_abs.index.tolist()
    if len(features) == 1:
        table = pd.DataFrame({"plot_order": [1], "feature": features, "cluster_id": [1]})
        mat = np.array([[1.0]])
        plot_features = features
        plot_cluster_ids = [1]
    else:
        _, ordered, cluster_ids = compute_feature_clustering(corr_abs, rho_threshold)
        cluster_map = dict(zip(features, cluster_ids))
        table = pd.DataFrame(
            {
                "plot_order": np.arange(1, len(ordered) + 1),
                "feature": ordered,
                "cluster_id": [int(cluster_map[f]) for f in ordered],
            }
        )
        plot_features = ordered[:max_features_for_plot]
        mat = corr_abs.loc[plot_features, plot_features].values
        plot_cluster_ids = [int(cluster_map[f]) for f in plot_features]

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="|Spearman rho|")
    plt.title(f"Clustered absolute Spearman correlation — {title_name}")
    if len(plot_features) <= 50:
        plt.xticks(np.arange(len(plot_features)), plot_features, rotation=90, fontsize=6)
        plt.yticks(np.arange(len(plot_features)), plot_features, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])

    for i in range(1, len(plot_cluster_ids)):
        if plot_cluster_ids[i] != plot_cluster_ids[i - 1]:
            pos = i - 0.5
            plt.axhline(pos, linewidth=0.8)
            plt.axvline(pos, linewidth=0.8)

    if len(features) > max_features_for_plot:
        plt.xlabel(
            f"Showing first {max_features_for_plot} features in dendrogram order of {len(features)}"
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return table
