from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def plot_icc_heatmap(
    icc_df: pd.DataFrame,
    title: str,
    out_path: Path,
    value_col: str = "icc",
    feature_col: str = "feature",
) -> None:
    if icc_df is None or icc_df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = icc_df[[feature_col, value_col]].copy().sort_values(value_col, ascending=False)
    if plot_df.empty:
        return

    values = plot_df[value_col].to_numpy(dtype=float).reshape(-1, 1)
    features = plot_df[feature_col].astype(str).tolist()

    fig_h = max(4, min(0.18 * len(features) + 1.5, 40))
    fig, ax = plt.subplots(figsize=(4.8, fig_h))
    im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xticks([0])
    ax.set_xticklabels(["ICC"])
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ICC(2,1)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_profile_simulated(
    df_split: pd.DataFrame,
    feature: str,
    split_name: str,
    simulated_masks: Sequence[str],
    out_dir: Path,
    patient_id_col: str = "patient_id",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df_split[df_split["mask"].isin(simulated_masks)][[patient_id_col, "mask", feature]].copy()
    sub = sub.dropna(subset=[feature])

    wide = sub.pivot_table(index=patient_id_col, columns="mask", values=feature, aggfunc="first")
    wide = wide.reindex(columns=list(simulated_masks)).dropna(axis=0, how="any")
    if wide.empty:
        return

    x_labels = list(simulated_masks)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in wide.iterrows():
        y = row.to_numpy(dtype=float)
        ax.plot(x, y, alpha=0.20, linewidth=0.9)
        ax.scatter(x, y, s=12, alpha=0.25)

    mean_y = wide.mean(axis=0).to_numpy(dtype=float)
    ax.plot(x, mean_y, linewidth=2.5, marker="o", markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_ylabel(feature)
    ax.set_xlabel("Mask variant")
    ax.set_title(f"Simulated masks profile: {feature} ({split_name})")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out_dir / f"profile_simulated_masks_{sanitize_filename(feature)}_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_profile_manual(
    df_split: pd.DataFrame,
    feature: str,
    split_name: str,
    manual_levels: Sequence[str],
    manual_raters: Sequence[str],
    out_dir: Path,
    patient_id_col: str = "patient_id",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_masks = [f"{r}_{l}" for l in manual_levels for r in manual_raters]
    sub = df_split[df_split["mask"].isin(valid_masks)][[patient_id_col, "mask", feature]].copy()
    sub = sub.dropna(subset=[feature])
    if sub.empty:
        return

    sub["rater"] = sub["mask"].str[0]
    sub["level"] = sub["mask"].str[-2:]
    sub["case_id"] = sub[patient_id_col].astype(str) + "__" + sub["level"].astype(str)

    wide = sub.pivot_table(index="case_id", columns="rater", values=feature, aggfunc="first")
    wide = wide.reindex(columns=list(manual_raters)).dropna(axis=0, how="any")
    if wide.empty:
        return

    level_map = {idx: str(idx).split("__")[-1] for idx in wide.index}
    x_labels = list(manual_raters)
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    marker_map = {"l2": "o", "l3": "s", "l4": "^"}

    for case_id, row in wide.iterrows():
        y = row.to_numpy(dtype=float)
        level = level_map.get(case_id, "")
        marker = marker_map.get(level, "o")
        ax.plot(x, y, alpha=0.20, linewidth=0.9)
        ax.scatter(x, y, s=14, alpha=0.28, marker=marker)

    mean_y = wide.mean(axis=0).to_numpy(dtype=float)
    ax.plot(x, mean_y, linewidth=2.5, marker="o", markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(feature)
    ax.set_xlabel("Rater / source")
    ax.set_title(f"Manual vs model profile: {feature} ({split_name})")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out_dir / f"profile_manual_vs_model_{sanitize_filename(feature)}_{split_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_icc_histogram(
    icc_df: pd.DataFrame,
    map_type: str,
    scenario_label: str,
    threshold: float,
    out_path: Path,
    value_col: str = "icc",
    bins: int = 25,
    bar_color: str = "#4C72B0",
    threshold_color: str = "#C44E52",
) -> None:
    """Histogram of feature ICCs for a single spectral map.

    Renders one PNG (300 DPI) showing the distribution of ICC(2,1) values
    across all features computed for one combination of (spectral map,
    robustness scenario). A vertical dashed line marks the ICC threshold
    used to define robust features. The legend reports both the threshold
    value and the number of robust features (ICC > threshold) as
    ``N_robust / N_total``.
    """
    if icc_df is None or icc_df.empty:
        return

    values = pd.to_numeric(icc_df[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = int(values.size)
    n_robust = int(np.sum(values > threshold))

    legend_label = f"Threshold = {threshold:.2f}\n" f"Robust features: {n_robust} / {n_total}"

    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.15):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        sns.histplot(
            values,
            bins=bins,
            binrange=(0.0, 1.0),
            color=bar_color,
            edgecolor="white",
            linewidth=0.6,
            ax=ax,
        )

        ax.axvline(
            threshold,
            color=threshold_color,
            linestyle="--",
            linewidth=2.0,
            label=legend_label,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("ICC(2,1)")
        ax.set_ylabel("Number of features")
        ax.set_title(f"{scenario_label} — {map_type} (train)")

        legend = ax.legend(loc="upper left", frameon=True, fontsize=10)
        legend.get_frame().set_edgecolor("0.7")

        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
