import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import config

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42  # editable fonts in PDF/EPS


def _build_mask(n: int) -> np.ndarray:
    """Upper-triangle mask (True = hidden), diagonal kept visible."""
    mask = np.zeros((n, n), dtype=bool)
    mask[np.triu_indices(n, k=1)] = True  # k=1: above diagonal
    return mask


def plot_kappa_heatmap(
    kappa_df: pd.DataFrame,
    title: str,
    output_path: str,
) -> None:
    """
    Render and save one Cohen's Kappa heatmap.

    Parameters
    ----------
    kappa_df : DataFrame
        Square kappa matrix as returned by kappa_utils.compute_kappa_matrix.
        Index and columns are internal keys (CLASSIFIER_LABELS keys).
    title : str
        Figure title (e.g. "Training set (n = 69)").
    output_path : str
        Full path where the PNG is saved.
    """
    # Remap internal keys → display labels
    display_labels = [config.CLASSIFIER_LABELS.get(k, k) for k in kappa_df.columns]
    plot_df = kappa_df.copy()
    plot_df.index = display_labels
    plot_df.columns = display_labels

    n = len(display_labels)
    mask = _build_mask(n)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    cmap = sns.color_palette(config.HEATMAP_COLORMAP, as_cmap=True)

    hm = sns.heatmap(
        plot_df,
        mask=mask,
        cmap=cmap,
        vmin=config.HEATMAP_VMIN,
        vmax=config.HEATMAP_VMAX,
        annot=True,
        fmt=".2f",
        annot_kws={"size": config.FONT_SIZE_ANNOT, "weight": "bold"},
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={
            "shrink": 0.75,
            "label": "Cohen's κ",
            "ticks": [-1, -0.5, 0, 0.5, 1],
        },
    )

    # ── Colour-bar styling ────────────────────────────────────────────────────
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=config.FONT_SIZE_CBAR)
    cbar.set_label("Cohen's κ", size=config.FONT_SIZE_CBAR + 1, labelpad=8)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xticklabels(
        display_labels,
        rotation=40,
        ha="right",
        fontsize=config.FONT_SIZE_LABELS,
    )
    ax.set_yticklabels(
        display_labels,
        rotation=0,
        fontsize=config.FONT_SIZE_LABELS,
    )
    ax.tick_params(axis="both", which="both", length=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=config.FONT_SIZE_TITLE, fontweight="bold", pad=14)

    # ── Layout & save ─────────────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(
        output_path,
        dpi=config.FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  Saved → {output_path}")
