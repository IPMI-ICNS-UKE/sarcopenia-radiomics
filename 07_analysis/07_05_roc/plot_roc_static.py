import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42   # editable fonts in PDF/AI
matplotlib.rcParams["ps.fonttype"] = 42

from config import (
    DPI,
    FIGURE_SIZE_IN,
    FONT_FALLBACKS,
    FONT_SIZE_AXIS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_STYLES,
    LINE_WIDTH,
    MODEL_ORDER,
    PALETTE,
)
from roc_utils import RocData

sns.set_theme(style="ticks", context="paper")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = FONT_FALLBACKS


def _make_linestyle(dash_pattern):
    """Convert a dash-gap tuple to a matplotlib linestyle tuple, or return solid."""
    if dash_pattern is None:
        return (0, ())  # solid
    return (0, dash_pattern)


def plot_roc_static(
    roc_data_list: List[RocData],
    title: str,
    output_path: str,
) -> None:
    """
    Draw and save a publication-ready ROC curve PNG.

    Parameters
    ----------
    roc_data_list : list of RocData
        Pre-computed ROC data, one entry per model.
    title : str
        Plot title (e.g. 'Training set (OOF)').
    output_path : str
        Full path for the output PNG file (must end in .png).
    """
    # Build a lookup from model_name → (colour, linestyle) in ORDER
    style_map = {
        m: (PALETTE[i], LINE_STYLES[i])
        for i, m in enumerate(MODEL_ORDER)
        if i < len(PALETTE)
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, dpi=DPI)

    # Diagonal reference line (drawn first, behind the ROC curves)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#a3a3a3",
        linestyle=(0, (2, 1.5)),
        linewidth=1.0,
        label="Reference (no discrimination)",
        zorder=0,
    )

    for roc in roc_data_list:
        colour, dash = style_map.get(roc.model_name, ("#333333", None))
        ls = _make_linestyle(dash)
        label = f"{roc.display_name} — {roc.auc_label}"
        ax.plot(
            roc.fpr,
            roc.tpr,
            color=colour,
            linestyle=ls,
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            dash_capstyle="round",
            label=label,
            zorder=2,
        )

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("1 − Specificity (FPR)", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=FONT_SIZE_AXIS)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, pad=10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_SIZE_TICK,
        length=4,
        width=0.8,
    )
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    sns.despine(ax=ax, trim=True)

    # ── Legend — placed below the axes as a single stacked column ──────────
    handles, labels = ax.get_legend_handles_labels()
    # Keep curve order first, reference line last
    handles = handles[1:] + handles[:1]
    labels = labels[1:] + labels[:1]
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
        ncol=1,
        labelspacing=0.7,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.30)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [static] Saved → {output_path}")
