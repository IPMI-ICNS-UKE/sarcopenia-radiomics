import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from typing import Dict

import config_extended as config


# Layout constants (tuned for Radiology journal style)
FONT_FAMILY = "Arial"
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 9
FONT_SIZE_ANNOT = 9
FONT_SIZE_TICK = 7
FIGURE_WIDTH = 7.0  # inches (journal column width ~ 3.5 in, full page ~ 7 in)
FIGURE_HEIGHT = 2.8

# Color boundaries for p-value significance
P_BOUNDARIES = [0.0, 0.001, 0.01, 0.05, 0.10, 1.01]
P_COLORS = ["#67001f", "#d6604d", "#f4a582", "#d1e5f0", "#e0e0e0"]
P_LABELS = ["p<0.001", "0.001≤p<0.01", "0.01≤p<0.05", "0.05≤p<0.10", "p≥0.10"]

ROW_METHODS = ["auto_smi", "auto_mra_score_clinical"]
COL_METHODS = [
    "musclefat_mean_fraction_score_clinical",
    "musclefat_signature_score_clinical",
    "mf_ct_scores_clinical",
]

ROW_LABELS = [config.METHOD_DISPLAY_NAMES[m] for m in ROW_METHODS]
COL_LABELS = [
    "Muscle fat -\nmean fraction",
    "Muscle fat -\nradiomics signature",
    "Handcrafted radiomics\n(CT + muscle fat)",
]


# Build p-value matrix
def _build_pvalue_matrix(
    pvalues: Dict,
    use_corrected: bool = True,
) -> np.ndarray:
    """
    Build a (n_baselines x n_new_methods) matrix of p-values.
    Rows = baselines (SMI, MRA), Cols = new methods (mean_fraction, signature, top3).
    Missing values are set to NaN.
    """
    matrix = np.full((len(ROW_METHODS), len(COL_METHODS)), np.nan)
    for i, baseline in enumerate(ROW_METHODS):
        for j, new_method in enumerate(COL_METHODS):
            key = (new_method, baseline)
            if key in pvalues:
                p = pvalues[key].get("corrected" if use_corrected else "raw")
                if p is not None:
                    matrix[i, j] = p
    return matrix


def _pvalue_to_annotation(p: float) -> str:
    """Convert p-value to annotation string."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"p<0.001\n***"
    if p < 0.01:
        return f"p={p:.3f}\n**"
    if p < 0.05:
        return f"p={p:.3f}\n*"
    return f"p={p:.3f}\nns"


# Main heatmap function
def plot_pvalue_heatmap(
    pvalues: Dict,
    subset_key: str,
    task: str,
    output_dir: str,
    use_corrected: bool = True,
) -> None:
    """
    Plot and save a publication-quality p-value heatmap.

    Parameters
    ----------
    pvalues : dict from compute_pvalues_block
    subset_key : str  "train", "test_1", "test_2"
    task : str  "cls" or "reg"
    output_dir : str
    use_corrected : bool  Use corrected or raw p-values for display
    """
    matrix = _build_pvalue_matrix(pvalues, use_corrected=use_corrected)

    # Assign color indices
    color_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p = matrix[i, j]
            if np.isnan(p):
                color_matrix[i, j] = len(P_COLORS) - 1  # grey for N/A
            else:
                for k in range(len(P_BOUNDARIES) - 1):
                    if P_BOUNDARIES[k] <= p < P_BOUNDARIES[k + 1]:
                        color_matrix[i, j] = k
                        break

    # ─── Figure setup ─────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # ─── Heatmap cells ────────────────────────────────────────────────────────
    cmap = matplotlib.colors.ListedColormap(P_COLORS)
    bounds = np.arange(-0.5, len(P_COLORS) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(color_matrix, cmap=cmap, norm=norm, aspect="auto")

    # ─── Cell annotations ─────────────────────────────────────────────────────
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p = matrix[i, j]
            annot = _pvalue_to_annotation(p)
            # Text color: white on dark cells, dark on light cells
            ci = int(color_matrix[i, j])
            text_color = "white" if ci <= 1 else "#1a1a1a"
            ax.text(
                j,
                i,
                annot,
                ha="center",
                va="center",
                fontsize=FONT_SIZE_ANNOT,
                color=text_color,
                fontweight="bold" if not np.isnan(p) and p < 0.05 else "normal",
                linespacing=1.4,
            )

    # ─── Axis labels ──────────────────────────────────────────────────────────
    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(
        COL_LABELS,
        fontsize=FONT_SIZE_TICK,
        ha="center",
        multialignment="center",
        wrap=True,
    )
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=FONT_SIZE_TICK)

    # ─── Gridlines between cells ──────────────────────────────────────────────
    ax.set_xticks(np.arange(-0.5, len(COL_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROW_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ─── Title ────────────────────────────────────────────────────────────────
    task_label = (
        "Classification (AUC, DeLong test)"
        if (task == "cls" or task == "cls_2")
        else "Regression (absolute errors, Wilcoxon signed-rank test)"
    )
    subset_label = config.DATA_SUBSETS[task][subset_key]["display_name"]
    correction_label = {
        "fdr_bh": "BH-FDR corrected",
        "bonferroni": "Bonferroni corrected",
        "none": "uncorrected",
    }.get(config.CORRECTION_METHOD, config.CORRECTION_METHOD)

    ax.set_title(
        f"P-value comparison — {task_label}\n{subset_label} | {correction_label}",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=12,
    )

    # ─── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=P_COLORS[k], edgecolor="grey", linewidth=0.5, label=P_LABELS[k])
        for k in range(len(P_COLORS))
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=FONT_SIZE_LABEL - 1,
        title="P-value range",
        title_fontsize=FONT_SIZE_LABEL,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
    )

    # ─── Axis labels ──────────────────────────────────────────────────────────
    ax.set_xlabel("New method", fontsize=FONT_SIZE_LABEL, labelpad=8)
    ax.set_ylabel("Baseline", fontsize=FONT_SIZE_LABEL, labelpad=8)

    # ─── Layout & export ──────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 0.82, 1.0])

    filename = f"heatmap_pvalue_{task}_{subset_key}.{config.FIGURE_FORMAT}"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=config.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# Convenience: plot all heatmaps
def plot_all_heatmaps(
    all_pvalues: Dict,
    output_dir: str,
    use_corrected: bool = True,
) -> None:
    for task in ["cls", "reg", "cls_2"]:
        for subset_key in ["train", "test_1", "test_2"]:
            pvalues = all_pvalues.get(task, {}).get(subset_key, {})
            if not pvalues:
                print(f"  [SKIP] No p-values for [{task}][{subset_key}]")
                continue
            plot_pvalue_heatmap(
                pvalues=pvalues,
                subset_key=subset_key,
                task=task,
                output_dir=output_dir,
                use_corrected=use_corrected,
            )
