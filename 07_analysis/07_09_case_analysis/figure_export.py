from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server / script use
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from config import FigureConfig, RCFG, RunConfig
from imaging import PatientImageContext


# Column definitions
_COL_LABELS = [
    "CT +\nMuscle Fat map",
]

_N_COLS = len(_COL_LABELS)  # 2


# Row-label colours (one per case type)
_ROW_COLORS = {
    "TP": "#2ca02c",  # green
    "TN": "#1f77b4",  # blue
    "FP": "#ff7f0e",  # orange
    "FN": "#d62728",  # red
}


# Single panel renderers


def _render_plain_ct(ax: plt.Axes, ctx: PatientImageContext) -> None:
    ax.imshow(ctx.ct_display, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")


def _render_ct_with_heatmap(
    ax: plt.Axes,
    ctx: PatientImageContext,
    values: np.ndarray,  # 2-D array, NaN outside mask
    cmap_name: str,
    vmin: float,
    vmax: float,
    fig_cfg: FigureConfig,
    cbar_label: str,
    fig: plt.Figure,
) -> None:
    ax.imshow(ctx.ct_display, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)  # NaN → transparent

    # Build RGBA overlay: heatmap values inside mask, transparent outside
    rgba = cmap(norm(values))
    rgba[..., 3] = np.where(np.isfinite(values), fig_cfg.overlay_alpha, 0.0)
    ax.imshow(rgba, interpolation="bilinear")

    # Colorbar attached to this panel
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, orientation="vertical")
    cbar.set_label(cbar_label, fontsize=fig_cfg.colorbar_label_fontsize, labelpad=2)
    cbar.ax.tick_params(labelsize=fig_cfg.colorbar_label_fontsize - 1)


# Main figure builder
def build_single_case_figure(
    ctx: Optional[PatientImageContext],
    case_type: str,
    run_cfg: RunConfig = RCFG,
) -> plt.Figure:
    fig_cfg = run_cfg.figure
    img_cfg = run_cfg.imaging

    n_cols = _N_COLS
    pw, ph = fig_cfg.panel_size_inches
    left_margin = 0.15  # inches (no row labels needed for a single case)
    top_margin = 0.9  # inches for suptitle + column labels

    fig_w = left_margin + n_cols * pw + 1.5  # + colorbar space
    fig_h = top_margin + ph

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=fig_cfg.dpi)

    gs = fig.add_gridspec(
        1,
        n_cols,
        left=left_margin / fig_w,
        right=0.92,
        top=1.0 - top_margin / fig_h,
        bottom=0.05,
        wspace=0.15,
    )

    axes: List[plt.Axes] = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]

    row_color = _ROW_COLORS.get(case_type, "black")

    fig.suptitle(
        case_type,
        fontsize=fig_cfg.row_label_fontsize + 2,
        fontweight="bold",
        color=row_color,
        y=0.99,
    )

    # ---- Column labels (top of each column) --------------------------------
    for c, label in enumerate(_COL_LABELS):
        axes[c].set_title(
            label, fontsize=fig_cfg.col_label_fontsize, pad=4, fontweight="bold", ha="center"
        )

    for c in range(n_cols):
        ax = axes[c]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(row_color)

        if ctx is None:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="grey",
            )
            continue

        # Col 1: CT + MuscleFat heatmap
        _render_ct_with_heatmap(
            ax=ax,
            ctx=ctx,
            values=ctx.musclefat_raw,
            cmap_name=fig_cfg.cmap_musclefat,
            vmin=img_cfg.musclefat_vmin,
            vmax=img_cfg.musclefat_vmax,
            fig_cfg=fig_cfg,
            cbar_label="Fat fraction (%)",
            fig=fig,
        )

    return fig


def save_case_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 300,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {path}")


# Helper
def _hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b
