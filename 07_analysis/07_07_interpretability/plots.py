from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config import ICFG, InterpretabilityConfig

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
matplotlib.rcParams["pdf.fonttype"] = 42  # embed fonts properly


# Internal helpers
def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "–"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _format_or(val: float, lo: float, hi: float, decimals: int = 2) -> str:
    if np.isnan(val):
        return "–"
    if np.isnan(lo) or np.isnan(hi):
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} ({lo:.{decimals}f}–{hi:.{decimals}f})"


# Forest plot
def plot_forest(
    df_or: pd.DataFrame,
    out_path: Path,
    title: str,
    level: str = "score",  # "score" | "feature"
    show_pvalues: bool = True,
    cfg: InterpretabilityConfig = ICFG,
) -> None:
    """
    Draw a forest plot of odds ratios.

    Expected columns in df_or:
        label, or_multi, or_multi_ci_low, or_multi_ci_high,
        or_uni, or_uni_ci_low, or_uni_ci_high,
        [p_value_multi]   -- optional; shown only when show_pvalues=True

    Features are plotted bottom-to-top (first row at the bottom) to match
    the standard radiology forest-plot convention.
    """
    pcfg = cfg.plot
    df = df_or.copy().reset_index(drop=True)
    n = len(df)
    if n == 0:
        print(f"  [forest] Empty OR table, skipping {out_path.name}.")
        return

    # --- Figure layout ---
    # Extra column width for the annotation text depends on show_pvalues
    n_text_cols = 3 if show_pvalues else 2  # OR_uni | OR_multi | [p]
    fig_w, fig_h = pcfg.forest_figsize
    fig_h = max(fig_h, 0.45 * n + 1.5)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Three sub-areas: [label col | plot area | text annotation area]
    label_frac = 0.28
    plot_frac = 0.40
    text_frac = 1.0 - label_frac - plot_frac

    # We use a GridSpec-like trick with axes positioned manually
    ax_plot = fig.add_axes([label_frac, 0.10, plot_frac, 0.80])
    ax_text = fig.add_axes([label_frac + plot_frac + 0.01, 0.10, text_frac - 0.01, 0.80])

    y_positions = np.arange(n)[::-1]  # top row = first feature

    # ---- Colour scheme ----
    COLOR_MULTI = "#1A5276"  # dark blue for multivariable
    COLOR_UNI = "#7F8C8D"  # grey for univariable
    MARKER_SIZE = 6
    CAP_SIZE = 3

    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]

        # Multivariable OR
        or_mv = float(row["or_multi"])
        lo_mv = float(row["or_multi_ci_low"])
        hi_mv = float(row["or_multi_ci_high"])

        # Univariable OR
        or_uv = float(row["or_uni"])
        lo_uv = float(row["or_uni_ci_low"])
        hi_uv = float(row["or_uni_ci_high"])

        y_mv = y + 0.15
        y_uv = y - 0.15

        # --- Multivariable ---
        if not np.isnan(or_mv):
            ax_plot.errorbar(
                or_mv,
                y_mv,
                xerr=[[max(or_mv - lo_mv, 0)], [max(hi_mv - or_mv, 0)]],
                fmt="D",
                color=COLOR_MULTI,
                markersize=MARKER_SIZE,
                capsize=CAP_SIZE,
                linewidth=1.2,
                elinewidth=1.0,
                label="Multivariable" if i == 0 else None,
            )

        # --- Univariable ---
        if not np.isnan(or_uv):
            ax_plot.errorbar(
                or_uv,
                y_uv,
                xerr=[[max(or_uv - lo_uv, 0)], [max(hi_uv - or_uv, 0)]],
                fmt="o",
                color=COLOR_UNI,
                markersize=MARKER_SIZE - 1,
                capsize=CAP_SIZE,
                linewidth=1.0,
                elinewidth=0.8,
                label="Univariable" if i == 0 else None,
            )

    # Reference line at OR = 1
    ax_plot.axvline(x=1.0, linestyle="--", linewidth=0.9, color="black", zorder=0)
    ax_plot.set_yticks(y_positions)
    ax_plot.set_yticklabels([])  # labels drawn separately
    ax_plot.set_xlabel("Odds ratio (95% CI)", fontsize=pcfg.fontsize_axis)
    ax_plot.tick_params(axis="x", labelsize=pcfg.fontsize_tick)
    ax_plot.tick_params(axis="y", length=0)
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_visible(False)
    ax_plot.set_ylim(-0.8, n - 0.2)

    # --- Feature labels (left of plot) ---
    ax_label = fig.add_axes([0.01, 0.10, label_frac - 0.02, 0.80])
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(-0.8, n - 0.2)
    ax_label.axis("off")
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        label_str = str(row["label"])
        ax_label.text(
            1.0,
            y,
            label_str,
            ha="right",
            va="center",
            fontsize=pcfg.fontsize_tick,
            linespacing=1.2,
        )

    # --- Annotation text (right of plot) ---
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(-0.8, n - 0.2)
    ax_text.axis("off")

    # Column headers
    header_y = n - 0.2
    col_xs = [0.02, 0.50]
    headers = ["Univariable OR (95% CI)", "Multivariable OR (95% CI)"]
    if show_pvalues:
        col_xs.append(0.88)
        headers.append("p")
    for cx, hdr in zip(col_xs, headers):
        ax_text.text(
            cx,
            header_y,
            hdr,
            ha="left",
            va="bottom",
            fontsize=pcfg.fontsize_annotation,
            fontweight="bold",
            transform=ax_text.transData,
        )

    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        # Univariable
        ax_text.text(
            col_xs[0],
            y,
            _format_or(row["or_uni"], row["or_uni_ci_low"], row["or_uni_ci_high"]),
            ha="left",
            va="center",
            fontsize=pcfg.fontsize_annotation,
        )
        # Multivariable
        ax_text.text(
            col_xs[1],
            y,
            _format_or(row["or_multi"], row["or_multi_ci_low"], row["or_multi_ci_high"]),
            ha="left",
            va="center",
            fontsize=pcfg.fontsize_annotation,
            color=COLOR_MULTI,
        )
        # p-value
        if show_pvalues and "p_value_multi" in row:
            ax_text.text(
                col_xs[2],
                y,
                _format_p(row["p_value_multi"]),
                ha="left",
                va="center",
                fontsize=pcfg.fontsize_annotation,
            )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_MULTI, label="Multivariable OR"),
        mpatches.Patch(facecolor=COLOR_UNI, label="Univariable OR"),
    ]
    ax_plot.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=pcfg.fontsize_annotation,
        frameon=False,
    )

    fig.suptitle(title, fontsize=pcfg.fontsize_title, y=0.98, fontweight="bold")
    _save(fig, out_path, dpi=pcfg.dpi)


# SHAP bee swarm plot
def plot_shap_beeswarm(
    df_shap: pd.DataFrame,
    out_path: Path,
    title: str,
    cfg: InterpretabilityConfig = ICFG,
    max_features: Optional[int] = None,
    jitter_seed: int = 0,
) -> None:
    """
    Draw a SHAP bee swarm plot.

    Expected columns in df_shap:
        patient_id, feature, label, shap_value, feature_value, target

    Features are sorted by mean |SHAP|, highest at the top.
    Points are jittered on the Y-axis to avoid overplotting.
    Colour encodes normalised feature value (blue = low, red = high).

    Parameters
    ----------
    max_features : if set, only the top-N features by mean |SHAP| are shown.
    jitter_seed  : random seed for Y-axis jitter.
    """
    pcfg = cfg.plot
    df = df_shap.copy()
    if df.empty:
        print(f"  [shap] Empty SHAP table, skipping {out_path.name}.")
        return

    # ---- Select and order features ----
    feat_order = (
        df.groupby("feature")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
    )
    if max_features is not None:
        feat_order = feat_order.head(max_features)

    features_ordered = feat_order.index.tolist()  # high |SHAP| at top
    n_feat = len(features_ordered)
    df = df[df["feature"].isin(features_ordered)].copy()

    # Map feature => y position (0 = bottom, n_feat-1 = top)
    feat_to_y = {f: i for i, f in enumerate(reversed(features_ordered))}
    df["y_pos"] = df["feature"].map(feat_to_y)

    # Feature labels in order (top to bottom on the plot)
    label_map = df.drop_duplicates("feature").set_index("feature")["label"].to_dict()
    y_labels = [label_map.get(f, f) for f in reversed(features_ordered)]

    # ---- Normalise feature values for colouring ----
    # Per-feature min-max normalisation
    norm_vals = np.full(len(df), 0.5)
    for feat in features_ordered:
        mask = df["feature"] == feat
        fv = df.loc[mask, "feature_value"].to_numpy(dtype=float)
        fv_min, fv_max = np.nanmin(fv), np.nanmax(fv)
        if fv_max > fv_min:
            norm_vals[mask.to_numpy()] = (fv - fv_min) / (fv_max - fv_min)
        else:
            norm_vals[mask.to_numpy()] = 0.5
    df["_norm_fv"] = norm_vals

    # Colour map: blue (low) → red (high)
    cmap = matplotlib.colormaps.get_cmap("RdBu_r")

    # ---- Jitter on Y-axis ----
    rng = np.random.default_rng(jitter_seed)
    jitter_strength = 0.22
    df["_jitter"] = rng.uniform(-jitter_strength, jitter_strength, size=len(df))
    df["_y_jittered"] = df["y_pos"] + df["_jitter"]

    # ---- Figure ----
    fig_w, fig_h = pcfg.shap_figsize
    fig_h = max(fig_h, 0.40 * n_feat + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    scatter = ax.scatter(
        df["shap_value"].to_numpy(),
        df["_y_jittered"].to_numpy(),
        c=df["_norm_fv"].to_numpy(),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=18,
        alpha=0.75,
        linewidths=0.0,
        zorder=2,
    )

    # Reference line at SHAP = 0
    ax.axvline(x=0.0, linestyle="--", linewidth=0.9, color="black", zorder=1)

    # Y-axis: feature labels
    ax.set_yticks(list(range(n_feat)))
    ax.set_yticklabels(y_labels, fontsize=pcfg.fontsize_tick)
    ax.set_ylim(-0.8, n_feat - 0.2)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=pcfg.fontsize_axis)
    ax.tick_params(axis="x", labelsize=pcfg.fontsize_tick)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colour bar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=18, pad=0.02)
    cbar.set_label("Feature value\n(normalised)", fontsize=pcfg.fontsize_annotation)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"], fontsize=pcfg.fontsize_annotation)

    fig.suptitle(title, fontsize=pcfg.fontsize_title, fontweight="bold")
    _save(fig, out_path, dpi=pcfg.dpi)
