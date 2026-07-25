from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


# ── Colour helpers ────────────────────────────────────────────────────────────
def _bar_color(target: int, sex: float, cfg) -> str:
    """Return RGBA hex for one bar based on outcome and sex."""
    is_male = sex == 1
    if int(target) == 1:
        return cfg.COLOR_POSITIVE_MALE if is_male else cfg.COLOR_POSITIVE_FEMALE
    return cfg.COLOR_NEGATIVE_MALE if is_male else cfg.COLOR_NEGATIVE_FEMALE


def _is_error(row: pd.Series) -> bool:
    """True when the predicted label disagrees with the ground-truth target."""
    return int(row["pred_label"]) != int(row["target"])


# ── Legend proxy artists ──────────────────────────────────────────────────────
def _build_legend(cfg) -> list[mpatches.Patch]:
    legend_items = [
        mpatches.Patch(facecolor=cfg.COLOR_POSITIVE_MALE, label="Sarcopenia (male)"),
        mpatches.Patch(facecolor=cfg.COLOR_NEGATIVE_MALE, label="No sarcopenia (male)"),
        mpatches.Patch(facecolor=cfg.COLOR_POSITIVE_FEMALE, label="Sarcopenia (female)"),
        mpatches.Patch(facecolor=cfg.COLOR_NEGATIVE_FEMALE, label="No sarcopenia (female)"),
        mpatches.Patch(
            facecolor="white", edgecolor="#555555", hatch="xx", label="Misclassified (FP / FN)"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=cfg.THRESHOLD_LINE_COLOR,
            linestyle="--",
            linewidth=cfg.THRESHOLD_LINE_WIDTH,
            label="Youden threshold",
        ),
    ]
    return legend_items


# ── Core plotting routine ─────────────────────────────────────────────────────
def plot_probability_distribution_static(
    df: pd.DataFrame,
    out_path: Path,
    split_label: str,
    display_name: str,
    cfg,
    threshold: float | None = None,
) -> None:
    if df is None or df.empty:
        print(f"[plot_static] Empty data for {out_path}, skipping.")
        return

    g = df.sort_values("pred_proba").reset_index(drop=True)
    n = len(g)
    x = np.arange(n, dtype=float)

    bar_width = 1.0 - cfg.BAR_GAP_FRACTION
    colors = [_bar_color(row["target"], row["sex"], cfg) for _, row in g.iterrows()]
    hatches = ["xx" if _is_error(row) else "" for _, row in g.iterrows()]

    fig, ax = plt.subplots(figsize=(cfg.FIGURE_WIDTH, cfg.FIGURE_HEIGHT))

    # Draw bars individually so each can have its own hatch
    for xi, (color, hatch, prob) in enumerate(zip(colors, hatches, g["pred_proba"])):
        ax.bar(
            xi,
            prob,
            width=bar_width,
            color=color,
            edgecolor="#555555" if hatch else "none",
            linewidth=0.5,
            hatch=hatch,
            zorder=2,
        )

    # Youden threshold line
    thr = threshold
    if thr is None and "selected_threshold" in g.columns and g["selected_threshold"].notna().any():
        thr = float(g["selected_threshold"].dropna().iloc[0])
    if thr is not None:
        ax.axhline(
            thr,
            linestyle="--",
            linewidth=cfg.THRESHOLD_LINE_WIDTH,
            color=cfg.THRESHOLD_LINE_COLOR,
            label=f"Youden thr = {thr:.3f}",
            zorder=3,
        )

    # Axes formatting
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cases (sorted by predicted probability)", fontsize=11)
    ax.set_ylabel("Predicted probability", fontsize=11)
    ax.set_title(
        f"{display_name}  ·  {split_label}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        g["patient_id"].tolist(),
        fontsize=6 if n > 40 else 8,
        rotation=90,
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    # Legend
    legend_items = _build_legend(cfg)
    ax.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        ncol=3,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_static] Saved: {out_path}")
