from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42


def plot_agreement_boxplot(
    per_patient_df: pd.DataFrame,
    output_path: str,
    rng_seed: int = 42,
) -> None:
    """Render and save the pairwise Dice box + scatter figure.

    Parameters
    ----------
    per_patient_df : DataFrame
        Columns "patient_id", "pair", "dice" — one participant-averaged
        Dice value per pair (as returned by dice_utils.aggregate_per_patient).
    output_path : str
        Full path where the PNG is saved.
    """
    pair_order = [label for _, _, label in config.PAIRS]
    positions = list(range(len(pair_order)))

    box_data = [
        per_patient_df.loc[per_patient_df["pair"] == label, "dice"].dropna().to_numpy()
        for label in pair_order
    ]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#9bc4e2")
        patch.set_alpha(0.55)

    rng = np.random.default_rng(rng_seed)
    for pos, label in zip(positions, pair_order):
        vals = per_patient_df.loc[per_patient_df["pair"] == label, "dice"].dropna().to_numpy()
        xs = pos + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xs, vals, s=35, alpha=0.85, color="#2f6690", edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{label}\n(n = {len(vals)})" for label, vals in zip(pair_order, box_data)],
        fontsize=config.FONT_SIZE,
    )
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_ylabel("Dice similarity coefficient", fontsize=config.FONT_SIZE)
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {output_path}")
