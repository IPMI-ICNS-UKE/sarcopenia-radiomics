from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.lines import Line2D

import config
from dice_utils import dice_coefficient

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42

RATER_COLORS = {"i": "#d62728", "j": "#2ca02c", "a": "#1f77b4"}
RATER_LABELS = {"i": "Reader 1", "j": "Reader 2", "a": "Automated"}


def _crop_bbox(mask_arrays: List[np.ndarray], margin: int) -> Tuple[int, int, int, int]:
    """Bounding box (r0, r1, c0, c1) around the union of binary arrays, plus margin."""
    union = np.zeros_like(mask_arrays[0], dtype=bool)
    for arr in mask_arrays:
        union |= arr.astype(bool)

    rows = np.where(np.any(union, axis=1))[0]
    cols = np.where(np.any(union, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])

    r0 = max(0, r0 - margin)
    r1 = min(union.shape[0] - 1, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(union.shape[1] - 1, c1 + margin)
    return r0, r1, c0, c1


def plot_segmentation_overlay(
    patient_id: str,
    level: str,
    ct_2d: sitk.Image,
    masks_by_rater: Dict[str, sitk.Image],
    output_path: str,
    window_center: float = 40.0,
    window_width: float = 400.0,
    margin_px: int = 25,
) -> None:
    """Render and save the L3 CT slice with the three rater contours overlaid."""
    ct_arr = sitk.GetArrayFromImage(ct_2d).astype(float)
    mask_arrays = {r: sitk.GetArrayFromImage(m).astype(np.uint8) for r, m in masks_by_rater.items()}

    r0, r1, c0, c1 = _crop_bbox(list(mask_arrays.values()), margin=margin_px)
    ct_crop = ct_arr[r0:r1 + 1, c0:c1 + 1]
    mask_crop = {r: arr[r0:r1 + 1, c0:c1 + 1] for r, arr in mask_arrays.items()}

    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    fig, ax = plt.subplots(figsize=(6.0, 6.5))
    ax.imshow(ct_crop, cmap="gray", vmin=vmin, vmax=vmax)

    for rater in ("i", "j", "a"):
        ax.contour(mask_crop[rater], levels=[0.5], colors=[RATER_COLORS[rater]], linewidths=1.8)

    dice_ij = dice_coefficient(masks_by_rater["i"], masks_by_rater["j"])
    dice_ia = dice_coefficient(masks_by_rater["i"], masks_by_rater["a"])
    dice_ja = dice_coefficient(masks_by_rater["j"], masks_by_rater["a"])

    handles = [
        Line2D([0], [0], color=RATER_COLORS[r], lw=1.8, label=RATER_LABELS[r])
        for r in ("i", "j", "a")
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    ax.set_title(
        f"{patient_id} — {level.upper()} axial slice\n"
        f"Dice: R1–R2 {dice_ij:.2f} · R1–Auto {dice_ia:.2f} · R2–Auto {dice_ja:.2f}",
        fontsize=11,
    )
    ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {output_path}")
