from typing import Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk

import config
from data_loader import MaskByLevel


def _to_bool_array(mask_2d: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.Cast(mask_2d, sitk.sitkUInt8)).astype(bool)


def dice_coefficient(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    """Dice similarity coefficient between two binary masks on the same grid.

    Returns NaN if both masks are empty (undefined overlap).
    """
    a = _to_bool_array(mask_a)
    b = _to_bool_array(mask_b)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * np.logical_and(a, b).sum() / denom


def compute_per_level_dice(patient_masks: Dict[str, MaskByLevel]) -> pd.DataFrame:
    """One row per (patient_id, level, pair) with the Dice coefficient."""
    rows: List[Dict[str, object]] = []
    for patient_id, by_level in patient_masks.items():
        for level in config.LEVELS:
            masks = by_level[level]
            for key_a, key_b, pair_label in config.PAIRS:
                dice = dice_coefficient(masks[key_a], masks[key_b])
                rows.append({
                    "patient_id": patient_id,
                    "level": level,
                    "pair": pair_label,
                    "dice": dice,
                })
    return pd.DataFrame(rows)


def aggregate_per_patient(per_level_df: pd.DataFrame) -> pd.DataFrame:
    """Average the 3 level-specific Dice coefficients within each participant."""
    return (
        per_level_df
        .groupby(["patient_id", "pair"], as_index=False)["dice"]
        .mean()
    )


def summarize_across_patients(per_patient_df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD and median [IQR] of the per-patient Dice, across participants, per pair."""
    pair_order = [label for _, _, label in config.PAIRS]
    summary = (
        per_patient_df
        .groupby("pair")["dice"]
        .agg(
            mean_dice="mean",
            sd_dice="std",
            median_dice="median",
            q1_dice=lambda s: s.quantile(0.25),
            q3_dice=lambda s: s.quantile(0.75),
            n="count",
        )
        .reindex(pair_order)
    )
    summary["iqr_dice"] = summary["q3_dice"] - summary["q1_dice"]
    return summary.reset_index()
