from typing import Dict

import pandas as pd
import SimpleITK as sitk

from imaging_io import resample_like
from mask import apply_hu_threshold_to_mask


def area_cm2_from_2d_mask(mask2d: sitk.Image) -> float:
    mask2d_u8 = sitk.Cast(mask2d, sitk.sitkUInt8)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask2d_u8)
    if not ls.HasLabel(1):
        return float("nan")
    return float(ls.GetPhysicalSize(1) / 100.0)


def area_cm2_avg_over_slices_3d(mask3d: sitk.Image) -> float:
    """Mean axial muscle area (cm^2) over non-empty slices of a 3D mask."""
    mask_u8 = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(mask_u8)  # (z, y, x)
    spacing = mask3d.GetSpacing()          # (x, y, z)
    pixel_area_mm2 = float(spacing[0]) * float(spacing[1])

    per_slice_counts = arr.reshape(arr.shape[0], -1).sum(axis=1)
    nonempty = per_slice_counts[per_slice_counts > 0]
    if nonempty.size == 0:
        return float("nan")

    mean_pixels = float(nonempty.mean())
    return mean_pixels * pixel_area_mm2 / 100.0


def n_nonempty_slices(mask3d: sitk.Image) -> int:
    mask_u8 = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    arr = sitk.GetArrayViewFromImage(mask_u8)
    per_slice_counts = arr.reshape(arr.shape[0], -1).sum(axis=1)
    return int((per_slice_counts > 0).sum())


def mean_in_mask(img: sitk.Image, mask: sitk.Image) -> float:
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    ls = sitk.LabelStatisticsImageFilter()
    ls.Execute(img_f, mask_u8)
    if not ls.HasLabel(1):
        return float("nan")
    return float(ls.GetMean(1))


def compute_smi(area_cm2: float, height_m: float) -> float:
    if pd.isna(area_cm2) or pd.isna(height_m) or height_m <= 0:
        return float("nan")
    return float(area_cm2 / (height_m ** 2))


# Backward-compatible alias
compute_smi_2d = compute_smi


def compute_conventional_features_on_slice(
    patient_id: str,
    cohort_name: str,
    ct_2d: sitk.Image,
    mask_2d: sitk.Image,
    mask_name: str,
    slice_name: str,
    slice_index: int,
    height_m: float,
    hu_min: float,
    hu_max: float,
) -> Dict[str, object]:
    mask_thr_2d = apply_hu_threshold_to_mask(mask_2d, ct_2d, hu_min=hu_min, hu_max=hu_max)

    area_cm2 = area_cm2_from_2d_mask(mask_thr_2d)
    smi_2d = compute_smi(area_cm2=area_cm2, height_m=height_m)
    mra_2d = mean_in_mask(ct_2d, mask_thr_2d)

    return {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "slice_name": slice_name,
        "slice_index": int(slice_index),
        "smi_2d": round(float(smi_2d), 3) if not pd.isna(smi_2d) else float("nan"),
        "mra_2d": round(float(mra_2d), 3) if not pd.isna(mra_2d) else float("nan"),
        "error": "",
    }


def compute_conventional_features_3d(
    patient_id: str,
    cohort_name: str,
    ct_3d: sitk.Image,
    mask_3d: sitk.Image,
    mask_name: str,
    height_m: float,
    hu_min: float,
    hu_max: float,
) -> Dict[str, object]:
    """3D counterpart of compute_conventional_features_on_slice.

    - SMI uses the muscle area averaged over the non-empty axial slices.
    - MRA is the mean HU within the 3D HU-thresholded mask.

    """
    ct_on_mask = resample_like(ct_3d, mask_3d, is_label=False)
    mask_thr_3d = apply_hu_threshold_to_mask(mask_3d, ct_on_mask, hu_min=hu_min, hu_max=hu_max)

    area_cm2 = area_cm2_avg_over_slices_3d(mask_thr_3d)
    smi_3d = compute_smi(area_cm2=area_cm2, height_m=height_m)
    mra_3d = mean_in_mask(ct_on_mask, mask_thr_3d)

    return {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "n_slices": n_nonempty_slices(mask_thr_3d),
        "smi_3d": round(float(smi_3d), 3) if not pd.isna(smi_3d) else float("nan"),
        "mra_3d": round(float(mra_3d), 3) if not pd.isna(mra_3d) else float("nan"),
        "error": "",
    }
