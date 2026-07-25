from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# Configuration
PATIENT_IDS: List[str] = [
    "Pat1",
    "Pat2",
    "Pat3",
    "Pat4",
    "Pat5",
    "Pat6",
    "Pat8",
    "Pat9",
    "Pat10",
    "Pat11",
    "Pat12",
    "Pat13",
    "Pat14",
    "Pat15",
    "Pat17",
    "Pat18",
    "Pat20",
    "Pat21",
    "Pat22",
    "Pat24",
    "Pat27",
    "Pat29",
    "Pat30",
]


@dataclass(frozen=True)
class PathsConfig:
    images_root: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll"
    )
    labels_root: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/labelsAll"
    )
    out_xlsx: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/full_table/ct_parameters_liver.xlsx"
    )
    out_csv: Optional[Path] = None


@dataclass(frozen=True)
class ParamConfig:
    hu_min: float = -29.0
    hu_max: float = 150.0

    abdominal_wall: str = "abdominal_wall.nii.gz"
    paravertebral: str = "paravertebral.nii.gz"
    l3_mask: str = "vertebra_l3.nii.gz"

    hu_map: str = "HU.nii.gz"
    fat_map: str = "MuscleFat.nii.gz"

    # Known problematic cases allowed to fail without stopping the run
    optionally_drop_on_failure: Tuple[str, ...] = ("Pat8", "Pat11")


CFG = PathsConfig()
PCFG = ParamConfig()


# Utilities
def natural_sort_key(s: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", s)
    return (int(match.group(1)) if match else 0, s)


def read_image(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return sitk.ReadImage(str(path), pixel_type)


def patient_map_path(patient_dir: Path, patient_id: str, suffix: str) -> Path:
    return patient_dir / f"{patient_id}-{suffix}"


def resample_like(
    moving: sitk.Image, fixed: sitk.Image, is_label: bool, default_value: float = 0.0
) -> sitk.Image:
    """
    Resample moving image onto the fixed image grid using identity transform.
    Labels use nearest neighbor interpolation.
    Continuous images use linear interpolation.
    """
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    out = sitk.Resample(
        moving,
        fixed,
        sitk.Transform(),
        interpolator,
        default_value,
        moving.GetPixelID(),
    )

    if is_label:
        return sitk.Cast(out, sitk.sitkUInt8)
    return sitk.Cast(out, sitk.sitkFloat32)


def _binary_foreground(img: sitk.Image) -> sitk.Image:
    """
    Robust conversion of segmentation-like images to binary.
    Handles masks encoded as 0/1 or -1024/-1023 or other non-zero labels.
    """
    img_i = sitk.Cast(img, sitk.sitkInt32)

    # Standard non-zero foreground
    foreground = sitk.NotEqual(img_i, 0)

    # If background is -1024 and foreground is -1023, non-zero alone is not enough.
    # Exclude the common background encoding explicitly.
    not_bg_1024 = sitk.NotEqual(img_i, -1024)
    foreground = sitk.And(foreground, not_bg_1024)

    foreground = sitk.Cast(foreground, sitk.sitkUInt8)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(foreground)
    if stats.GetSum() == 0:
        # Fallback: any negative value except -1024
        neg = sitk.And(sitk.Less(img_i, 0), sitk.NotEqual(img_i, -1024))
        neg = sitk.Cast(neg, sitk.sitkUInt8)
        stats.Execute(neg)
        if stats.GetSum() == 0:
            raise ValueError("Could not identify foreground voxels in segmentation.")
        return neg

    return foreground


def l3_binary_mask_from_encoded(l3_img: sitk.Image) -> sitk.Image:
    """
    Robustly decode L3 mask from possible encodings:
    - 1 / 0
    - -1023 / -1024
    - other non-zero foreground encodings
    """
    l3 = sitk.Cast(l3_img, sitk.sitkInt32)

    is_neg1023 = sitk.Equal(l3, -1023)
    is_one = sitk.Equal(l3, 1)
    is_l3 = sitk.Or(is_neg1023, is_one)
    is_l3 = sitk.Cast(is_l3, sitk.sitkUInt8)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(is_l3)
    if stats.GetSum() > 0:
        return is_l3

    return _binary_foreground(l3_img)


def l3_centroid_phys(l3_mask_img: sitk.Image) -> Tuple[float, float, float]:
    l3_bin = l3_binary_mask_from_encoded(l3_mask_img)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(l3_bin)
    if not ls.HasLabel(1):
        raise ValueError("L3 binary mask has no label=1.")
    return ls.GetCentroid(1)


def axial_index_from_phys(img: sitk.Image, phys_point_xyz: Tuple[float, float, float]) -> int:
    idx = img.TransformPhysicalPointToIndex(phys_point_xyz)
    return int(np.clip(idx[2], 0, img.GetSize()[2] - 1))


def extract_axial_slice(img: sitk.Image, k: int) -> sitk.Image:
    size = list(img.GetSize())
    if len(size) != 3:
        raise ValueError(f"Expected 3D image, got size={size}")
    size[2] = 0
    index = [0, 0, int(k)]
    return sitk.Extract(img, size=size, index=index)


def combine_muscle_masks(abw: sitk.Image, para: sitk.Image) -> sitk.Image:
    abw_b = _binary_foreground(abw)
    para_b = _binary_foreground(para)
    return sitk.Or(abw_b, para_b)


def apply_hu_threshold_to_mask(
    muscle_mask: sitk.Image, hu_img: sitk.Image, hu_min: float, hu_max: float
) -> sitk.Image:
    hu_ok = sitk.And(sitk.GreaterEqual(hu_img, hu_min), sitk.LessEqual(hu_img, hu_max))
    hu_ok = sitk.Cast(hu_ok, sitk.sitkUInt8)
    return sitk.And(sitk.Cast(muscle_mask, sitk.sitkUInt8), hu_ok)


def area_cm2_from_2d_mask(mask2d: sitk.Image) -> float:
    mask2d_u8 = sitk.Cast(mask2d, sitk.sitkUInt8)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask2d_u8)
    if not ls.HasLabel(1):
        return float("nan")
    area_mm2 = ls.GetPhysicalSize(1)
    return float(area_mm2 / 100.0)


def volume_ml_from_3d_mask(mask3d: sitk.Image) -> float:
    mask3d_u8 = sitk.Cast(mask3d, sitk.sitkUInt8)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask3d_u8)
    if not ls.HasLabel(1):
        return float("nan")
    vol_mm3 = ls.GetPhysicalSize(1)
    return float(vol_mm3 / 1000.0)


def roi_height_cm_from_mask_slicecount(mask3d: sitk.Image) -> float:
    arr = sitk.GetArrayViewFromImage(mask3d)  # z, y, x
    present = np.any(arr > 0, axis=(1, 2))
    n_slices = int(present.sum())
    spacing_z_mm = float(mask3d.GetSpacing()[2])
    return float((n_slices * spacing_z_mm) / 10.0)


def count_foreground(mask: sitk.Image) -> int:
    arr = sitk.GetArrayViewFromImage(mask)
    return int(np.sum(arr > 0))


def mean_in_mask(img: sitk.Image, mask: sitk.Image) -> float:
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    mask_u8 = sitk.Cast(mask, sitk.sitkUInt8)
    ls = sitk.LabelStatisticsImageFilter()
    ls.Execute(img_f, mask_u8)
    if not ls.HasLabel(1):
        return float("nan")
    return float(ls.GetMean(1))


def clip_below_zero(img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


# Core computation
def compute_patient(patient_id: str, cfg: PathsConfig, pcfg: ParamConfig) -> Dict[str, object]:
    p_img_dir = cfg.images_root / patient_id
    p_lbl_dir = cfg.labels_root / patient_id

    hu_native = read_image(patient_map_path(p_img_dir, patient_id, pcfg.hu_map), sitk.sitkFloat32)
    fat_native = read_image(patient_map_path(p_img_dir, patient_id, pcfg.fat_map), sitk.sitkFloat32)
    fat_clip = clip_below_zero(fat_native)

    abw_native = read_image(p_lbl_dir / pcfg.abdominal_wall, sitk.sitkInt16)
    para_native = read_image(p_lbl_dir / pcfg.paravertebral, sitk.sitkInt16)
    l3_native = read_image(p_lbl_dir / pcfg.l3_mask, sitk.sitkInt16)

    # Always resample masks to HU image grid, even if already aligned.
    abw = resample_like(abw_native, hu_native, is_label=True)
    para = resample_like(para_native, hu_native, is_label=True)
    l3 = resample_like(l3_native, hu_native, is_label=True)

    # Sanity check after resampling
    if (
        hu_native.GetSize() != abw.GetSize()
        or hu_native.GetSize() != para.GetSize()
        or hu_native.GetSize() != l3.GetSize()
    ):
        raise ValueError(
            f"Resampled mask size mismatch after alignment. "
            f"HU={hu_native.GetSize()}, abw={abw.GetSize()}, para={para.GetSize()}, l3={l3.GetSize()}"
        )

    muscle = combine_muscle_masks(abw, para)
    muscle_thr = apply_hu_threshold_to_mask(muscle, hu_native, pcfg.hu_min, pcfg.hu_max)

    centroid_phys = l3_centroid_phys(l3)
    k = axial_index_from_phys(hu_native, centroid_phys)

    # 2D slices
    hu_2d = extract_axial_slice(hu_native, k)
    fat_2d = extract_axial_slice(fat_clip, k)
    muscle_2d = extract_axial_slice(muscle, k)
    muscle_thr_2d = extract_axial_slice(muscle_thr, k)

    # Geometry / SMI area definitions
    area_cm2_2d = area_cm2_from_2d_mask(muscle_thr_2d)

    volume_ml_3d = volume_ml_from_3d_mask(muscle_thr)
    roi_height_cm_3d = roi_height_cm_from_mask_slicecount(muscle_thr)
    area_cm2_3d = (
        float("nan")
        if (
            not np.isfinite(volume_ml_3d)
            or not np.isfinite(roi_height_cm_3d)
            or roi_height_cm_3d == 0
        )
        else float(volume_ml_3d / roi_height_cm_3d)
    )

    # Density and fat fraction on the HU-thresholded combined muscle mask
    ct_muscle_density_2d = mean_in_mask(hu_2d, muscle_thr_2d)
    ct_muscle_density_3d = mean_in_mask(hu_native, muscle_thr)

    ct_muscle_fat_2d = mean_in_mask(fat_2d, muscle_thr_2d)
    ct_muscle_fat_3d = mean_in_mask(fat_clip, muscle_thr)

    return {
        "patient_id": patient_id,
        "status": "ok",
        "error": "",
        "l3_slice_index": int(k),
        "spacing_x_mm": float(hu_native.GetSpacing()[0]),
        "spacing_y_mm": float(hu_native.GetSpacing()[1]),
        "spacing_z_mm": float(hu_native.GetSpacing()[2]),
        "n_pix_2d_muscle": count_foreground(muscle_2d),
        "n_vox_3d_muscle": count_foreground(muscle),
        "n_pix_2d_muscle_thr": count_foreground(muscle_thr_2d),
        "n_vox_3d_muscle_thr": count_foreground(muscle_thr),
        "area_cm2_2d": float(area_cm2_2d),
        "volume_ml_3d": float(volume_ml_3d),
        "roi_height_cm_3d": float(roi_height_cm_3d),
        "area_cm2_3d": float(area_cm2_3d),
        "ct_muscle_density_2d": float(ct_muscle_density_2d),
        "ct_muscle_density_3d": float(ct_muscle_density_3d),
        "ct_muscle_fat_2d": float(ct_muscle_fat_2d),
        "ct_muscle_fat_3d": float(ct_muscle_fat_3d),
    }


def safe_compute(patient_id: str, cfg: PathsConfig, pcfg: ParamConfig) -> Dict[str, object]:
    try:
        return compute_patient(patient_id, cfg, pcfg)
    except Exception as exc:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "error": repr(exc),
        }


def run_all(
    cfg: PathsConfig = CFG,
    pcfg: ParamConfig = PCFG,
    patient_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    use_ids = list(patient_ids) if patient_ids is not None else list(PATIENT_IDS)
    use_ids = sorted(use_ids, key=natural_sort_key)

    rows: List[Dict[str, object]] = []
    for patient_id in tqdm(use_ids, desc="Computing spectral CT liver parameters"):
        rows.append(safe_compute(patient_id, cfg, pcfg))

    df = pd.DataFrame(rows)

    # Optionally remove known problematic cases if they failed
    if pcfg.optionally_drop_on_failure:
        drop_ids = set(pcfg.optionally_drop_on_failure)
        failed_drop = df["patient_id"].isin(drop_ids) & (df["status"] == "failed")
        df = df.loc[~failed_drop].reset_index(drop=True)

    preferred_cols = [
        "patient_id",
        "status",
        "error",
        "l3_slice_index",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "n_pix_2d_muscle",
        "n_vox_3d_muscle",
        "n_pix_2d_muscle_thr",
        "n_vox_3d_muscle_thr",
        "area_cm2_2d",
        "volume_ml_3d",
        "roi_height_cm_3d",
        "area_cm2_3d",
        "ct_muscle_density_2d",
        "ct_muscle_density_3d",
        "ct_muscle_fat_2d",
        "ct_muscle_fat_3d",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    df = df.loc[:, cols]

    cfg.out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(cfg.out_xlsx, index=False)
    print(f"Saved Excel: {cfg.out_xlsx}")

    if cfg.out_csv is not None:
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)
        print(f"Saved CSV: {cfg.out_csv}")

    return df


if __name__ == "__main__":
    df_out = run_all()
    print(df_out.head())
