from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from config import ImagingConfig, PathsConfig


# Low-level image I/O  (mirrors imaging_io.py patterns)
def _read(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return sitk.ReadImage(str(path), pixel_type)


def _resample_like(moving: sitk.Image, fixed: sitk.Image, *, is_label: bool) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out = sitk.Resample(
        moving,
        fixed,
        sitk.Transform(),
        interp,
        0.0,
        moving.GetPixelID(),
    )
    return sitk.Cast(out, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)


def _to_binary(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.NotEqual(img, 0), sitk.sitkUInt8)


# L3 centroid helpers  (mirrors mask.py patterns)
def _l3_binary(l3_img: sitk.Image) -> sitk.Image:
    """Decode the L3 vertebra mask that may use -1023/1 or negative encoding."""
    l3 = sitk.Cast(l3_img, sitk.sitkInt32)
    is_neg1023 = sitk.Equal(l3, -1023)
    is_one = sitk.Equal(l3, 1)
    candidate = sitk.Or(is_neg1023, is_one)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk.Cast(candidate, sitk.sitkUInt8))
    if stats.GetSum() > 0:
        return sitk.Cast(candidate, sitk.sitkUInt8)

    # fallback: any negative voxel that is not -1024 background
    neg = sitk.And(sitk.Greater(l3, -2000), sitk.Less(l3, 0))
    not_bg = sitk.Not(sitk.Equal(l3, -1024))
    neg = sitk.And(neg, not_bg)
    stats.Execute(sitk.Cast(neg, sitk.sitkUInt8))
    if stats.GetSum() == 0:
        raise ValueError("Could not identify L3 voxels from vertebra_l3 mask.")
    return sitk.Cast(neg, sitk.sitkUInt8)


def _l3_k_index(ct: sitk.Image, l3_mask: sitk.Image) -> int:
    """Return the axial (k) slice index of the L3 vertebra centre-of-mass."""
    l3_bin = _l3_binary(l3_mask)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(l3_bin)
    if not ls.HasLabel(1):
        raise ValueError("L3 binary mask contains no foreground label.")
    centroid_phys = ls.GetCentroid(1)
    ijk = ct.TransformPhysicalPointToIndex(centroid_phys)
    return int(np.clip(ijk[2], 0, ct.GetSize()[2] - 1))


# Slice extraction
def _extract_slice_2d(img: sitk.Image, k: int) -> sitk.Image:
    size = list(img.GetSize())
    size[2] = 0
    index = [0, 0, int(k)]
    return sitk.Extract(img, size=size, index=index)


def _sitk_to_np(img: sitk.Image) -> np.ndarray:
    """Return a 2-D (or 3-D) numpy array from a SimpleITK image."""
    return sitk.GetArrayFromImage(img)


# HU windowing for display
def _apply_window(arr: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# Patient image context
@dataclass
class PatientImageContext:
    """All 2-D axial arrays at L3 level for one patient, ready for plotting."""

    patient_id: str

    # Display arrays (float32, range 0-1 for CT; raw values for MuscleFat)
    ct_display: np.ndarray  # CT windowed to [0, 1]
    musclefat_raw: np.ndarray  # MuscleFat values within muscle mask (NaN outside)

    # Binary masks at L3 (on CT grid and on MuscleFat grid)
    muscle_mask_on_ct: np.ndarray  # uint8, shape of CT slice
    muscle_mask_on_spectral: np.ndarray  # uint8, shape of MuscleFat slice

    # Axial index used
    k_l3: int


def load_patient_images(
    patient_id: str,
    cohort_label: str,
    paths_cfg: PathsConfig,
    img_cfg: ImagingConfig,
) -> PatientImageContext:
    """
    Load and pre-process all images for one patient at the L3 level.

    Parameters
    ----------
    patient_id   : e.g. "Pat58"
    cohort_label : "cohort1" or "cohort2"
    paths_cfg    : PathsConfig instance
    img_cfg      : ImagingConfig instance
    """
    is_c1 = cohort_label == "cohort1"
    img_root = paths_cfg.images_root_cohort1 if is_c1 else paths_cfg.images_root_cohort2
    labels_root = paths_cfg.labels_root_cohort1 if is_c1 else paths_cfg.labels_root_cohort2

    patient_img_dir = img_root / patient_id
    patient_labels_dir = labels_root / patient_id

    # ---- Load images -------------------------------------------------------
    ct = _read(patient_img_dir / f"{patient_id}-HU.nii.gz")
    mf = _read(patient_img_dir / f"{patient_id}-MuscleFat.nii.gz")

    # ---- Load and resample masks to CT grid --------------------------------
    abw_raw = _read(patient_labels_dir / img_cfg.abdominal_wall_mask, sitk.sitkInt16)
    para_raw = _read(patient_labels_dir / img_cfg.paravertebral_mask, sitk.sitkInt16)
    l3_raw = _read(patient_labels_dir / img_cfg.l3_mask, sitk.sitkInt16)

    abw = _resample_like(abw_raw, ct, is_label=True)
    para = _resample_like(para_raw, ct, is_label=True)
    l3 = _resample_like(l3_raw, ct, is_label=True)

    # ---- Merge muscle mask (ABW + paravertebral) ---------------------------
    muscle_3d = sitk.Or(_to_binary(abw), _to_binary(para))

    # ---- L3 axial slice index ----------------------------------------------
    k_l3 = _l3_k_index(ct, l3)

    # ---- 2-D slice of CT and muscle mask at L3 -----------------------------
    ct_2d = _extract_slice_2d(ct, k_l3)
    muscle_2d = _extract_slice_2d(muscle_3d, k_l3)

    # ---- HU-based muscle refinement on CT slice ----------------------------
    hu_ok = sitk.Cast(
        sitk.And(
            sitk.GreaterEqual(ct_2d, img_cfg.hu_min),
            sitk.LessEqual(ct_2d, img_cfg.hu_max),
        ),
        sitk.sitkUInt8,
    )
    # Add an option to skip HU-based refinement and use the full muscle mask for resampling:
    if img_cfg.skip_hu_refinement:
        refined_ct_2d = sitk.Cast(muscle_2d, sitk.sitkUInt8)
    else:
        refined_ct_2d = sitk.And(sitk.Cast(muscle_2d, sitk.sitkUInt8), hu_ok)

    # ---- Resample refined mask to the MuscleFat map grid --------------------
    # Use 3-D versions for resampling, then extract slice at equivalent level
    if img_cfg.skip_hu_refinement:
        muscle_3d_refined = muscle_3d
    else:
        muscle_3d_refined = _build_refined_3d_mask(muscle_3d, ct, img_cfg)

    # Resample 3-D refined mask to the MuscleFat grid
    spectral_ref = mf  # use MuscleFat as reference grid
    muscle_spec_3d = _resample_like(
        sitk.Cast(muscle_3d_refined, sitk.sitkUInt8), spectral_ref, is_label=True
    )

    # Find k index in MuscleFat space that matches the L3 physical centroid
    centroid_phys = _l3_centroid_phys_from_mask(l3)
    k_l3_spec = _phys_to_k(spectral_ref, centroid_phys)

    # ---- 2-D MuscleFat slice at L3 -----------------------------------------
    mf_2d = _extract_slice_2d(mf, k_l3_spec)
    msk_2d = _extract_slice_2d(muscle_spec_3d, k_l3_spec)

    # ---- Convert to numpy --------------------------------------------------
    ct_arr = _sitk_to_np(ct_2d).astype(np.float32)
    mf_arr = _sitk_to_np(mf_2d).astype(np.float32)
    msk_arr = _sitk_to_np(msk_2d).astype(np.uint8)
    ct_msk = _sitk_to_np(refined_ct_2d).astype(np.uint8)

    # ---- Clip MuscleFat map --------------------------------------------------
    mf_arr = np.maximum(mf_arr, img_cfg.spectral_clip_min)

    # ---- Mask MuscleFat values outside muscle region (set to NaN) ----------
    mf_masked = np.where(msk_arr > 0, mf_arr, np.nan)

    # ---- CT display (windowed) ----------------------------------------------
    ct_display = _apply_window(ct_arr, img_cfg.ct_wl, img_cfg.ct_ww)

    return PatientImageContext(
        patient_id=patient_id,
        ct_display=ct_display,
        musclefat_raw=mf_masked,
        muscle_mask_on_ct=ct_msk,
        muscle_mask_on_spectral=msk_arr,
        k_l3=k_l3,
    )


# Internal helpers
def _build_refined_3d_mask(
    muscle_3d: sitk.Image,
    ct: sitk.Image,
    img_cfg: ImagingConfig,
) -> sitk.Image:
    """Apply HU thresholding to the full 3-D muscle mask using CT intensities."""
    hu_ok = sitk.Cast(
        sitk.And(
            sitk.GreaterEqual(ct, img_cfg.hu_min),
            sitk.LessEqual(ct, img_cfg.hu_max),
        ),
        sitk.sitkUInt8,
    )
    return sitk.And(sitk.Cast(muscle_3d, sitk.sitkUInt8), hu_ok)


def _l3_centroid_phys_from_mask(l3_mask: sitk.Image):
    """Return physical XYZ centroid of L3 vertebra."""
    l3_bin = _l3_binary(l3_mask)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(l3_bin)
    if not ls.HasLabel(1):
        raise ValueError("L3 binary mask has no foreground after decoding.")
    return ls.GetCentroid(1)


def _phys_to_k(img: sitk.Image, phys_xyz) -> int:
    """Convert a physical point to the axial (k) slice index of *img*."""
    ijk = img.TransformPhysicalPointToIndex(phys_xyz)
    return int(np.clip(ijk[2], 0, img.GetSize()[2] - 1))
