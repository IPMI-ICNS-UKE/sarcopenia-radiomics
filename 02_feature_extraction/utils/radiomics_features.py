from typing import Dict, Tuple

import SimpleITK as sitk
from radiomics import featureextractor

from imaging_io import crop_image_to_reference_region, resample_like
from mask import apply_hu_threshold_to_mask, clip_lower, to_binary_mask


def make_radiomics_extractor(params_yaml, mode: str = "2d") -> featureextractor.RadiomicsFeatureExtractor:
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_yaml))
    if str(mode) == "3d":
        # The shipped YAMLs are tuned for slice-wise (force2D) extraction.
        # For volumetric extraction we disable force2D so texture features
        # (firstorder/glcm/glrlm/glszm/ngtdm) are computed over the 3D volume.
        # Bin widths and enabled feature classes are kept unchanged.
        extractor.settings["force2D"] = False
    return extractor


def crop_2d_to_mask_bbox(image2d: sitk.Image, mask2d: sitk.Image, margin_px: int) -> Tuple[sitk.Image, sitk.Image]:
    mask_u8 = sitk.Cast(mask2d > 0, sitk.sitkUInt8)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return image2d, mask_u8

    x, y, w, h = stats.GetBoundingBox(1)
    start = [max(0, x - margin_px), max(0, y - margin_px)]
    end = [
        min(image2d.GetSize()[0], x + w + margin_px),
        min(image2d.GetSize()[1], y + h + margin_px),
    ]

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(start)
    roi.SetSize([end[0] - start[0], end[1] - start[1]])
    return roi.Execute(image2d), roi.Execute(mask_u8)


def crop_3d_to_mask_bbox(
    image3d: sitk.Image,
    mask3d: sitk.Image,
    margin_xyz: Tuple[int, int, int] = (12, 12, 2),
) -> Tuple[sitk.Image, sitk.Image]:
    mask_u8 = sitk.Cast(mask3d > 0, sitk.sitkUInt8)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return image3d, mask_u8

    x, y, z, w, h, d = stats.GetBoundingBox(1)
    mx, my, mz = margin_xyz
    size_img = image3d.GetSize()
    start = [max(0, x - mx), max(0, y - my), max(0, z - mz)]
    end = [
        min(size_img[0], x + w + mx),
        min(size_img[1], y + h + my),
        min(size_img[2], z + d + mz),
    ]

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(start)
    roi.SetSize([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    return roi.Execute(image3d), roi.Execute(mask_u8)


def preprocess_map(map_name: str, img: sitk.Image) -> sitk.Image:
    if map_name in {"musclefat", "iodine"}:
        return clip_lower(sitk.Cast(img, sitk.sitkFloat32), lower=0.0)
    return sitk.Cast(img, sitk.sitkFloat32)


# Backward-compatible alias (clip is dimension-agnostic).
def preprocess_map_2d(map_name: str, img2d: sitk.Image) -> sitk.Image:
    return preprocess_map(map_name, img2d)


def clean_radiomics_output(result: Dict[str, object], map_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        col = f"{key}_{map_name}"
        try:
            out[col] = float(value)
        except Exception:
            continue
    return out


def compute_radiomics_on_slice(
    patient_id: str,
    cohort_name: str,
    ct_2d: sitk.Image,
    image_2d: sitk.Image,
    map_name: str,
    mask_2d_ct_grid: sitk.Image,
    mask_name: str,
    slice_name: str,
    slice_index_ct: int,
    hu_min: float,
    hu_max: float,
    crop_margin_px: int,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> Dict[str, object]:
    mask_thr_ct_2d = apply_hu_threshold_to_mask(mask_2d_ct_grid, ct_2d, hu_min=hu_min, hu_max=hu_max)
    mask_thr_img_2d = resample_like(mask_thr_ct_2d, image_2d, is_label=True)
    mask_thr_img_2d = sitk.Cast(to_binary_mask(mask_thr_img_2d), sitk.sitkUInt8)
    image_2d = preprocess_map(map_name, image_2d)

    n_pix_mask_2d = int(sitk.GetArrayViewFromImage(sitk.Cast(mask_2d_ct_grid, sitk.sitkUInt8)).sum())
    n_pix_mask_2d_thr_ct = int(sitk.GetArrayViewFromImage(mask_thr_ct_2d).sum())
    n_pix_mask_2d_thr_img = int(sitk.GetArrayViewFromImage(mask_thr_img_2d).sum())

    row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "slice_name": slice_name,
        "slice_index_ct": int(slice_index_ct),
        "hu_min": float(hu_min) if hu_min is not None else None,
        "hu_max": float(hu_max) if hu_max is not None else None,
        "n_pix_mask_2d": n_pix_mask_2d,
        "n_pix_mask_2d_thr_ct": n_pix_mask_2d_thr_ct,
        "n_pix_mask_2d_thr_img": n_pix_mask_2d_thr_img,
        "map_name": map_name,
        "img_spacing_x_mm": float(image_2d.GetSpacing()[0]),
        "img_spacing_y_mm": float(image_2d.GetSpacing()[1]),
        "error": "",
    }

    if n_pix_mask_2d_thr_img == 0:
        row["status"] = "failed"
        row["error"] = "Empty HU-thresholded mask on image grid."
        return row

    image_crop, mask_crop = crop_2d_to_mask_bbox(image_2d, mask_thr_img_2d, margin_px=crop_margin_px)
    rad_result = extractor.execute(image_crop, mask_crop)
    row.update(clean_radiomics_output(rad_result, map_name=map_name))
    return row


def compute_radiomics_3d(
    patient_id: str,
    cohort_name: str,
    ct_3d: sitk.Image,
    image_3d: sitk.Image,
    map_name: str,
    mask_3d_ct_grid: sitk.Image,
    mask_name: str,
    hu_min: float,
    hu_max: float,
    crop_margin_px: int,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> Dict[str, object]:
    """Volumetric counterpart of compute_radiomics_on_slice.

    For speed, every step is restricted to the muscle region. ``mask_3d_ct_grid``
    is expected to be the bbox-cropped muscle volume (see
    ``automated_3d_mask_variants``). The CT is resampled onto that small mask
    grid for HU thresholding, and the spectral map is cropped to the same
    physical region (at its native resolution) before the mask is resampled
    onto it. Features are then computed over the cropped volume.
    """
    # CT restricted to the mask sub-grid -> guaranteed co-grid, cheap HU threshold.
    ct_on_mask = resample_like(ct_3d, mask_3d_ct_grid, is_label=False)
    mask_thr_ct_3d = apply_hu_threshold_to_mask(mask_3d_ct_grid, ct_on_mask, hu_min=hu_min, hu_max=hu_max)

    # Spectral map restricted to the muscle region at its native resolution.
    img_region = crop_image_to_reference_region(image_3d, mask_3d_ct_grid, margin_xyz=(2, 2, 2))
    img_region = preprocess_map(map_name, img_region)

    mask_thr_img_3d = resample_like(mask_thr_ct_3d, img_region, is_label=True)
    mask_thr_img_3d = sitk.Cast(to_binary_mask(mask_thr_img_3d), sitk.sitkUInt8)

    n_pix_mask_3d = int(sitk.GetArrayViewFromImage(sitk.Cast(mask_3d_ct_grid, sitk.sitkUInt8)).sum())
    n_pix_mask_3d_thr_ct = int(sitk.GetArrayViewFromImage(mask_thr_ct_3d).sum())
    n_pix_mask_3d_thr_img = int(sitk.GetArrayViewFromImage(mask_thr_img_3d).sum())

    row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "hu_min": float(hu_min) if hu_min is not None else None,
        "hu_max": float(hu_max) if hu_max is not None else None,
        "n_pix_mask_3d": n_pix_mask_3d,
        "n_pix_mask_3d_thr_ct": n_pix_mask_3d_thr_ct,
        "n_pix_mask_3d_thr_img": n_pix_mask_3d_thr_img,
        "map_name": map_name,
        "img_spacing_x_mm": float(img_region.GetSpacing()[0]),
        "img_spacing_y_mm": float(img_region.GetSpacing()[1]),
        "img_spacing_z_mm": float(img_region.GetSpacing()[2]),
        "error": "",
    }

    if n_pix_mask_3d_thr_img == 0:
        row["status"] = "failed"
        row["error"] = "Empty HU-thresholded mask on image grid."
        return row

    image_crop, mask_crop = crop_3d_to_mask_bbox(
        img_region, mask_thr_img_3d, margin_xyz=(crop_margin_px, crop_margin_px, 2)
    )
    rad_result = extractor.execute(image_crop, mask_crop)
    row.update(clean_radiomics_output(rad_result, map_name=map_name))
    return row
