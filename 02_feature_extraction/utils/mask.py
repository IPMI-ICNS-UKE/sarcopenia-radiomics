from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion

from common_config import CohortConfig, CommonRunConfig
from imaging_io import read_ct_nifti, read_image, resample_like


def to_binary_mask(img: sitk.Image) -> sitk.Image:
    return sitk.Cast(sitk.NotEqual(img, 0), sitk.sitkUInt8)


def combine_muscle_masks(abw: sitk.Image, para: sitk.Image) -> sitk.Image:
    return sitk.Or(to_binary_mask(abw), to_binary_mask(para))


def l3_binary_mask_from_encoded(l3_img: sitk.Image) -> sitk.Image:
    l3 = sitk.Cast(l3_img, sitk.sitkInt32)

    is_neg1023 = sitk.Equal(l3, -1023)
    is_one = sitk.Equal(l3, 1)
    is_l3 = sitk.Or(is_neg1023, is_one)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(sitk.Cast(is_l3, sitk.sitkUInt8))
    if stats.GetSum() > 0:
        return sitk.Cast(is_l3, sitk.sitkUInt8)

    neg = sitk.And(sitk.Greater(l3, -2000), sitk.Less(l3, 0))
    not_bg = sitk.Not(sitk.Equal(l3, -1024))
    neg = sitk.And(neg, not_bg)

    stats.Execute(sitk.Cast(neg, sitk.sitkUInt8))
    if stats.GetSum() == 0:
        raise ValueError("Could not identify any L3 voxels from vertebra_l3 mask.")

    return sitk.Cast(neg, sitk.sitkUInt8)


def l3_centroid_phys(l3_mask_img: sitk.Image) -> Tuple[float, float, float]:
    l3_bin = l3_binary_mask_from_encoded(l3_mask_img)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(l3_bin)
    if not ls.HasLabel(1):
        raise ValueError("L3 binary mask has no foreground.")
    return ls.GetCentroid(1)


def axial_index_from_phys(img: sitk.Image, phys_point_xyz: Tuple[float, float, float]) -> int:
    ijk = img.TransformPhysicalPointToIndex(phys_point_xyz)
    return int(np.clip(ijk[2], 0, img.GetSize()[2] - 1))


def extract_axial_slice(img: sitk.Image, k: int) -> sitk.Image:
    size = list(img.GetSize())
    size[2] = 0
    index = [0, 0, int(k)]
    return sitk.Extract(img, size=size, index=index)


def get_nonempty_slice_indices(mask_3d: sitk.Image) -> List[int]:
    arr = sitk.GetArrayFromImage(sitk.Cast(mask_3d, sitk.sitkUInt8))
    return np.where(arr.reshape(arr.shape[0], -1).sum(axis=1) > 0)[0].tolist()


def perturb_mask_2d(mask_2d: sitk.Image, mode: str, radius: int) -> sitk.Image:
    arr = sitk.GetArrayFromImage(sitk.Cast(mask_2d, sitk.sitkUInt8)).astype(bool)

    if mode == "original":
        out = arr
    elif mode == "dilate":
        out = binary_dilation(arr, iterations=radius)
    elif mode == "erode":
        out = binary_erosion(arr, iterations=radius)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(mask_2d)
    return sitk.Cast(out_img, sitk.sitkUInt8)


def perturb_mask_3d(mask_3d: sitk.Image, mode: str, radius: int) -> sitk.Image:
    """3D counterpart of perturb_mask_2d. Morphology runs on the whole volume."""
    arr = sitk.GetArrayFromImage(sitk.Cast(mask_3d, sitk.sitkUInt8)).astype(bool)

    if mode == "original":
        out = arr
    elif mode == "dilate":
        out = binary_dilation(arr, iterations=radius)
    elif mode == "erode":
        out = binary_erosion(arr, iterations=radius)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    out_img = sitk.GetImageFromArray(out.astype(np.uint8))
    out_img.CopyInformation(mask_3d)
    return sitk.Cast(out_img, sitk.sitkUInt8)


def apply_hu_threshold_to_mask(
    mask_2d: sitk.Image, ct_2d: sitk.Image, hu_min: float, hu_max: float
) -> sitk.Image:
    if hu_min is None or hu_max is None:
        print("No HU thresholds provided, skipping HU-based masking.")
        return sitk.Cast(mask_2d, sitk.sitkUInt8)
    else:
        hu_ok = sitk.Cast(
            sitk.And(sitk.GreaterEqual(ct_2d, hu_min), sitk.LessEqual(ct_2d, hu_max)),
            sitk.sitkUInt8,
        )
        return sitk.And(sitk.Cast(mask_2d, sitk.sitkUInt8), hu_ok)


def clip_lower(img: sitk.Image, lower: float) -> sitk.Image:
    return sitk.Maximum(sitk.Cast(img, sitk.sitkFloat32), lower)


def read_manual_pair_if_exists(labels_dir: Path, fname1: str, fname2: str) -> Optional[sitk.Image]:
    p1 = labels_dir / fname1
    p2 = labels_dir / fname2
    if not p1.exists() or not p2.exists():
        return None

    m1 = read_image(p1, sitk.sitkInt16)
    m2 = read_image(p2, sitk.sitkInt16)
    return combine_muscle_masks(m1, m2)


def build_base_patient_context(
    patient_id: str, cohort_cfg: CohortConfig, run_cfg: CommonRunConfig
) -> Dict[str, object]:
    labels_dir = cohort_cfg.labels_root / patient_id
    ct = read_ct_nifti(
        patient_id=patient_id, cohort_cfg=cohort_cfg, run_cfg=run_cfg, pixel_type=sitk.sitkFloat32
    )

    abw = resample_like(
        read_image(labels_dir / run_cfg.abdominal_wall, sitk.sitkInt16), ct, is_label=True
    )
    para = resample_like(
        read_image(labels_dir / run_cfg.paravertebral, sitk.sitkInt16), ct, is_label=True
    )
    l3 = resample_like(read_image(labels_dir / run_cfg.l3_mask, sitk.sitkInt16), ct, is_label=True)

    muscle = combine_muscle_masks(abw, para)
    k_l3 = axial_index_from_phys(ct, l3_centroid_phys(l3))

    return {
        "ct": ct,
        "muscle": muscle,
        "k_l3": k_l3,
        "labels_dir": labels_dir,
    }


def automated_l3_mask_variants(
    muscle_3d: sitk.Image, k_l3: int
) -> List[Tuple[str, str, int, sitk.Image]]:
    muscle_2d = sitk.Cast(extract_axial_slice(muscle_3d, k_l3), sitk.sitkUInt8)
    variants = [
        ("a_original", "original", 0),
        ("a_dilate_p1", "dilate", 1),
        ("a_dilate_p2", "dilate", 2),
        ("a_erode_p1", "erode", 1),
        ("a_erode_p2", "erode", 2),
    ]
    return [
        (mask_name, "l3_com", int(k_l3), perturb_mask_2d(muscle_2d, mode=mode, radius=radius))
        for mask_name, mode, radius in variants
    ]


def crop_mask_to_self_bbox_3d(mask_3d: sitk.Image, margin_xyz: Tuple[int, int, int]) -> sitk.Image:
    """Crop a binary 3D mask to its own foreground bounding box plus margin.

    The crop preserves physical space (only the origin shifts), so the cropped
    mask stays spatially consistent with the CT and spectral maps. Returns the
    (binarized) input unchanged if it has no foreground.
    """
    mask_u8 = sitk.Cast(to_binary_mask(mask_3d), sitk.sitkUInt8)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    if not stats.HasLabel(1):
        return mask_u8

    x, y, z, w, h, d = stats.GetBoundingBox(1)
    mx, my, mz = margin_xyz
    size_img = mask_u8.GetSize()
    start = [max(0, x - mx), max(0, y - my), max(0, z - mz)]
    end = [
        min(size_img[0], x + w + mx),
        min(size_img[1], y + h + my),
        min(size_img[2], z + d + mz),
    ]

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(start)
    roi.SetSize([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    return roi.Execute(mask_u8)


def automated_3d_mask_variants(
    muscle_3d: sitk.Image,
    margin_xyz: Tuple[int, int, int] = (12, 12, 2),
    use_original: bool = False,
) -> List[Tuple[str, sitk.Image]]:
    """3D counterpart of automated_l3_mask_variants.

    Returns (mask_name, mask_3d) pairs for the muscle volume and its
    morphological perturbations, used for 3D feature robustness analysis.

    The muscle volume is first cropped to its bounding box (plus ``margin_xyz``,
    in voxels) so all downstream 3D operations run on a small sub-volume instead
    of the full scan. The crop keeps physical space, so the cropped masks remain
    spatially consistent with the CT and the spectral maps. ``margin_xyz`` must
    be at least the largest perturbation radius on each axis; the default
    (12, 12, 2) leaves room for dilation up to radius 2.
    """
    muscle_cropped = crop_mask_to_self_bbox_3d(muscle_3d, margin_xyz=margin_xyz)

    if use_original:
        variants = [
            ("a_original", "original", 0),
        ]
    else:
        variants = [
            ("a_original", "original", 0),
            ("a_dilate_p1", "dilate", 1),
            ("a_dilate_p2", "dilate", 2),
            ("a_erode_p1", "erode", 1),
            ("a_erode_p2", "erode", 2),
        ]
    return [
        (mask_name, perturb_mask_3d(muscle_cropped, mode=mode, radius=radius))
        for mask_name, mode, radius in variants
    ]


def manual_and_model_masksets(
    cohort_name: str,
    ct: sitk.Image,
    model_muscle_3d: sitk.Image,
    labels_dir: Path,
    run_cfg: CommonRunConfig,
    manual_annotation_flag: int,
) -> List[Tuple[str, str, int, sitk.Image]]:
    if cohort_name != "cohort1" or int(manual_annotation_flag) != 1:
        return []

    isabel_native = read_manual_pair_if_exists(
        labels_dir, run_cfg.abdominal_wall_i, run_cfg.paravertebral_i
    )
    jenni_native = read_manual_pair_if_exists(
        labels_dir, run_cfg.abdominal_wall_j, run_cfg.paravertebral_j
    )
    if isabel_native is None or jenni_native is None:
        return []

    isabel = resample_like(isabel_native, ct, is_label=True)
    jenni = resample_like(jenni_native, ct, is_label=True)

    shared_idx = sorted(
        set(get_nonempty_slice_indices(isabel)).intersection(get_nonempty_slice_indices(jenni))
    )
    if len(shared_idx) != 3:
        return []

    out: List[Tuple[str, str, int, sitk.Image]] = []
    for level_name, idx_ct in zip(("l2", "l3", "l4"), shared_idx):
        out.append(
            (f"i_{level_name}", level_name, int(idx_ct), extract_axial_slice(isabel, idx_ct))
        )
        out.append((f"j_{level_name}", level_name, int(idx_ct), extract_axial_slice(jenni, idx_ct)))
        out.append(
            (
                f"a_{level_name}",
                level_name,
                int(idx_ct),
                extract_axial_slice(model_muscle_3d, idx_ct),
            )
        )
    return out
