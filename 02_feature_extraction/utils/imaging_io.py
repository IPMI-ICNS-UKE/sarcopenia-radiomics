import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from common_config import CohortConfig, CommonRunConfig


def load_ground_truth(
    cohort_cfg: CohortConfig, run_cfg: CommonRunConfig, required_columns=None
) -> pd.DataFrame:
    df = pd.read_excel(cohort_cfg.table_path)

    for col, val in run_cfg.filters:
        if col not in df.columns:
            raise KeyError(f"Required filter column missing: {col}")
        df = df[df[col] == val]

    required = ["Pat ID"]
    if required_columns:
        required.extend(required_columns)

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")

    return df.reset_index(drop=True)


def natural_patient_sort(patient_ids: List[str]) -> List[str]:
    def key(s: str):
        m = re.search(r"(\d+)", str(s))
        return (int(m.group(1)) if m else 0, str(s))

    return sorted(patient_ids, key=key)


def read_image(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return sitk.ReadImage(str(path), pixel_type)


def read_ct_nifti(
    patient_id: str, cohort_cfg: CohortConfig, run_cfg: CommonRunConfig, pixel_type=sitk.sitkFloat32
) -> sitk.Image:
    return read_image(
        cohort_cfg.images_root / patient_id / f"{patient_id}{run_cfg.ct_nifti_suffix}", pixel_type
    )


def same_grid(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    if np.max(np.abs(np.array(a.GetSpacing()) - np.array(b.GetSpacing()))) > tol:
        return False
    if np.max(np.abs(np.array(a.GetOrigin()) - np.array(b.GetOrigin()))) > tol:
        return False
    if np.max(np.abs(np.array(a.GetDirection()) - np.array(b.GetDirection()))) > tol:
        return False
    return True


def resample_like(
    moving: sitk.Image, fixed: sitk.Image, is_label: bool, default_value: float = 0.0
) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out = sitk.Resample(
        moving,
        fixed,
        sitk.Transform(),
        interpolator,
        default_value,
        moving.GetPixelID(),
    )
    return sitk.Cast(out, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)


def crop_image_to_reference_region(
    image: sitk.Image,
    reference: sitk.Image,
    margin_xyz: Tuple[int, int, int] = (0, 0, 0),
) -> sitk.Image:
    """Crop ``image`` to the physical bounding box of ``reference``.

    Works across different grids: the eight physical corners of ``reference``
    are mapped into ``image`` index space, and the axis-aligned index bounding
    box (expanded by ``margin_xyz``, measured in ``image`` voxels) is cropped
    out. Spacing and direction of ``image`` are preserved; only the origin is
    shifted to the cropped sub-region. If there is no overlap, the original
    image is returned unchanged.
    """
    rx, ry, rz = reference.GetSize()
    corner_indices = [
        (0, 0, 0),
        (rx - 1, 0, 0),
        (0, ry - 1, 0),
        (0, 0, rz - 1),
        (rx - 1, ry - 1, 0),
        (rx - 1, 0, rz - 1),
        (0, ry - 1, rz - 1),
        (rx - 1, ry - 1, rz - 1),
    ]
    idx = [
        image.TransformPhysicalPointToIndex(reference.TransformIndexToPhysicalPoint(c))
        for c in corner_indices
    ]
    xs = [p[0] for p in idx]
    ys = [p[1] for p in idx]
    zs = [p[2] for p in idx]

    mx, my, mz = margin_xyz
    size_img = image.GetSize()
    start = [
        max(0, min(xs) - mx),
        max(0, min(ys) - my),
        max(0, min(zs) - mz),
    ]
    end = [
        min(size_img[0], max(xs) + mx + 1),
        min(size_img[1], max(ys) + my + 1),
        min(size_img[2], max(zs) + mz + 1),
    ]
    size = [end[i] - start[i] for i in range(3)]
    if min(size) <= 0:
        return image

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([int(s) for s in start])
    roi.SetSize([int(s) for s in size])
    return roi.Execute(image)


def add_repo_to_syspath(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def get_torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def axial_index_from_reference_to_target(
    reference: sitk.Image, target: sitk.Image, k_reference: int
) -> int:
    from mask import axial_index_from_phys

    phys = reference.TransformIndexToPhysicalPoint((0, 0, int(k_reference)))
    return axial_index_from_phys(target, phys)
