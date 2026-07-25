from typing import Dict, NamedTuple

import pandas as pd
import SimpleITK as sitk

import config
from imaging_io import (
    axial_index_from_reference_to_target,
    load_ground_truth,
    natural_patient_sort,
    read_image,
)
from mask import build_base_patient_context, extract_axial_slice, manual_and_model_masksets


class PatientL3Data(NamedTuple):
    ct_2d: sitk.Image
    musclefat_2d: sitk.Image
    masks_by_rater: Dict[str, sitk.Image]  # {"i": ..., "j": ..., "a": ...}
    height_m: float


def get_manual_annotation_cohort() -> pd.DataFrame:
    """Return the cohort-1 ground-truth rows for the manual-annotation subset."""
    df_gt = load_ground_truth(
        config.COHORT_CFG, config.RUN_CFG, required_columns=["manual_annotation", "height"]
    )
    df_gt = df_gt[df_gt["manual_annotation"] == 1].copy()
    df_gt["Pat ID"] = df_gt["Pat ID"].astype(str)

    ordered_ids = natural_patient_sort(df_gt["Pat ID"].tolist())
    return df_gt.set_index("Pat ID").loc[ordered_ids].reset_index()


def get_patient_l3_data(patient_id: str, height_m: float) -> PatientL3Data:
    """Build the L3 CT slice, muscle-fat slice, and per-rater masks for one patient."""
    ctx = build_base_patient_context(patient_id, config.COHORT_CFG, config.RUN_CFG)

    masksets = manual_and_model_masksets(
        cohort_name=config.COHORT_CFG.cohort_name,
        ct=ctx["ct"],
        model_muscle_3d=ctx["muscle"],
        labels_dir=ctx["labels_dir"],
        run_cfg=config.RUN_CFG,
        manual_annotation_flag=1,
    )
    l3_masksets = [entry for entry in masksets if entry[1] == config.LEVEL]
    if len(l3_masksets) != len(config.RATERS):
        raise ValueError(
            f"[{patient_id}] Expected {len(config.RATERS)} L3 masks, found {len(l3_masksets)}."
        )

    slice_index_ct = l3_masksets[0][2]
    masks_by_rater = {
        mask_name.split("_", 1)[0]: mask_2d for mask_name, _level, _idx, mask_2d in l3_masksets
    }

    ct_2d = extract_axial_slice(ctx["ct"], slice_index_ct)

    musclefat_path = (
        config.COHORT_CFG.images_root / patient_id / f"{patient_id}{config.MUSCLEFAT_FILE_SUFFIX}"
    )
    musclefat_3d = read_image(musclefat_path, sitk.sitkFloat32)
    idx_musclefat = axial_index_from_reference_to_target(ctx["ct"], musclefat_3d, slice_index_ct)
    musclefat_2d = extract_axial_slice(musclefat_3d, idx_musclefat)

    return PatientL3Data(
        ct_2d=ct_2d,
        musclefat_2d=musclefat_2d,
        masks_by_rater=masks_by_rater,
        height_m=height_m,
    )
