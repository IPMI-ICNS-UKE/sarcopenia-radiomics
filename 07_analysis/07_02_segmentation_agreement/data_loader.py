from typing import Dict, Tuple

import SimpleITK as sitk

import config
from imaging_io import load_ground_truth, natural_patient_sort
from mask import build_base_patient_context, extract_axial_slice, manual_and_model_masksets

MaskByReader = Dict[str, sitk.Image]  # {"i": ..., "j": ..., "a": ...}
MaskByLevel = Dict[str, MaskByReader]  # {"l2": ..., "l3": ..., "l4": ...}


def get_manual_annotation_patients() -> list[str]:
    """Return the cohort 1 patient IDs with a manual annotation subset."""
    df_gt = load_ground_truth(
        config.COHORT_CFG, config.RUN_CFG, required_columns=["manual_annotation"]
    )
    df_gt = df_gt[df_gt["manual_annotation"] == 1]
    return natural_patient_sort(df_gt["Pat ID"].astype(str).tolist())


def get_patient_masks_by_level(patient_id: str) -> MaskByLevel:
    """Build {level: {"i": mask_2d, "j": mask_2d, "a": mask_2d}} for one patient."""
    ctx = build_base_patient_context(patient_id, config.COHORT_CFG, config.RUN_CFG)

    masksets = manual_and_model_masksets(
        cohort_name=config.COHORT_CFG.cohort_name,
        ct=ctx["ct"],
        model_muscle_3d=ctx["muscle"],
        labels_dir=ctx["labels_dir"],
        run_cfg=config.RUN_CFG,
        manual_annotation_flag=1,
    )

    by_level: MaskByLevel = {level: {} for level in config.LEVELS}
    for mask_name, level_name, _slice_index, mask_2d in masksets:
        reader = mask_name.split("_", 1)[0]  # "i", "j", or "a"
        by_level[level_name][reader] = mask_2d

    return by_level


def get_patient_ct_and_masks_by_level(
    patient_id: str,
) -> Dict[str, Tuple[sitk.Image, Dict[str, sitk.Image]]]:
    """Return {level: (ct_2d, {"i": mask_2d, "j": mask_2d, "a": mask_2d})} for one patient.

    Used for the qualitative segmentation-overlay figures (one CT slice per
    level, with the three rater masks on that same slice).
    """
    ctx = build_base_patient_context(patient_id, config.COHORT_CFG, config.RUN_CFG)

    masksets = manual_and_model_masksets(
        cohort_name=config.COHORT_CFG.cohort_name,
        ct=ctx["ct"],
        model_muscle_3d=ctx["muscle"],
        labels_dir=ctx["labels_dir"],
        run_cfg=config.RUN_CFG,
        manual_annotation_flag=1,
    )

    out: Dict[str, Tuple[sitk.Image, Dict[str, sitk.Image]]] = {}
    for level in config.LEVELS:
        level_masksets = [entry for entry in masksets if entry[1] == level]
        if len(level_masksets) != 3:
            raise ValueError(
                f"[{patient_id}] Expected 3 masks at level {level}, found {len(level_masksets)}."
            )

        slice_index_ct = level_masksets[0][2]
        ct_2d = extract_axial_slice(ctx["ct"], slice_index_ct)
        masks_by_rater = {
            mask_name.split("_", 1)[0]: mask_2d
            for mask_name, _level, _idx, mask_2d in level_masksets
        }
        out[level] = (ct_2d, masks_by_rater)

    return out


def load_all_patient_masks() -> Dict[str, MaskByLevel]:
    """Return {patient_id: {level: {reader: mask_2d}}} for every eligible patient."""
    patient_ids = get_manual_annotation_patients()

    all_masks: Dict[str, MaskByLevel] = {}
    for patient_id in patient_ids:
        by_level = get_patient_masks_by_level(patient_id)
        missing = [
            level
            for level in config.LEVELS
            if not all(k in by_level[level] for k in ("i", "j", "a"))
        ]
        if missing:
            raise ValueError(
                f"[{patient_id}] Missing reader/automated masks at level(s) {missing}."
            )
        all_masks[patient_id] = by_level

    return all_masks
