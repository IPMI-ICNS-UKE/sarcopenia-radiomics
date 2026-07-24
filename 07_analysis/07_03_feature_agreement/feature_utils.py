from typing import Dict, List

import pandas as pd
import SimpleITK as sitk

import config
from clinical_features import area_cm2_from_2d_mask, compute_smi, mean_in_mask
from data_loader import PatientL3Data
from mask import apply_hu_threshold_to_mask


def compute_biomarkers_for_mask(
    mask_2d: sitk.Image,
    ct_2d: sitk.Image,
    musclefat_2d: sitk.Image,
    height_m: float,
) -> Dict[str, float]:
    mask_thr_2d = apply_hu_threshold_to_mask(
        mask_2d, ct_2d, hu_min=config.RUN_CFG.hu_min, hu_max=config.RUN_CFG.hu_max
    )

    area_cm2 = area_cm2_from_2d_mask(mask_thr_2d)
    return {
        "smi_2d": compute_smi(area_cm2=area_cm2, height_m=height_m),
        "mra_2d": mean_in_mask(ct_2d, mask_thr_2d),
        "ff_2d": mean_in_mask(musclefat_2d, mask_thr_2d),
    }


def compute_raw_values(patient_id: str, patient_data: PatientL3Data) -> List[Dict[str, object]]:
    """One row per (patient_id, rater) with smi_2d, mra_2d, ff_2d."""
    rows: List[Dict[str, object]] = []
    for rater in config.RATERS:
        biomarkers = compute_biomarkers_for_mask(
            mask_2d=patient_data.masks_by_rater[rater],
            ct_2d=patient_data.ct_2d,
            musclefat_2d=patient_data.musclefat_2d,
            height_m=patient_data.height_m,
        )
        rows.append({
            "patient_id": patient_id,
            "rater": rater,
            **biomarkers,
        })
    return rows


def build_raw_values_df(all_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    return df[["patient_id", "rater"] + list(config.BIOMARKERS)]
