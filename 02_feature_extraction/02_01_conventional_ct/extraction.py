import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
UTILS_DIR = THIS_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from clinical_features import (
    compute_conventional_features_3d,
    compute_conventional_features_on_slice,
)
from common_config import CohortConfig
from imaging_io import load_ground_truth, natural_patient_sort
from mask import (
    automated_3d_mask_variants,
    automated_l3_mask_variants,
    build_base_patient_context,
    extract_axial_slice,
    manual_and_model_masksets,
)


def process_patient_2d(row: pd.Series, cohort_cfg: CohortConfig, run_cfg, ctx, height_m: float) -> List[Dict[str, object]]:
    patient_id = str(row["Pat ID"])
    manual_annotation_flag = (
        int(row["manual_annotation"])
        if "manual_annotation" in row and not pd.isna(row["manual_annotation"])
        else 0
    )

    masksets = automated_l3_mask_variants(ctx["muscle"], ctx["k_l3"])
    masksets.extend(
        manual_and_model_masksets(
            cohort_name=cohort_cfg.cohort_name,
            ct=ctx["ct"],
            model_muscle_3d=ctx["muscle"],
            labels_dir=ctx["labels_dir"],
            run_cfg=run_cfg,
            manual_annotation_flag=manual_annotation_flag,
        )
    )

    rows: List[Dict[str, object]] = []
    for mask_name, slice_name, slice_index, mask_2d in masksets:
        rows.append(
            compute_conventional_features_on_slice(
                patient_id=patient_id,
                cohort_name=cohort_cfg.cohort_name,
                ct_2d=extract_axial_slice(ctx["ct"], slice_index),
                mask_2d=mask_2d,
                mask_name=mask_name,
                slice_name=slice_name,
                slice_index=slice_index,
                height_m=height_m,
                hu_min=run_cfg.hu_min,
                hu_max=run_cfg.hu_max,
            )
        )
    return rows


def process_patient_3d(row: pd.Series, cohort_cfg: CohortConfig, run_cfg, ctx, height_m: float) -> List[Dict[str, object]]:
    patient_id = str(row["Pat ID"])
    masksets_3d = automated_3d_mask_variants(ctx["muscle"])

    return [
        compute_conventional_features_3d(
            patient_id=patient_id,
            cohort_name=cohort_cfg.cohort_name,
            ct_3d=ctx["ct"],
            mask_3d=mask_3d,
            mask_name=mask_name,
            height_m=height_m,
            hu_min=run_cfg.hu_min,
            hu_max=run_cfg.hu_max,
        )
        for mask_name, mask_3d in masksets_3d
    ]


def process_patient(row: pd.Series, cohort_cfg: CohortConfig, run_cfg) -> List[Dict[str, object]]:
    patient_id = str(row["Pat ID"])
    height_m = float(row["height"])

    try:
        ctx = build_base_patient_context(patient_id, cohort_cfg, run_cfg)
        if run_cfg.mode == "3d":
            return process_patient_3d(row, cohort_cfg, run_cfg, ctx, height_m)
        return process_patient_2d(row, cohort_cfg, run_cfg, ctx, height_m)

    except Exception as e:
        return [{
            "patient_id": patient_id,
            "cohort": cohort_cfg.cohort_name,
            "status": "failed",
            "mask": "",
            "error": repr(e),
        }]


def run_cohort(cohort_cfg: CohortConfig, run_cfg) -> pd.DataFrame:
    df_gt = load_ground_truth(cohort_cfg, run_cfg, required_columns=["height"])
    patient_ids = natural_patient_sort(df_gt["Pat ID"].astype(str).tolist())
    df_gt = df_gt.set_index("Pat ID").loc[patient_ids].reset_index()

    rows: List[Dict[str, object]] = []
    desc = f"Processing {cohort_cfg.cohort_name} [{run_cfg.mode}]"
    for _, row in tqdm(df_gt.iterrows(), total=len(df_gt), desc=desc):
        rows.extend(process_patient(row=row, cohort_cfg=cohort_cfg, run_cfg=run_cfg))

    out_df = pd.DataFrame(rows)
    preferred_cols = [
        "patient_id",
        "cohort",
        "status",
        "mask",
        "smi_2d",
        "mra_2d",
        "smi_3d",
        "mra_3d",
    ]

    cols_to_drop = ["slice_name", "slice_index", "error", "n_slices"]
    out_df = out_df.drop(columns=[c for c in cols_to_drop if c in out_df.columns])

    existing = [c for c in preferred_cols if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in existing]
    return out_df[existing + remaining]
