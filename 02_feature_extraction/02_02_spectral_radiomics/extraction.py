import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
UTILS_DIR = THIS_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from common_config import CohortConfig
from imaging_io import load_ground_truth, natural_patient_sort, read_ct_nifti, read_image
from mask import (
    automated_3d_mask_variants,
    automated_l3_mask_variants,
    axial_index_from_phys,
    build_base_patient_context,
    extract_axial_slice,
    manual_and_model_masksets,
)
from radiomics_features import (
    compute_radiomics_3d,
    compute_radiomics_on_slice,
    make_radiomics_extractor,
)


def read_all_images(patient_id: str, cohort_cfg: CohortConfig, run_cfg) -> Dict[str, sitk.Image]:
    patient_img_dir = cohort_cfg.images_root / patient_id
    images: Dict[str, sitk.Image] = {}

    for map_name, file_stub in run_cfg.map_file_names.items():
        if map_name == "ct":
            images[map_name] = read_ct_nifti(patient_id, cohort_cfg, run_cfg, sitk.sitkFloat32)
        else:
            images[map_name] = read_image(patient_img_dir / f"{patient_id}-{file_stub}.nii.gz", sitk.sitkFloat32)

    return images


def ct_index_to_other_image_index(ct: sitk.Image, other: sitk.Image, k_ct: int) -> int:
    phys = ct.TransformIndexToPhysicalPoint((0, 0, int(k_ct)))
    return axial_index_from_phys(other, phys)


def extract_one_maskset_all_maps(
    patient_id: str,
    cohort_name: str,
    ct: sitk.Image,
    images: Dict[str, sitk.Image],
    mask_2d_ct_grid: sitk.Image,
    mask_name: str,
    slice_name: str,
    slice_index_ct: int,
    run_cfg,
    extractors: Dict[str, featureextractor.RadiomicsFeatureExtractor],
) -> Dict[str, object]:
    ct_2d = extract_axial_slice(ct, slice_index_ct)

    merged_row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "slice_name": slice_name,
        "slice_index_ct": int(slice_index_ct),
        "error": "",
    }

    for map_name, img3d in images.items():
        idx_img = slice_index_ct if map_name == "ct" else ct_index_to_other_image_index(ct, img3d, slice_index_ct)
        image_2d = extract_axial_slice(img3d, idx_img)

        try:
            row_map = compute_radiomics_on_slice(
                patient_id=patient_id,
                cohort_name=cohort_name,
                ct_2d=ct_2d,
                image_2d=image_2d,
                map_name=map_name,
                mask_2d_ct_grid=mask_2d_ct_grid,
                mask_name=mask_name,
                slice_name=slice_name,
                slice_index_ct=slice_index_ct,
                hu_min=run_cfg.hu_min,
                hu_max=run_cfg.hu_max,
                crop_margin_px=run_cfg.crop_margin_px,
                extractor=extractors[map_name],
            )
        except Exception as e:
            row_map = {"status": "failed", "error": repr(e)}

        if row_map.get("status") == "failed":
            merged_row["status"] = "failed"

        for key, value in row_map.items():
            if key.startswith("original_") or key.startswith("wavelet"):
                merged_row[key] = value

        if row_map.get("error"):
            merged_row[f"error_{map_name}"] = row_map.get("error")

    return merged_row


def extract_one_maskset_all_maps_3d(
    patient_id: str,
    cohort_name: str,
    ct: sitk.Image,
    images: Dict[str, sitk.Image],
    mask_3d_ct_grid: sitk.Image,
    mask_name: str,
    run_cfg,
    extractors: Dict[str, featureextractor.RadiomicsFeatureExtractor],
) -> Dict[str, object]:
    merged_row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "error": "",
    }

    for map_name, img3d in images.items():
        try:
            row_map = compute_radiomics_3d(
                patient_id=patient_id,
                cohort_name=cohort_name,
                ct_3d=ct,
                image_3d=img3d,
                map_name=map_name,
                mask_3d_ct_grid=mask_3d_ct_grid,
                mask_name=mask_name,
                hu_min=run_cfg.hu_min,
                hu_max=run_cfg.hu_max,
                crop_margin_px=run_cfg.crop_margin_px,
                extractor=extractors[map_name],
            )
        except Exception as e:
            row_map = {"status": "failed", "error": repr(e)}

        if row_map.get("status") == "failed":
            merged_row["status"] = "failed"

        for key, value in row_map.items():
            if key.startswith("original_") or key.startswith("wavelet"):
                merged_row[key] = value

        if row_map.get("error"):
            merged_row[f"error_{map_name}"] = row_map.get("error")

    return merged_row


def process_patient(
    row: pd.Series,
    cohort_cfg: CohortConfig,
    run_cfg,
    extractors: Dict[str, featureextractor.RadiomicsFeatureExtractor],
) -> List[Dict[str, object]]:
    patient_id = str(row["Pat ID"])
    manual_annotation_flag = (
        int(row["manual_annotation"])
        if "manual_annotation" in row and not pd.isna(row["manual_annotation"])
        else 0
    )

    try:
        ctx = build_base_patient_context(patient_id, cohort_cfg, run_cfg)
        images = read_all_images(patient_id, cohort_cfg, run_cfg)

        if run_cfg.mode == "3d":
            masksets_3d = automated_3d_mask_variants(ctx["muscle"])
            return [
                extract_one_maskset_all_maps_3d(
                    patient_id=patient_id,
                    cohort_name=cohort_cfg.cohort_name,
                    ct=ctx["ct"],
                    images=images,
                    mask_3d_ct_grid=mask_3d,
                    mask_name=mask_name,
                    run_cfg=run_cfg,
                    extractors=extractors,
                )
                for mask_name, mask_3d in masksets_3d
            ]

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

        return [
            extract_one_maskset_all_maps(
                patient_id=patient_id,
                cohort_name=cohort_cfg.cohort_name,
                ct=ctx["ct"],
                images=images,
                mask_2d_ct_grid=mask_2d,
                mask_name=mask_name,
                slice_name=slice_name,
                slice_index_ct=slice_index_ct,
                run_cfg=run_cfg,
                extractors=extractors,
            )
            for mask_name, slice_name, slice_index_ct, mask_2d in masksets
        ]

    except Exception as e:
        return [{
            "patient_id": patient_id,
            "cohort": cohort_cfg.cohort_name,
            "status": "failed",
            "mask": "",
            "slice_name": "",
            "error": repr(e),
        }]


def build_extractors(run_cfg) -> Dict[str, featureextractor.RadiomicsFeatureExtractor]:
    return {
        map_name: make_radiomics_extractor(yaml_path, mode=run_cfg.mode)
        for map_name, yaml_path in run_cfg.pyradiomics_yaml.items()
        if map_name in run_cfg.map_file_names
    }


def run_cohort(cohort_cfg: CohortConfig, run_cfg) -> pd.DataFrame:
    df_gt = load_ground_truth(cohort_cfg, run_cfg)
    patient_ids = natural_patient_sort(df_gt["Pat ID"].astype(str).tolist())
    df_gt = df_gt.set_index("Pat ID").loc[patient_ids].reset_index()

    extractors = build_extractors(run_cfg)
    rows: List[Dict[str, object]] = []

    desc = f"Processing {cohort_cfg.cohort_name} [{run_cfg.mode}]"
    for _, row in tqdm(df_gt.iterrows(), total=len(df_gt), desc=desc):
        rows.extend(process_patient(row=row, cohort_cfg=cohort_cfg, run_cfg=run_cfg, extractors=extractors))

    out_df = pd.DataFrame(rows)
    preferred_cols = ["patient_id", "cohort", "status", "mask"]

    cols_to_drop = ["slice_name", "slice_index", "slice_index_ct", "error"]
    out_df = out_df.drop(columns=[c for c in cols_to_drop if c in out_df.columns])

    existing = [c for c in preferred_cols if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in existing]
    return out_df[existing + remaining]
