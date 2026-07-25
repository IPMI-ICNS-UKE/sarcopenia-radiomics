from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from common_config import DeepRadiomicsRunConfig, MapPreprocSpec
from imaging_io import add_repo_to_syspath, crop_image_to_reference_region, resample_like
from mask import apply_hu_threshold_to_mask, extract_axial_slice, to_binary_mask

# MedDINOv3 backbone constants
MODEL_INPUT_SIZE = 512
EMBEDDING_DIM = 768


def load_meddinov3_model(run_cfg: DeepRadiomicsRunConfig, device: torch.device) -> torch.nn.Module:
    add_repo_to_syspath(run_cfg.meddinov3_repo_root)
    add_repo_to_syspath(run_cfg.dinov3_repo_root)

    from dinov3.models.vision_transformer import vit_base

    model = vit_base(
        drop_path_rate=0.2,
        layerscale_init=1.0e-05,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )

    checkpoint = torch.load(str(run_cfg.meddinov3_weights), map_location="cpu", weights_only=False)
    state_dict = checkpoint["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if "ibot" not in k and "dino_head" not in k
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_backbone_model(run_cfg: DeepRadiomicsRunConfig, device: torch.device) -> torch.nn.Module:
    if run_cfg.backbone_name.lower() != "meddinov3":
        raise ValueError(f"Unsupported backbone_name: {run_cfg.backbone_name}")
    return load_meddinov3_model(run_cfg, device)


def sitk_to_np2d(img2d: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img2d).astype(np.float32)


def pad_to_square_np(img: np.ndarray, fill_value: float) -> np.ndarray:
    h, w = img.shape
    side = int(max(h, w))
    out = np.full((side, side), fill_value, dtype=np.float32)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img.astype(np.float32)
    return out


def pad_to_square_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = int(max(h, w))
    out = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = mask.astype(np.uint8)
    return out


def apply_pixel_mask(img: np.ndarray, mask: np.ndarray, fill_value: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    out[mask == 0] = float(fill_value)
    return out


def choose_preproc_spec(map_name: str, run_cfg: DeepRadiomicsRunConfig) -> MapPreprocSpec:
    specs = run_cfg.meddinov3_map_specs
    if map_name not in specs:
        raise ValueError(f"Unknown map_name={map_name!r}. Configured maps: {sorted(specs.keys())}")
    return specs[map_name]


def clip_then_zscore_meddinov3(
    img: np.ndarray, map_name: str, run_cfg: DeepRadiomicsRunConfig
) -> np.ndarray:
    spec = choose_preproc_spec(map_name, run_cfg)
    x = np.clip(img.astype(np.float32), float(spec.clip_min), float(spec.clip_max))
    x = (x - float(spec.mean)) / (float(spec.std) + float(run_cfg.eps))
    return x.astype(np.float32)


def to_3ch_tensor_resized(x_norm: np.ndarray, out_size: int) -> torch.Tensor:
    t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).repeat(3, 1, 1)
    return t.contiguous().float()


@torch.no_grad()
def extract_cls_embedding(
    model: torch.nn.Module,
    x3chw: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    x_bchw = x3chw.unsqueeze(0).to(device)

    out = model(x_bchw, is_training=True)
    if not isinstance(out, dict) or "x_norm_clstoken" not in out:
        raise RuntimeError("Expected dict output with key 'x_norm_clstoken' for MedDINOv3.")
    cls_token = out["x_norm_clstoken"].squeeze(0)

    if cls_token.ndim != 1:
        raise RuntimeError(f"Expected [C] cls token after squeeze, got {tuple(cls_token.shape)}")
    return cls_token.detach().float().cpu().numpy()


def build_masked_model_input(
    image_2d: sitk.Image,
    mask_2d_img_grid: sitk.Image,
    map_name: str,
    run_cfg: DeepRadiomicsRunConfig,
) -> Tuple[torch.Tensor, int]:
    spec = choose_preproc_spec(map_name, run_cfg)

    img_np = sitk_to_np2d(image_2d)
    mask_np = sitk.GetArrayFromImage(sitk.Cast(mask_2d_img_grid, sitk.sitkUInt8)).astype(np.uint8)
    n_pix = int(mask_np.sum())
    if n_pix == 0:
        raise ValueError(f"Empty HU-thresholded mask for map {map_name}")

    img_sq = pad_to_square_np(img_np, fill_value=float(spec.fill_value))
    mask_sq = pad_to_square_mask(mask_np)
    img_masked = apply_pixel_mask(img_sq, mask_sq, fill_value=float(spec.fill_value))

    img_norm = clip_then_zscore_meddinov3(img_masked, map_name=map_name, run_cfg=run_cfg)
    return to_3ch_tensor_resized(img_norm, out_size=MODEL_INPUT_SIZE), n_pix


def compute_deep_features_on_slice(
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
    model: torch.nn.Module,
    device: torch.device,
    run_cfg: DeepRadiomicsRunConfig,
) -> Dict[str, object]:
    mask_thr_ct_2d = apply_hu_threshold_to_mask(
        mask_2d_ct_grid, ct_2d, hu_min=hu_min, hu_max=hu_max
    )
    mask_thr_img_2d = resample_like(mask_thr_ct_2d, image_2d, is_label=True)
    mask_thr_img_2d = sitk.Cast(to_binary_mask(mask_thr_img_2d), sitk.sitkUInt8)

    row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "slice_name": slice_name,
        "slice_index_ct": int(slice_index_ct),
        "map_name": map_name,
        "n_pix_mask_2d_thr_img": int(sitk.GetArrayViewFromImage(mask_thr_img_2d).sum()),
        "error": "",
    }

    try:
        x3, n_pix = build_masked_model_input(
            image_2d=image_2d,
            mask_2d_img_grid=mask_thr_img_2d,
            map_name=map_name,
            run_cfg=run_cfg,
        )

        emb = extract_cls_embedding(model=model, x3chw=x3, device=device)
        if emb.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension: expected {EMBEDDING_DIM}, got {emb.shape[0]}"
            )

        row["n_pixels"] = int(n_pix)
        for i, value in enumerate(emb.tolist()):
            row[f"d{i:03d}_{map_name}"] = float(value)
        return row

    except Exception as e:
        row["status"] = "failed"
        row["error"] = repr(e)
        return row


def nonempty_axial_indices(mask_3d: sitk.Image) -> List[int]:
    arr = sitk.GetArrayViewFromImage(sitk.Cast(mask_3d, sitk.sitkUInt8))
    per_slice = arr.reshape(arr.shape[0], -1).sum(axis=1)
    return np.where(per_slice > 0)[0].tolist()


def compute_deep_features_3d(
    patient_id: str,
    cohort_name: str,
    ct_3d: sitk.Image,
    image_3d: sitk.Image,
    map_name: str,
    mask_3d_ct_grid: sitk.Image,
    mask_name: str,
    hu_min: float,
    hu_max: float,
    model: torch.nn.Module,
    device: torch.device,
    run_cfg: DeepRadiomicsRunConfig,
) -> Dict[str, object]:
    """Volumetric deep descriptor with a 2D backbone.

    Conventional CT images and spectral CT maps are restricted to the muscle region (the mask
    is expected to be the bbox-cropped muscle volume from
    ``automated_3d_mask_variants``). A CLS embedding is extracted per non-empty
    axial slice of the cropped region (same masked-input pipeline as 2D), and the
    slice embeddings are mean-pooled into a single volume-level vector.
    """
    # CT restricted to the mask sub-grid -> guaranteed co-grid, cheap HU threshold.
    ct_on_mask = resample_like(ct_3d, mask_3d_ct_grid, is_label=False)
    mask_thr_ct_3d = apply_hu_threshold_to_mask(
        mask_3d_ct_grid, ct_on_mask, hu_min=hu_min, hu_max=hu_max
    )

    # Map restricted to the muscle region at its native resolution.
    img_region = crop_image_to_reference_region(image_3d, mask_3d_ct_grid, margin_xyz=(2, 2, 2))
    img_region = sitk.Cast(img_region, sitk.sitkFloat32)

    mask_thr_img_3d = resample_like(mask_thr_ct_3d, img_region, is_label=True)
    mask_thr_img_3d = sitk.Cast(to_binary_mask(mask_thr_img_3d), sitk.sitkUInt8)

    nonempty_idx = nonempty_axial_indices(mask_thr_img_3d)

    row: Dict[str, object] = {
        "patient_id": patient_id,
        "cohort": cohort_name,
        "status": "ok",
        "mask": mask_name,
        "map_name": map_name,
        "n_slices": int(len(nonempty_idx)),
        "n_pix_mask_3d_thr_img": int(sitk.GetArrayViewFromImage(mask_thr_img_3d).sum()),
        "error": "",
    }

    try:
        if len(nonempty_idx) == 0:
            raise ValueError(f"Empty 3D HU-thresholded mask for map {map_name}")

        embeddings: List[np.ndarray] = []
        total_pix = 0
        for k in nonempty_idx:
            image_2d = extract_axial_slice(img_region, int(k))
            mask_2d = sitk.Cast(extract_axial_slice(mask_thr_img_3d, int(k)), sitk.sitkUInt8)

            if sitk.GetArrayViewFromImage(mask_2d).sum() == 0:
                print(f"Skipping empty mask slice {k} for patient {patient_id}, map {map_name}")
                continue

            x3, n_pix = build_masked_model_input(
                image_2d=image_2d,
                mask_2d_img_grid=mask_2d,
                map_name=map_name,
                run_cfg=run_cfg,
            )
            emb = extract_cls_embedding(model=model, x3chw=x3, device=device)
            embeddings.append(emb)
            total_pix += int(n_pix)

        emb_mean = np.mean(np.stack(embeddings, axis=0), axis=0)
        if emb_mean.shape[0] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension: expected {EMBEDDING_DIM}, got {emb_mean.shape[0]}"
            )

        row["n_pixels"] = int(total_pix)
        for i, value in enumerate(emb_mean.tolist()):
            row[f"d{i:03d}_{map_name}"] = float(value)
        return row

    except Exception as e:
        row["status"] = "failed"
        row["error"] = repr(e)
        return row
