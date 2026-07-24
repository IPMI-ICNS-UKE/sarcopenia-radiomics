# models

Trained-model file referenced by the feature-extraction pipeline (manuscript
model-availability statement).

| File | Size | Used by |
|---|---|---|
| `model_meddinov3.pth` | ~327 MB | `02_feature_extraction/02_03_deep_radiomics` — frozen MedDINOv3 backbone (`DeepRadiomicsRunConfig.meddinov3_weights` in `utils/common_config.py`). Checkpoint contains a `"teacher"` state dict loaded into a `dinov3` `vit_base` architecture (see `utils/deep_features.py::load_meddinov3_model`). |


