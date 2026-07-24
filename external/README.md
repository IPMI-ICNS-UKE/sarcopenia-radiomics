# external

Third-party repository checkouts required by `02_feature_extraction/02_03_deep_radiomics`
(deep radiomics via the frozen MedDINOv3 backbone). Not vendored in this repo — clone them
here yourself:

```
external/
  MedDINOv3/   # clone of the MedDINOv3 repo: https://github.com/ricklisz/MedDINOv3.git
  dinov3/      # clone of the dinov3 repo: https://github.com/facebookresearch/dinov3.git
```

Both are required even though only the `meddinov3` backbone is used: MedDINOv3
loads its weights (`models/model_meddinov3.pth`) into the `vit_base` class
defined in the `dinov3` package (see
`02_feature_extraction/utils/deep_features.py::load_meddinov3_model`).

These default paths are used automatically; to point at checkouts living
elsewhere instead, set `MEDDINOV3_REPO_ROOT` / `DINOV3_REPO_ROOT` before
running (see `02_feature_extraction/README.md`).
