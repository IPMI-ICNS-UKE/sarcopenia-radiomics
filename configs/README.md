# configs

PyRadiomics parameter files used by `02_feature_extraction/02_02_spectral_radiomics`
for handcrafted radiomics extraction (Appendix S4/S5 of the manuscript). Each file
fixes the bin width for one group of image types; feature classes
(first-order, GLCM, GLRLM, GLSZM, NGTDM), `force2D`, and all other PyRadiomics
settings are otherwise identical across files. `force2D` is overridden
programmatically to `False` for 3D extraction (`make_radiomics_extractor` in
`utils/radiomics_features.py`); it does not need to be edited here.

| File | Bin width | Used for |
|---|---|---|
| `pyradiomics_2d_5bw_allfilt_allfeat.yaml` | 5 HU | Conventional CT, VNC, and all virtual monoenergetic maps (40–120 keV) |
| `pyradiomics_2d_2bw_allfilt_allfeat.yaml` | 2 percentage points | Muscle fat fraction map |
| `pyradiomics_2d_1bw_allfilt_allfeat.yaml` | 1 unit | Electron density map |
| `pyradiomics_2d_025bw_allfilt_allfeat.yaml` | 0.25 mg/mL | Iodine density map |
| `pyradiomics_2d_01bw_allfilt_allfeat.yaml` | 0.1 unit | Effective atomic number map |

The mapping from map name to YAML file is defined in
`02_feature_extraction/utils/common_config.py` (`RadiomicsRunConfig.pyradiomics_yaml`).
