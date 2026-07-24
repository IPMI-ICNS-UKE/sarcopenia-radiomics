# 02_feature_extraction

Implementation of feature extraction for:
- `02_01_conventional_ct` — SMI / MRA (Appendix S4, "Conventional CT Biomarkers")
- `02_02_spectral_radiomics` — spectral CT biomarkers + handcrafted radiomics (Appendix S4, "Spectral CT Biomarkers" / "Handcrafted Radiomics")
- `02_03_deep_radiomics` — MedDINOv3-based deep radiomics (Appendix S4, "Deep Radiomics")

Each method supports **two extraction modes**:
- `2d` (default): extraction on the L3 axial slice (and, for cohort 1, the
  manual / inter-rater annotated L2/L3/L4 slices).
- `3d`: extraction over the **whole 3D muscle mask** (abdominal wall +
  paravertebral, combined volumetrically, L2–L4).

Shared logic is stored in `utils/`:
- `common_config.py`: cohort paths, common run parameters, radiomics parameters, deep-radiomics parameters, and the `mode` field (`"2d"` / `"3d"`) on `CommonRunConfig`.
- `imaging_io.py`: table loading, NIfTI loading, sorting, resampling, torch device helper.
- `mask.py`: mask merging, L3 slice selection, 2D perturbations, manual-annotation masksets, plus 3D perturbations and 3D maskset variants.
- `clinical_features.py`: SMI and MRA feature calculation (2D and 3D).
- `radiomics_features.py`: PyRadiomics extraction helpers (2D and 3D).
- `deep_features.py`: frozen MedDINOv3 loading, preprocessing, and embedding extraction (2D and 3D).

## Required companion folders

The default config paths assume the following live at the repository root:

| Path | Contents | Used by |
|---|---|---|
| `configs/pyradiomics_*.yaml` | PyRadiomics parameter files (bin widths, enabled feature classes) — see `configs/README.md` | `02_02_spectral_radiomics` |
| `models/model_meddinov3.pth` | Trained MedDINOv3 weights (see manuscript's model-availability statement) — see `models/README.md` | `02_03_deep_radiomics` |
| `external/MedDINOv3`, `external/dinov3` | Local checkouts of the MedDINOv3 / dinov3 repos (MedDINOv3 reuses the dinov3 vision-transformer architecture) | `02_03_deep_radiomics` |

By default (no environment variables set), these resolve to
`<repo_root>/external/MedDINOv3` and `<repo_root>/external/dinov3` — so simply
cloning the two repos into `external/` as `external/MedDINOv3` and
`external/dinov3` is enough for `02_03_deep_radiomics` to find them with no
further configuration. To use checkouts that live elsewhere instead, set
`MEDDINOV3_REPO_ROOT` / `DINOV3_REPO_ROOT` before running:

```bash
export MEDDINOV3_REPO_ROOT=/path/to/MedDINOv3
export DINOV3_REPO_ROOT=/path/to/dinov3
```

## Running

```bash
cd 02_feature_extraction/02_01_conventional_ct
python run.py

cd ../02_02_spectral_radiomics
python run.py

cd ../02_03_deep_radiomics
python run.py
```

Run each `run.py` from within its own folder — the modules import each other
(and from `utils/`) by inserting `utils/` into `sys.path`, not via package-relative imports.

Each `run.py` produces **both** 2D and 3D outputs for both cohorts in a single
invocation. The 3D blocks are clearly marked in each `run.py` and can be
commented out if only 2D outputs are needed.

## Selecting the mode programmatically

The mode is a field on the run config, so it can be toggled with
`dataclasses.replace` without editing the config class:

```python
from dataclasses import replace
from config import COHORT1, RCFG
from extraction import run_cohort

cfg_3d = replace(RCFG, mode="3d")
df_3d = run_cohort(COHORT1, cfg_3d)
```

## Output files

Each method writes separate CSVs per mode and cohort to
`<repo_root>/output/<method>/`:

| Method | 2D cohort 1 | 2D cohort 2 | 3D cohort 1 | 3D cohort 2 |
| --- | --- | --- | --- | --- |
| `02_01_conventional_ct` | `raw_baseline_ct.csv` | `raw_baseline_ct_cohort_2.csv` | `raw_baseline_ct_3d.csv` | `raw_baseline_ct_cohort_2_3d.csv` |
| `02_02_spectral_radiomics` | `raw_spectral_radiomics.csv` | `raw_spectral_radiomics_cohort_2.csv` | `raw_spectral_radiomics_3d.csv` | `raw_spectral_radiomics_cohort_2_3d.csv` |
| `02_03_deep_radiomics` | `raw_deep_radiomics.csv` | `raw_deep_radiomics_cohort_2.csv` | `raw_deep_radiomics_3d.csv` | `raw_deep_radiomics_cohort_2_3d.csv` |

## 3D masksets

In 3D mode each patient yields five volumetric masksets derived from the 3D
muscle mask: `a_original`, `a_dilate_p1`, `a_dilate_p2`, `a_erode_p1`,
`a_erode_p2` (whole-volume morphological perturbations, used for the feature
stability analysis in Appendix S5). The 2D-only inter-rater / manual
single-slice masksets are not applicable in 3D and are skipped.

## 3D method semantics

- **Conventional CT (`02_01`)** — The muscle area is averaged over all
  non-empty axial slices of the 3D HU-thresholded mask (matching the
  ground-truth `area_cm2_3d` definition); `smi_3d = area / height²`, and
  `mra_3d` is the mean HU inside the 3D HU-thresholded mask. Output columns:
  `smi_3d`, `mra_3d`, `n_slices`. (2D columns remain `smi_2d`, `mra_2d`.)
- **Spectral radiomics (`02_02`)** — The same per-map bin widths are reused, but
  the extractor runs with `force2D = False` for true volumetric texture. The 3D
  mask is HU-thresholded on the CT grid, resampled to each map grid, and cropped
  to its 3D bounding box (margin applied in all three dimensions). The configs
  enable first-order/GLCM/GLRLM/GLSZM/NGTDM (no shape features), so the
  `force2D` switch alone is sufficient.
- **Deep radiomics (`02_03`)** — The 2D backbone is applied slice-wise over
  every non-empty axial slice of the 3D HU-thresholded mask (resampled to the
  map grid), and the per-slice CLS embeddings are **mean-pooled** into a single
  volume-level vector per map. The embedding dimension is preserved
  (`d000_<map>`, `d001_<map>`, ...). Output includes `n_slices`.

## Notes

- Input data paths (`COHORT1`, `COHORT2` in `utils/common_config.py`) point to
  local data storage and must be adjusted to your own paths before running.
- Requires `pandas`, `numpy`, `scipy`, `SimpleITK`, `pyradiomics`, `torch`, `tqdm`.
