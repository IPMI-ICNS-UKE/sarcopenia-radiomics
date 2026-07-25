# 00_data_utilities

Data-wrangling utilities used to organize, rename, align, and quality-check the raw
imaging data for Cohort 1 (gastrointestinal adenocarcinoma) and Cohort 2 (end-stage
liver disease) before feature extraction. These scripts do not perform segmentation,
radiomics extraction, or modeling.

## Scripts

| Script | Purpose |
|---|---|
| `00_00_copy_inter_reader_annotation.py` | Copies manual inter-reader segmentation exports (abdominal wall, paravertebral, psoas) into the per-patient stability-test folder structure, one file per reader. |
| `00_01_organize_spectral_ct.py` | Sorts exported Cohort 2 spectral CT NIfTI files into `Pat<ID>/<phase>/` folders based on filename patterns. |
| `00_02_check_spectral_ct.py` | Generates an Excel summary of which contrast phases and how many NIfTI files are present per Cohort 2 patient. |
| `00_03_cohort_2_copy.py` | Aligns Cohort 2 segmentation-mask exports to standardized filenames (`abdominal_wall.nii.gz`, `psoas.nii.gz`, etc.) and copies/moves them into `Pat<ID>/` folders, with CSV/JSON reports of missing, duplicate, or unrecognized files. |
| `00_04_cohort_2_unify_naming.py` | Renames Cohort 2 image files from the raw export convention (`S2P<ID>-<phase>-<map>.nii.gz`) to the unified convention (`Pat<ID>-<map>.nii.gz`). |
| `00_05_cohort_2_check_spacing.py` | QC check comparing image/label geometry (size, spacing, origin, direction) for Cohort 2 HU and muscle-fat maps against segmentation masks. |
| `00_06_cohort_2_spectral_ct_parameters.py` | Computes 2D/3D muscle area, density (HU), and fat-fraction summary values per Cohort 2 patient for a sanity check against the main pipeline. |
| `00_07_cohort_2_align_data/` | Rigidly registers Cohort 2 segmentation masks to the HU spectral CT grid when origin/orientation mismatches were detected (`alignment_config.py` for parameters, `run.py` as the entry point). |

## Running

Each script is standalone; run individually from within this folder, e.g.:

```bash
python 00_01_organize_spectral_ct.py
```

## Notes

- Paths at the top of each script (e.g. `IMAGES_ROOT`, `LABELS_ROOT`) point to local
  data storage and are **not portable**. Adjust them to your own data location before
  running, or pass equivalents via the available `argparse` options where present.
- Requires `SimpleITK`, `pandas`, `openpyxl`, `numpy`, `tqdm`.
- Run `00_07_cohort_2_align_data/run.py` from within that folder (or add it to
  `PYTHONPATH`) so that `alignment_config.py` can be imported.
