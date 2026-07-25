import os
from pathlib import Path

# Root paths

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = str(REPO_ROOT / "output")

GROUND_TRUTH_COHORT_1 = (
    "/home/gkolokolnikov/PhD_project/vault_data/" "SarcopeniaData/table/ground_truth.xlsx"
)

GROUND_TRUTH_COHORT_2 = (
    "/home/gkolokolnikov/PhD_project/vault_data/" "SarcopeniaDataLiver/table/ground_truth.xlsx"
)

# Input subfolders (relative to OUTPUT_ROOT)

CONVENTIONAL_CT_SUBFOLDER = "05_01_conventional_ct"
CONVENTIONAL_CT_MODELS_SUBFOLDER = "06_01_conventional_ct"
MEAN_FRACTION_SUBFOLDER = "06_02_01_spectral_radiomics_mean_fraction_3d"
SIGNATURE_SUBFOLDER = "06_02_02_spectral_radiomics_signature_3d"

# Conventional CT filenames

CONVENTIONAL_CT_COHORT_12_FILE = "signature_baseline_ct.csv"  # train + test 1
CONVENTIONAL_CT_COHORT_2_FILE = "signature_baseline_ct_cohort_2.csv"

# Prediction filenames (inside "predictions" sub-subfolder)

PRED_TRAIN_FILE = "train_oof_cls_predictions_sarcopenia_composite_cls.csv"
PRED_TEST1_FILE = "test_cls_predictions_sarcopenia_composite_cls.csv"
PRED_TEST2_FILE = "test_cls_predictions_sarcopenia_composite_cls_cohort_2.csv"

# Model names to select from prediction CSVs
# Each key = display column name; value = model_name string in prediction CSV.
# For mean-fraction folder we expect one dominant model; same for signature folder.

AUTO_SMI_MODEL: str = "auto_smi"  # auto-select
AUTO_MRA_MODEL: str = "auto_mra_score_clinical"  # auto-select
MUSCLEFAT_MEAN_FRACTION_MODEL: str = "musclefat_mean_fraction_score_clinical_3d"  # auto-select
CT_MUSCLEFAT_SIGNATURE_MODEL: str = "mf_ct_scores_clinical_3d"  # auto-select

# van der Werf thresholds (sex-dependent)
# sex column: 1 = male, 0 = female

SMI_THRESHOLD = {"male": 43.1, "female": 32.7}  # cm²/m²
MRA_THRESHOLD = {"male": 30.9, "female": 24.8}  # HU

# Display labels (order defines heatmap row/column order)

CLASSIFIER_LABELS = {
    "functional_test": "Functional test",
    "smi_van_der_werf": "Van der Werf - SMI",
    "mra_van_der_werf": "Van der Werf - MRA",
    "auto_smi": "2D SMI",
    "auto_mra": "2D MRA",
    "musclefat_mean_fraction_3d": "3D Muscle fat fraction",
    "mf_ct_scores_clinical_3d": "3D Handcrafted radiomics (MF + CT)",
}

# Output

OUTPUT_SUBFOLDER = "07_06_cohens_kappa"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, OUTPUT_SUBFOLDER)

FIGURE_DPI = 300
FIGURE_FORMAT = "png"  # saved as .png

OUTPUT_FILES = {
    "train": f"cohens_kappa_train.{FIGURE_FORMAT}",
    "test_1": f"cohens_kappa_test_1.{FIGURE_FORMAT}",
    "test_2": f"cohens_kappa_test_2.{FIGURE_FORMAT}",
}

# Plot aesthetics

HEATMAP_COLORMAP = "RdYlGn"  # diverging: red (low) → green (high)
HEATMAP_VMIN = -1.0
HEATMAP_VMAX = 1.0
FIGURE_SIZE = (8, 6.5)  # inches (width, height)
FONT_SIZE_ANNOT = 10  # annotation font size inside cells
FONT_SIZE_LABELS = 10  # axis tick font size
FONT_SIZE_TITLE = 12  # figure title font size
FONT_SIZE_CBAR = 9  # color-bar label font size
