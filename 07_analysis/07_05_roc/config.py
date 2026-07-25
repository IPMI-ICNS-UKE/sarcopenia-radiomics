import os
from pathlib import Path

# Root output directory
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = str(REPO_ROOT / "output")

# Stage output directory
STAGE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "07_05_roc")

# Target / endpoint
TARGET = "sarcopenia_composite"
TASK_SUFFIX = "cls"  # classification suffix used in file names

# Model source subfolders and their model names
# Each entry: (subfolder_name, [model_names_to_include])
MODEL_SOURCES = [
    (
        "06_01_conventional_ct",
        ["auto_smi", "auto_mra_score_clinical"],
    ),
    (
        "06_01_conventional_ct_3d",
        ["auto_smi_3d", "auto_mra_score_clinical_3d"],
    ),
    (
        "06_02_01_spectral_radiomics_mean_fraction_3d",
        ["musclefat_mean_fraction_score_clinical_3d"],
    ),
    (
        "06_02_02_spectral_radiomics_signature_3d",
        ["mf_ct_scores_clinical_3d"],
    ),
]

# Display names for each model
MODEL_DISPLAY_NAMES = {
    "auto_smi": "2D SMI",
    "auto_mra_score_clinical": "2D MRA",
    "auto_smi_3d": "3D SMI",
    "auto_mra_score_clinical_3d": "3D MRA",
    "musclefat_mean_fraction_score_clinical_3d": "3D Muscle fat - mean fraction",
    "mf_ct_scores_clinical_3d": "3D Handcrafted radiomics (CT and MF)",
}

# Ordered list for consistent legend / plot ordering
MODEL_ORDER = [
    "auto_smi",
    "auto_mra_score_clinical",
    "auto_smi_3d",
    "auto_mra_score_clinical_3d",
    "musclefat_mean_fraction_score_clinical_3d",
    "mf_ct_scores_clinical_3d",
]

# Dataset splits
# split_key → (metrics_filename, predictions_filename, plot_title_label)
SPLITS = {
    "train": (
        f"train_oof_cls_metrics_{TARGET}_{TASK_SUFFIX}.xlsx",
        f"train_oof_cls_predictions_{TARGET}_{TASK_SUFFIX}.csv",
        "Training set (OOF)",
    ),
    "test_1": (
        f"test_cls_metrics_{TARGET}_{TASK_SUFFIX}.xlsx",
        f"test_cls_predictions_{TARGET}_{TASK_SUFFIX}.csv",
        "Test set 1 (temporal hold-out)",
    ),
    "test_2": (
        f"test_cls_metrics_{TARGET}_{TASK_SUFFIX}_cohort_2.xlsx",
        f"test_cls_predictions_{TARGET}_{TASK_SUFFIX}_cohort_2.csv",
        "Test set 2 (external cohort)",
    ),
}

# Plot aesthetics
# Colour palette — one colour per model (in MODEL_ORDER).
# Seaborn's "colorblind" palette: distinguishable under common colour-vision
# deficiencies and when converted to greyscale for print.
PALETTE = [
    "#0173b2",  # 2D SMI    — blue
    "#de8f05",  # 2D MRA    — orange
    "#029e73",  # 3D SMI    — green
    "#d55e00",  # 3D MRA    — vermillion
    "#cc78bc",  # 3D Frac   — pink
    "#333333",  # 3D Sig    — near-black
]

# Dash patterns (on, off, ...) in points; None = solid. Varied per curve so
# curves remain distinguishable even in greyscale/B&W print.
LINE_STYLES = [
    (6, 2),  # 2D SMI  — dashed
    (1, 1),  # 2D MRA  — dotted
    (4, 1, 1, 1),  # 3D SMI  — dash-dot
    (3, 1.5),  # 3D MRA  — short dash
    (6, 1.5, 1, 1.5),  # 3D Frac — long dash-dot
    None,  # 3D Sig  — solid
]

LINE_WIDTH = 1.8
FIGURE_SIZE_IN = (5.0, 6.6)  # inches — single-column Radiology figure + legend strip below
DPI = 300
FONT_FAMILY = "Arial"
FONT_FALLBACKS = ["Arial", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE_AXIS = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 8
FONT_SIZE_TITLE = 11

# Output file base names
OUTPUT_FILENAMES = {
    "train": "ROC_train",
    "test_1": "ROC_test_1",
    "test_2": "ROC_test_2",
}
