from pathlib import Path

# ── Model of interest ────────────────────────────────────────────────────────
MODEL_NAME = "mf_ct_scores_clinical_3d"
DISPLAY_NAME = "3D handcrafted radiomics signature (CT + MF)"

# ── Input: prediction CSVs from 06_02_02_spectral_radiomics_signature ────────
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = REPO_ROOT / "output"
PREDICTIONS_DIR = ROOT_DIR / "06_02_02_spectral_radiomics_signature_3d/predictions"
TRAIN_CSV = PREDICTIONS_DIR / "train_oof_cls_predictions_sarcopenia_composite_cls.csv"
TEST1_CSV = PREDICTIONS_DIR / "test_cls_predictions_sarcopenia_composite_cls.csv"
TEST2_CSV = PREDICTIONS_DIR / "test_cls_predictions_sarcopenia_composite_cls_cohort_2.csv"

# ── Input: ground truth tables (for sex column) ───────────────────────────────
GT_COHORT1 = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
)
GT_COHORT2 = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
)

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT_DIR / "07_08_probability_distribution"

# ── Plot aesthetics ───────────────────────────────────────────────────────────
DPI = 300

# Base colours (solid = male, faded = female)
COLOR_POSITIVE_MALE = "#C0392B"  # vivid red
COLOR_NEGATIVE_MALE = "#2980B9"  # vivid blue
COLOR_POSITIVE_FEMALE = "#E8A89C"  # faded red
COLOR_NEGATIVE_FEMALE = "#A8CCE8"  # faded blue

BAR_GAP_FRACTION = 0.15  # fraction of bar width reserved as gap (0 = no gap)
THRESHOLD_LINE_COLOR = "#2C3E50"
THRESHOLD_LINE_WIDTH = 1.5

FIGURE_WIDTH = 12  # inches
FIGURE_HEIGHT = 5  # inches per split
