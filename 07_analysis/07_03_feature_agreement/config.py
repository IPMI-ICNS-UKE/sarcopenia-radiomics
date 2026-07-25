import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FEATURE_EXTRACTION_UTILS_DIR = THIS_DIR.parent.parent / "02_feature_extraction" / "utils"
if str(FEATURE_EXTRACTION_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTION_UTILS_DIR))

from common_config import COHORT1, CommonRunConfig  # noqa: E402

# Cohort / run config
# Only cohort 1 has manual reader annotations (manual_annotation == 1).
COHORT_CFG = COHORT1
RUN_CFG = CommonRunConfig()

# Restricted to L3, per the study protocol.
LEVEL = "l3"

MUSCLEFAT_FILE_SUFFIX = "-MuscleFat.nii.gz"

# Reader / automated identifiers
# Matches the (mask_name, level, slice_index, mask_2d) triplets produced by
# manual_and_model_masksets(): "i_*" = Reader 1 (Isabel), "j_*" = Reader 2
# (Jenni), "a_*" = automated segmentation.
READER_1 = "i"
READER_2 = "j"
AUTOMATED = "a"
RATERS = (READER_1, READER_2, AUTOMATED)
RATER_LABELS = {
    READER_1: "Reader 1",
    READER_2: "Reader 2",
    AUTOMATED: "Automated",
}

# Biomarkers to evaluate
BIOMARKERS = ("smi_2d", "mra_2d", "ff_2d")
BIOMARKER_LABELS = {
    "smi_2d": "2D SMI",
    "mra_2d": "2D MRA",
    "ff_2d": "2D Muscle fat fraction",
}

# Output
REPO_ROOT = THIS_DIR.parent.parent
OUTPUT_ROOT = str(REPO_ROOT / "output")
OUTPUT_SUBFOLDER = "07_03_feature_agreement"
OUTPUT_DIR = str(Path(OUTPUT_ROOT) / OUTPUT_SUBFOLDER)

RAW_VALUES_CSV = "feature_agreement_raw_values.csv"
ICC_SUMMARY_CSV = "feature_agreement_icc_summary.csv"
