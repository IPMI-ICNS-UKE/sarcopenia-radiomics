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

# Vertebral levels and reader/automated identifiers
# Matches the (mask_name, level, slice_index, mask_2d) triplets produced by
# manual_and_model_masksets(): "i_*" = Reader 1 (Isabel), "j_*" = Reader 2
# (Jenni), "a_*" = automated segmentation.

LEVELS = ("l2", "l3", "l4")

READER_1 = "i"
READER_2 = "j"
AUTOMATED = "a"

# Pairwise comparisons to report
# (key_a, key_b, display_label)

PAIRS = (
    (READER_1, READER_2, "Reader 1 vs Reader 2"),
    (READER_1, AUTOMATED, "Reader 1 vs Automated"),
    (READER_2, AUTOMATED, "Reader 2 vs Automated"),
)

# Output

REPO_ROOT = THIS_DIR.parent.parent
OUTPUT_ROOT = str(REPO_ROOT / "output")
OUTPUT_SUBFOLDER = "07_02_segmentation_agreement"
OUTPUT_DIR = str(Path(OUTPUT_ROOT) / OUTPUT_SUBFOLDER)

PER_LEVEL_CSV = "segmentation_agreement_per_level.csv"
PER_PATIENT_CSV = "segmentation_agreement_per_patient.csv"
SUMMARY_CSV = "segmentation_agreement_summary.csv"
FIGURE_FILE = "segmentation_agreement_boxplot.png"

FIGURE_DPI = 300
FIGURE_SIZE = (8.0, 5.5)  # inches (width, height)
FONT_SIZE = 11
