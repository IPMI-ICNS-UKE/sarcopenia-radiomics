import os
from dataclasses import dataclass
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
# Ground-truth tables (override via CLI if needed).
COHORT1_TABLE = "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
COHORT2_TABLE = (
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "07_01_cohort_description")

OUTPUT_WORKBOOK_NAME = "cohort_description_tables.xlsx"


# Column names (as they appear in ground_truth.xlsx)
@dataclass(frozen=True)
class Columns:
    patient_id: str = "Pat ID"

    # shared (both cohorts)
    age: str = "age"
    sex: str = "sex"
    bmi: str = "bmi"
    hand_grip_bin: str = "hand_grip"
    hand_grip_cont: str = "hand_grip_cont"
    chair_rise_bin: str = "chair_rise"
    chair_rise_cont: str = "chair_rise_cont"
    sarcopenia_composite: str = "sarcopenia_composite"
    test_temporal: str = "test_temporal"
    use: str = "use"

    # cohort 1 only
    ecog: str = "ecog"
    tumor_type: str = "tumor_type"
    primary_metastasized: str = "primary_metastasized"
    secondary_metastasized: str = "secondary_metastasized"
    num_chem_cycles: str = "num_chem_cycles"

    # cohort 2 only
    meld_score: str = "meld_score"
    child_pugh_grade: str = "child_pugh_grade"
    hcc: str = "hcc"
    listed_for_trans: str = "listed_for_trans"


COLUMNS = Columns()

# Categorical encodings
# sex: 1 -> male, 0 -> female
SEX_MALE_VALUE = 1
SEX_FEMALE_VALUE = 0

# Binary "positive"/"event" value for sarcopenia and metastasis flags.
POSITIVE_VALUE = 1

# ECOG categories shown in Table 2 (in display order).
ECOG_CATEGORIES: List[int] = [0, 1, 2]

# Tumor-type code -> display label.  Any code NOT present here (and missing
# values, unless TUMOR_MISSING_AS_CUP is False) is collapsed into "CUP".
TUMOR_TYPE_MAP: Dict[int, str] = {
    1: "Esophageal Carcinoma",
    12: "Esophageal Carcinoma",
    2: "Gastric Cancer",
    3: "Kolorectal",
    8: "Kolorectal",
    10: "Kolorectal",
    4: "Pancreatic",
    6: "Pancreatic",
    9: "Pancreatic",
    5: "Cholangio",
    7: "Rectal",
    11: "Analkarzinom",
}
TUMOR_CUP_LABEL = "CUP"
# Display order of tumor-type rows in Table 2 (matches the journal template).
TUMOR_TYPE_ORDER: List[str] = [
    "Esophageal Carcinoma",
    "Gastric Cancer",
    "Kolorectal",
    "Pancreatic",
    "Cholangio",
    "Rectal",
    "CUP",
    "Analkarzinom",
]
# If True, a missing tumor_type is also treated as CUP (per "Not listed - CUP").
TUMOR_MISSING_AS_CUP = True

# Child-Pugh grades shown in Table 3 (display order).
CHILD_PUGH_CATEGORIES: List[str] = ["A", "B", "C"]

# Group / split definitions
# Cohort 1 is split by `test_temporal`: 0 -> training, 1 -> testing set 1.
TRAINING_TEST_TEMPORAL_VALUE = 0
TEST1_TEST_TEMPORAL_VALUE = 1
USE_FLAG_VALUE = 1  # rows kept only when use == this value

GROUP_TRAINING = "training"
GROUP_TEST1 = "test1"
GROUP_TEST2 = "test2"

GROUP_LABELS = {
    GROUP_TRAINING: "Training data set",
    GROUP_TEST1: "Testing data set 1",
    GROUP_TEST2: "Testing data set 2",
}

COHORT1_DESCRIPTION = "Cohort 1: gastrointestinal tumors with palliative chemotherapy"
COHORT2_DESCRIPTION = "Cohort 2: patients with end-stage liver disease awaiting liver transplant"

# Formatting options
DECIMALS = 1  # rounding for every value except p-values
PVALUE_DECIMALS = 3  # rounding for p-values
PVALUE_SMALL_CUTOFF = 0.001  # show "<0.001" below this
PERCENT_SYMBOL = "%"  # set to "" to drop the percent sign

# Denominator for "n (%)" cells:
#   "valid"  -> percentage over non-missing values within the group (default)
#   "group"  -> percentage over the full group size
PERCENT_DENOMINATOR = "valid"

# Statistics options
# Continuous comparison across the 3 groups in Table 1:
#   "auto"    -> Shapiro-Wilk (per group) + Levene; ANOVA if all normal &
#                equal variance, otherwise Kruskal-Wallis  (recommended)
#   "anova"   -> always one-way ANOVA
#   "kruskal" -> always Kruskal-Wallis
CONTINUOUS_TEST = "auto"

# Categorical comparison across the 3 groups in Table 1:
#   "auto"      -> Pearson chi-square; if any expected cell < 5 use Fisher's
#                  exact (2x2) or a Monte-Carlo chi-square (r x c) (recommended)
#   "chi2"      -> always Pearson chi-square
#   "fisher_mc" -> always exact / Monte-Carlo
CATEGORICAL_TEST = "auto"

NORMALITY_ALPHA = 0.05  # Shapiro-Wilk / Levene decision threshold
MIN_EXPECTED_COUNT = 5  # chi-square validity threshold
MONTE_CARLO_RESAMPLES = 10000  # permutations for r x c Monte-Carlo test
RANDOM_SEED = 2025  # reproducible Monte-Carlo p-values

# Whether to also compute a p-value for the composite-sarcopenia row of
# Table 1.  Enabled per study request (compares the composite definition
# across the three groups with the same categorical test as the other
# binary rows); set to False to leave the cell empty as in the raw template.
COMPUTE_COMPOSITE_PVALUE = True

# Also write a "Statistical methods" sheet documenting the test used and the
# numeric p-value for every compared variable (useful for the Methods section).
WRITE_METHODS_SHEET = True


@dataclass(frozen=True)
class GroupConst:
    """Study-specific split constants passed into the generic utils layer."""

    use_flag_value: int = USE_FLAG_VALUE
    training_value: int = TRAINING_TEST_TEMPORAL_VALUE
    test1_value: int = TEST1_TEST_TEMPORAL_VALUE
    group_training: str = GROUP_TRAINING
    group_test1: str = GROUP_TEST1
    group_test2: str = GROUP_TEST2


GROUP_CONST = GroupConst()


@dataclass
class RunConfig:
    """Resolved configuration for a single run (CLI can override fields)."""

    cohort1_table: str = COHORT1_TABLE
    cohort2_table: str = COHORT2_TABLE
    output_dir: str = OUTPUT_DIR
    workbook_name: str = OUTPUT_WORKBOOK_NAME
    continuous_test: str = CONTINUOUS_TEST
    categorical_test: str = CATEGORICAL_TEST
    decimals: int = DECIMALS
    pvalue_decimals: int = PVALUE_DECIMALS
    percent_denominator: str = PERCENT_DENOMINATOR
    compute_composite_pvalue: bool = COMPUTE_COMPOSITE_PVALUE
    write_methods_sheet: bool = WRITE_METHODS_SHEET
    random_seed: int = RANDOM_SEED

    @property
    def workbook_path(self) -> str:
        return os.path.join(self.output_dir, self.workbook_name)
