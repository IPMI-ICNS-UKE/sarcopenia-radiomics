import sys
from dataclasses import dataclass
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
UTILS_DIR = THIS_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from common_config import COHORT1, COHORT2, CommonRunConfig

OUTPUT_DIR = REPO_ROOT / "output" / "02_01_conventional_ct"


@dataclass(frozen=True)
class RunConfig(CommonRunConfig):
    out_csv: Path = OUTPUT_DIR / "raw_baseline_ct.csv"
    out_csv_cohort2: Path = OUTPUT_DIR / "raw_baseline_ct_cohort_2.csv"
    out_csv_3d: Path = OUTPUT_DIR / "raw_baseline_ct_3d.csv"
    out_csv_cohort2_3d: Path = OUTPUT_DIR / "raw_baseline_ct_cohort_2_3d.csv"


RCFG = RunConfig()
