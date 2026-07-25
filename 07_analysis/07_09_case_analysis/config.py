from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_ROOT = _REPO_ROOT / "output"


# Map names used by the selected model
SCORE_MAP_NAMES: Tuple[str, ...] = ("musclefat", "ct")

SCORE_MAP_LABELS: Dict[str, str] = {
    "musclefat": "Muscle fat score",
    "ct": "CT score",
}

# Selected model identifier
MODEL_NAME: str = "mf_ct_scores_clinical_3d"
TASK_NAME: str = "sarcopenia_composite_cls"

# Case definitions
CaseEntry = Tuple[str, str]  # (patient_id, cohort_label)

CASES: Dict[str, Dict[str, CaseEntry]] = {
    "train": {
        "TP": ("Pat61", "cohort1"),
        "TN": ("Pat7", "cohort1"),
        "FP": ("Pat33", "cohort1"),
        "FN": ("Pat57", "cohort1"),
    },
    "test_1": {
        "TP": ("Pat92", "cohort1"),
        "TN": ("Pat93", "cohort1"),
        "FP": ("Pat90", "cohort1"),
        "FN": ("Pat77", "cohort1"),
    },
    "test_2": {
        "TP": ("Pat10", "cohort2"),
        "TN": ("Pat5", "cohort2"),
        "FP": ("Pat29", "cohort2"),  # Mockup
        "FN": ("Pat24", "cohort2"),
    },
}

CASE_ORDER: Tuple[str, ...] = ("TP", "TN", "FP", "FN")


# Paths
@dataclass(frozen=True)
class PathsConfig:
    gt_table_cohort1: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    gt_table_cohort2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
    )
    images_root_cohort1: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/imagesAll"
    )
    images_root_cohort2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll"
    )
    labels_root_cohort1: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/labelsAll"
    )
    labels_root_cohort2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/labelsAll"
    )
    feature_root: Path = _OUTPUT_ROOT / "05_02_spectral_radiomics_signature_3d"
    model_path: Path = (
        _OUTPUT_ROOT
        / "06_02_02_spectral_radiomics_signature_3d"
        / "models"
        / "cls"
        / f"{TASK_NAME}__{MODEL_NAME}.joblib"
    )
    output_root: Path = _OUTPUT_ROOT / "07_09_case_analysis"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    def make_all(self) -> None:
        for d in [self.figures_dir, self.tables_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Imaging config
@dataclass(frozen=True)
class ImagingConfig:
    hu_min: float = -29.0
    hu_max: float = 150.0
    spectral_clip_min: float = 0.0
    skip_hu_refinement: bool = False

    abdominal_wall_mask: str = "abdominal_wall.nii.gz"
    paravertebral_mask: str = "paravertebral.nii.gz"
    l3_mask: str = "vertebra_l3.nii.gz"

    # Display window for the MuscleFat heatmap
    musclefat_vmin: float = 0.0
    musclefat_vmax: float = 100.0

    # CT windowing (soft-tissue window)
    ct_wl: float = 60.0
    ct_ww: float = 350.0


# Figure config
@dataclass(frozen=True)
class FigureConfig:
    dpi: int = 300
    panel_size_inches: Tuple[float, float] = (4, 4)
    row_label_fontsize: int = 11
    col_label_fontsize: int = 9
    colorbar_label_fontsize: int = 8
    overlay_alpha: float = 0.45
    mask_color: str = "#00FF7F"
    cmap_musclefat: str = "hot"


# Excel table config
@dataclass(frozen=True)
class TableConfig:
    color_group1: str = "D9E1F2"  # patient ID, case type
    color_group2: str = "E2EFDA"  # demographics
    color_group3: str = "FFF2CC"  # functional tests
    color_group4: str = "FCE4D6"  # model scores / probability
    decimals: int = 2


# Top-level run config
@dataclass(frozen=True)
class RunConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    table: TableConfig = field(default_factory=TableConfig)


RCFG = RunConfig()
