from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    # 2D or 3D. Settable either by editing this default or via
    # RunConfig(use_3d=True) — see __post_init__.
    use_3d: bool = False

    stable_root: Path = None
    output_root: Path = None

    ground_truth_path_cohort_1: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    ground_truth_path_cohort_2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
    )

    map_names: Tuple[str, ...] = (
        "ct",
        "vnc",
        "musclefat",
        "iodine",
        "electrondensity",
        "effectivez",
        "40kev",
        "60kev",
        "80kev",
        "100kev",
        "120kev",
    )

    stable_features_pattern_cohort_1: str = "stable_deep_radiomics_{map_name}.csv"
    stable_features_pattern_cohort_2: str = "stable_deep_radiomics_{map_name}_cohort_2.csv"

    filtered_features_pattern: str = "filtered_deep_radiomics_{map_name}.csv"
    filtered_train_pattern: str = "filtered_deep_radiomics_{map_name}_train.csv"
    filtered_test_pattern: str = "filtered_deep_radiomics_{map_name}_test.csv"
    filtered_features_pattern_cohort_2: str = "filtered_deep_radiomics_{map_name}_cohort_2.csv"
    filtered_test_pattern_cohort_2: str = "filtered_deep_radiomics_{map_name}_test_cohort_2.csv"

    patient_id_col_gt: str = "Pat ID"
    test_flag_col_gt: str = "test_temporal"
    patient_id_col_data: str = "patient_id"
    cohort_col_data: str = "cohort"
    cohort_1_label: str = "cohort1"
    cohort_2_label: str = "cohort2"

    spearman_rho_threshold_qc: float = 0.90
    make_plots: bool = True
    heatmap_max_features: int = 150
    id_columns: Tuple[str, ...] = field(default_factory=lambda: ("patient_id",))

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        if self.stable_root is None:
            object.__setattr__(
                self, "stable_root", REPO_ROOT / "output" / f"03_03_deep_radiomics{suffix}"
            )
        if self.output_root is None:
            object.__setattr__(
                self, "output_root", REPO_ROOT / "output" / f"04_03_deep_radiomics{suffix}"
            )


RCFG = RunConfig()
