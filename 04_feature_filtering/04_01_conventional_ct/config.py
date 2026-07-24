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
    feature_columns: Tuple[str, ...] = None

    ground_truth_path_cohort_1: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    ground_truth_path_cohort_2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
    )
    patient_id_col_gt: str = "Pat ID"
    test_flag_col_gt: str = "test_temporal"
    patient_id_col_data: str = "patient_id"

    stable_features_filename: str = "stable_baseline_ct.csv"
    stable_features_filename_cohort_2: str = "stable_baseline_ct_cohort_2.csv"

    id_columns: Tuple[str, ...] = field(default_factory=lambda: ("patient_id",))
    split_col: str = "split"
    dataset_name: str = "baseline_ct"
    cohort_2_postfix: str = "_cohort_2"
    spearman_rho_threshold: float = 0.90
    make_plots: bool = True

    filtered_features_filename: str = "filtered_baseline_ct.csv"
    filtered_train_filename: str = "filtered_baseline_ct_train.csv"
    filtered_test_filename: str = "filtered_baseline_ct_test.csv"

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        if self.stable_root is None:
            object.__setattr__(
                self, "stable_root", REPO_ROOT / "output" / f"03_01_conventional_ct{suffix}"
            )
        if self.output_root is None:
            object.__setattr__(
                self, "output_root", REPO_ROOT / "output" / f"04_01_conventional_ct{suffix}"
            )
        if self.feature_columns is None:
            object.__setattr__(
                self, "feature_columns", ("smi_3d", "mra_3d") if self.use_3d else ("smi_2d", "mra_2d")
            )


RCFG = RunConfig()
