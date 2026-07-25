from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    # 2D or 3D. Settable either by editing this default or via
    # RunConfig(use_3d=True) — see __post_init__.
    use_3d: bool = False

    input_features_csv: Path = None
    input_features_csv_cohort_2: Path = None
    out_dir: Path = None
    feature_cols: Tuple[str, ...] = None

    ground_truth_xlsx: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )

    stable_csv_name: str = "stable_baseline_ct.csv"
    stable_csv_name_cohort_2: str = "stable_baseline_ct_cohort_2.csv"

    split_col: str = "split"
    split_source_col: str = "test_temporal"
    patient_id_col_features: str = "patient_id"
    patient_id_col_gt: str = "Pat ID"

    filters: Tuple[Tuple[str, int], ...] = (
        ("use", 1),
        ("image_present", 1),
        ("labels_present", 1),
    )

    keep_id_cols: Tuple[str, ...] = ("patient_id",)

    simulated_masks: Tuple[str, ...] = (
        "a_original",
        "a_dilate_p1",
        "a_dilate_p2",
        "a_erode_p1",
        "a_erode_p2",
    )
    manual_levels: Tuple[str, ...] = ("l2", "l3", "l4")
    manual_raters: Tuple[str, ...] = ("i", "j", "a")

    keep_status_value: str = "ok"
    train_label: str = "train"
    test_label: str = "test"
    cohort_2_label: str = "test_cohort_2"

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        stage_out_dir = REPO_ROOT / "output" / "02_01_conventional_ct"

        if self.input_features_csv is None:
            object.__setattr__(
                self, "input_features_csv", stage_out_dir / f"raw_baseline_ct{suffix}.csv"
            )
        if self.input_features_csv_cohort_2 is None:
            object.__setattr__(
                self,
                "input_features_csv_cohort_2",
                stage_out_dir / f"raw_baseline_ct_cohort_2{suffix}.csv",
            )
        if self.out_dir is None:
            object.__setattr__(
                self, "out_dir", REPO_ROOT / "output" / f"03_01_conventional_ct{suffix}"
            )
        if self.feature_cols is None:
            object.__setattr__(
                self, "feature_cols", ("smi_3d", "mra_3d") if self.use_3d else ("smi_2d", "mra_2d")
            )


RCFG = RunConfig()
