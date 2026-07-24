from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    # 2D or 3D. Settable either by editing this default or via
    # RunConfig(use_3d=True) — see __post_init__.
    use_3d: bool = True

    features_csv_cohort1: Path = None
    features_csv_cohort2: Path = None
    output_dir: Path = None

    ground_truth_xlsx: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )

    top_n_profile_features: int = 5

    metadata_cols: Tuple[str, ...] = (
        "patient_id",
        "cohort",
        "status",
        "mask",
    )
    keep_id_cols_in_stable_table: Tuple[str, ...] = ("patient_id", )

    simulated_masks: Tuple[str, ...] = (
        "a_original",
        "a_dilate_p1",
        "a_dilate_p2",
        "a_erode_p1",
        "a_erode_p2",
    )
    manual_rater_order: Tuple[str, ...] = ("i", "j", "a")
    manual_level_order: Tuple[str, ...] = ("l2", "l3", "l4")

    patient_id_col_gt: str = "Pat ID"
    test_col_gt: str = "test_temporal"
    split_col: str = "split"

    cohort1_train_label: str = "train"
    cohort1_test_label: str = "test"
    cohort2_test_label: str = "test_cohort_2"

    filters: Tuple[Tuple[str, int], ...] = (
        ("use", 1),
        ("image_present", 1),
        ("labels_present", 1),
    )

    target_map_types: Tuple[str, ...] = (
        "ct",
        "VNC",
        "MuscleFat",
        "Iodine",
        "ElectronDensity",
        "EffectiveZ",
        "40keV",
        "60keV",
        "80keV",
        "100keV",
        "120keV",
    )

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        stage_out_dir = REPO_ROOT / "output" / "02_03_deep_radiomics"

        if self.features_csv_cohort1 is None:
            object.__setattr__(
                self, "features_csv_cohort1", stage_out_dir / f"raw_deep_radiomics{suffix}.csv"
            )
        if self.features_csv_cohort2 is None:
            object.__setattr__(
                self,
                "features_csv_cohort2",
                stage_out_dir / f"raw_deep_radiomics_cohort_2{suffix}.csv",
            )
        if self.output_dir is None:
            object.__setattr__(self, "output_dir", REPO_ROOT / "output" / f"03_03_deep_radiomics{suffix}")


RCFG = RunConfig()
