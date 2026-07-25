from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunConfig:
    # 2D or 3D. Settable either by editing this default or via
    # RunConfig(use_3d=True) — see __post_init__.
    use_3d: bool = True

    cohort1_features_csv: Path = None
    cohort2_features_csv: Path = None
    output_dir: Path = None
    selection_mode: str = None  # simulated | manual | both

    ground_truth_xlsx: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )

    icc_threshold: float = 0.80
    top_n_profile_features: int = 5

    metadata_cols: Tuple[str, ...] = (
        "patient_id",
        "cohort",
        "status",
        "mask",
    )
    keep_id_cols_in_stable_table: Tuple[str, ...] = ("patient_id",)

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

    train_split_name: str = "train"
    temporal_test_split_name: str = "test"
    external_test_split_name: str = "test_cohort_2"

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
        stage_out_dir = REPO_ROOT / "output" / "02_02_spectral_radiomics"

        if self.cohort1_features_csv is None:
            object.__setattr__(
                self, "cohort1_features_csv", stage_out_dir / f"raw_spectral_radiomics{suffix}.csv"
            )
        if self.cohort2_features_csv is None:
            object.__setattr__(
                self,
                "cohort2_features_csv",
                stage_out_dir / f"raw_spectral_radiomics_cohort_2{suffix}.csv",
            )
        if self.output_dir is None:
            object.__setattr__(
                self, "output_dir", REPO_ROOT / "output" / f"03_02_spectral_radiomics{suffix}"
            )
        if self.selection_mode is None:
            object.__setattr__(self, "selection_mode", "simulated" if self.use_3d else "both")


RCFG = RunConfig()
