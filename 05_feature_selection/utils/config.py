from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaseStage05Config:
    ground_truth_path: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    ground_truth_path_cohort_2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
    )

    id_col_gt: str = "Pat ID"
    patient_id_col: str = "patient_id"
    cohort_col: str = "cohort"
    split_col: str = "split"
    temporal_flag_col: str = "test_temporal"
    target_col: str = "sarcopenia_composite"

    cohort_1_key: str = "cohort_1"
    cohort_2_key: str = "cohort_2"
    cohort_1_value: str = "cohort1"
    cohort_2_value: str = "cohort2"
    cohort_2_suffix: str = "_cohort_2"

    verbose: bool = True

    def get_ground_truth_path(self, cohort_key: str) -> Path:
        if cohort_key == self.cohort_1_key:
            return self.ground_truth_path
        if cohort_key == self.cohort_2_key:
            return self.ground_truth_path_cohort_2
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")

    def get_cohort_value(self, cohort_key: str) -> str:
        if cohort_key == self.cohort_1_key:
            return self.cohort_1_value
        if cohort_key == self.cohort_2_key:
            return self.cohort_2_value
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")

    def get_output_suffix(self, cohort_key: str) -> str:
        if cohort_key == self.cohort_1_key:
            return ""
        if cohort_key == self.cohort_2_key:
            return self.cohort_2_suffix
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")


@dataclass(frozen=True)
class ConventionalCTConfig(BaseStage05Config):
    # 2D or 3D. Settable either by editing this default or via
    # ConventionalCTConfig(use_3d=True) — see __post_init__.
    use_3d: bool = False

    feature_root: Path = None
    output_root: Path = None
    feature_cols: Tuple[str, ...] = None

    cohort_1_features_filename: str = "filtered_baseline_ct.csv"
    cohort_2_features_filename: str = "filtered_baseline_ct_cohort_2.csv"
    signature_filename: str = "signature_baseline_ct.csv"

    def get_feature_path(self, cohort_key: str) -> Path:
        if cohort_key == self.cohort_1_key:
            return self.feature_root / self.cohort_1_features_filename
        if cohort_key == self.cohort_2_key:
            return self.feature_root / self.cohort_2_features_filename
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        if self.feature_root is None:
            object.__setattr__(self, "feature_root", REPO_ROOT / "output" / f"04_01_conventional_ct{suffix}")
        if self.output_root is None:
            object.__setattr__(self, "output_root", REPO_ROOT / "output" / f"05_01_conventional_ct{suffix}")
        if self.feature_cols is None:
            object.__setattr__(
                self, "feature_cols", ("smi_3d", "mra_3d") if self.use_3d else ("smi_2d", "mra_2d")
            )


@dataclass(frozen=True)
class SpectralMeanFractionConfig(BaseStage05Config):
    # 2D or 3D. Settable either by editing this default or via
    # SpectralMeanFractionConfig(use_3d=False) — see __post_init__.
    use_3d: bool = True

    raw_features_path: Path = None
    raw_features_path_cohort_2: Path = None
    output_root: Path = None

    maps: Tuple[str, ...] = (
        "ct",
        "vnc",
        "musclefat",
        "effectivez",
        "electrondensity",
        "iodine",
        "40kev",
        "60kev",
        "80kev",
        "100kev",
        "120kev",
    )
    mask_col: str = "mask"
    required_mask_value: str = "a_original"
    exported_feature_name: str = "mean_fraction"
    signature_template: str = "signature_spectral_radiomics_mean_fraction_{map_name}.csv"

    def get_raw_features_path(self, cohort_key: str) -> Path:
        if cohort_key == self.cohort_1_key:
            return self.raw_features_path
        if cohort_key == self.cohort_2_key:
            return self.raw_features_path_cohort_2
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        stage_dir = REPO_ROOT / "output" / "02_02_spectral_radiomics"
        if self.raw_features_path is None:
            object.__setattr__(self, "raw_features_path", stage_dir / f"raw_spectral_radiomics{suffix}.csv")
        if self.raw_features_path_cohort_2 is None:
            object.__setattr__(
                self, "raw_features_path_cohort_2", stage_dir / f"raw_spectral_radiomics_cohort_2{suffix}.csv"
            )
        if self.output_root is None:
            object.__setattr__(
                self, "output_root", REPO_ROOT / "output" / f"05_02_spectral_radiomics_mean_fraction{suffix}"
            )


@dataclass(frozen=True)
class SpectralSignatureConfig(BaseStage05Config):
    # 2D or 3D. Settable either by editing this default or via
    # SpectralSignatureConfig(use_3d=False) — see __post_init__.
    use_3d: bool = True

    filtered_root: Path = None
    output_root: Path = None

    maps: Tuple[str, ...] = (
        "ct",
        "vnc",
        "musclefat",
        "effectivez",
        "electrondensity",
        "iodine",
        "40kev",
        "60kev",
        "80kev",
        "100kev",
        "120kev",
    )

    feature_file_template: str = "filtered_spectral_radiomics_{map_name}.csv"
    feature_file_template_cohort_2: str = "filtered_spectral_radiomics_{map_name}_cohort_2.csv"
    signature_template: str = "signature_spectral_radiomics_signature_{map_name}.csv"

    n_splits: int = 5
    random_state: int = 42
    max_iter: int = 10000
    tol: float = 1e-4
    scoring: str = "roc_auc"
    c_grid: Tuple[float, ...] = (
        0.001, 0.002, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1.0, 2.0, 5.0,
        10.0, 20.0, 50.0, 100.0,
    )
    selection_frequency_repeats: int = 100
    selection_frequency_threshold: float = 0.0
    stability_resampling_method: str = "bootstrap"
    bootstrap_stratified: bool = True
    min_non_missing_fraction: float = 0.8
    near_constant_threshold: float = 1e-12

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        if self.filtered_root is None:
            object.__setattr__(self, "filtered_root", REPO_ROOT / "output" / f"04_02_spectral_radiomics{suffix}")
        if self.output_root is None:
            object.__setattr__(
                self, "output_root", REPO_ROOT / "output" / f"05_02_spectral_radiomics_signature{suffix}"
            )


@dataclass(frozen=True)
class DeepRadiomicsConfig(BaseStage05Config):
    # 2D or 3D. Settable either by editing this default or via
    # DeepRadiomicsConfig(use_3d=False) — see __post_init__.
    use_3d: bool = True

    raw_features_path: Path = None
    raw_features_path_cohort_2: Path = None
    output_root_raw: Path = None

    maps: Tuple[str, ...] = (
        "ct",
        "vnc",
        "musclefat",
        "effectivez",
        "electrondensity",
        "iodine",
        "40kev",
        "60kev",
        "80kev",
        "100kev",
        "120kev",
    )

    raw_map_aliases: Tuple[Tuple[str, str], ...] = (
        ("ct", "ct"),
        ("vnc", "vnc"),
        ("musclefat", "musclefat"),
        ("effectivez", "effectivez"),
        ("electrondensity", "electrondensity"),
        ("iodine", "iodine"),
        ("40kev", "40kev"),
        ("60kev", "60kev"),
        ("80kev", "80kev"),
        ("100kev", "100kev"),
        ("120kev", "120kev"),
    )

    status_col: str = "status"
    required_status_value: str = "ok"
    mask_col: str = "mask"
    required_mask_value: str = "a_original"

    raw_signature_template: str = "signature_deep_radiomics_{map_name}.csv"

    def get_raw_features_path(self, cohort_key: str) -> Path:
        if cohort_key == self.cohort_1_key:
            return self.raw_features_path
        if cohort_key == self.cohort_2_key:
            return self.raw_features_path_cohort_2
        raise ValueError(f"Unsupported cohort key: {cohort_key!r}")

    def get_raw_source_map_name(self, map_name: str) -> str:
        mapping = dict(self.raw_map_aliases)
        if map_name not in mapping:
            raise ValueError(f"Unsupported deep-radiomics map: {map_name!r}")
        return mapping[map_name]

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        stage_dir = REPO_ROOT / "output" / "02_03_deep_radiomics"
        if self.raw_features_path is None:
            object.__setattr__(self, "raw_features_path", stage_dir / f"raw_deep_radiomics{suffix}.csv")
        if self.raw_features_path_cohort_2 is None:
            object.__setattr__(
                self, "raw_features_path_cohort_2", stage_dir / f"raw_deep_radiomics_cohort_2{suffix}.csv"
            )
        if self.output_root_raw is None:
            object.__setattr__(self, "output_root_raw", REPO_ROOT / "output" / f"05_03_deep_radiomics{suffix}")
