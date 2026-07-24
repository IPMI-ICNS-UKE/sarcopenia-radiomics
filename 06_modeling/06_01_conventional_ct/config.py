from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


TaskName = Literal[
    "sarcopenia_composite_cls",
    "hand_grip_reg",
    "chair_rise_cls"
]


# Path configuration
@dataclass(frozen=True)
class PathsConfig:
    # 2D or 3D. Settable either by editing this default or via
    # PathsConfig(use_3d=False) — see __post_init__.
    use_3d: bool = True

    gt_table: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    gt_table_cohort_2: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
    )
    baseline_signature_dir: Path = None
    output_root: Path = None

    def __post_init__(self) -> None:
        suffix = "_3d" if self.use_3d else ""
        if self.baseline_signature_dir is None:
            object.__setattr__(
                self, "baseline_signature_dir", REPO_ROOT / "output" / f"05_01_conventional_ct{suffix}"
            )
        if self.output_root is None:
            object.__setattr__(self, "output_root", REPO_ROOT / "output" / f"06_01_conventional_ct{suffix}")

    # Derived directories
    @property
    def metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.output_root / "predictions"

    @property
    def models_dir(self) -> Path:
        return self.output_root / "models"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def plots_dir(self) -> Path:
        return self.output_root / "plots"

    @property
    def interactive_plots_dir(self) -> Path:
        return self.output_root / "plots_interactive"

    def make_all(self) -> None:
        """Create all output directories."""
        for d in [
            self.metrics_dir,
            self.predictions_dir,
            self.models_dir / "cls",
            self.models_dir / "reg",
            self.tables_dir,
            self.plots_dir / "roc",
            self.plots_dir / "calibration",
            self.plots_dir / "decision_curve",
            self.plots_dir / "probability_distribution",
            self.plots_dir / "prediction_scatter",
            self.plots_dir / "coefficients",
            self.interactive_plots_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Cross-validation settings
@dataclass(frozen=True)
class CVConfig:
    n_splits_outer: int = 5
    n_repeats_outer: int = 5
    n_splits_inner: int = 5
    random_state: int = 42
    classification_threshold: float = 0.5
    threshold_selection_method: str = "youden_train"


# Elastic-net hyper-parameter grid
@dataclass(frozen=True)
class ElasticNetConfig:
    l1_ratios: Tuple[float, ...] = (0.0, 0.5, 1.0)
    logistic_cs: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    linear_alphas: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    max_iter: int = 20000
    convergence_tol: float = 1e-4



# Dataset filters
@dataclass(frozen=True)
class FilterConfig:
    required_filters: Tuple[Tuple[str, int], ...] = (("use", 1),)


# Debug / fast-run override
@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    n_repeats_outer_override: int = 3
    n_boot_override: int = 200


# Output formatting
@dataclass(frozen=True)
class OutputConfig:
    metric_decimals: int = 3
    save_interactive_plots: bool = True


# Calibration
@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int = 4
    strategy: Literal["quantile", "uniform"] = "quantile"


# Decision curve
@dataclass(frozen=True)
class DecisionCurveConfig:
    threshold_start: float = 0.10
    threshold_stop: float = 0.80
    threshold_step: float = 0.01
    plot_xlim: Tuple[float, float] = (0.10, 0.80)


# Task definitions
@dataclass(frozen=True)
class TaskConfig:
    name: TaskName
    task_kind: Literal["classification", "regression"]
    target_column_gt: str
    signature_filename: str

    @property
    def cohort_2_signature_filename(self) -> str:
        p = Path(self.signature_filename)
        return f"{p.stem}_cohort_2{p.suffix}"


TASKS: Dict[str, TaskConfig] = {
    "sarcopenia_composite_cls": TaskConfig(
        name="sarcopenia_composite_cls",
        task_kind="classification",
        target_column_gt="sarcopenia_composite",
        signature_filename="signature_baseline_ct.csv",
    ),
    "hand_grip_reg": TaskConfig(
        name="hand_grip_reg",
        task_kind="regression",
        target_column_gt="hand_grip_cont",
        # Signature was built for sarcopenia_composite; reused for regression
        signature_filename="signature_baseline_ct.csv",
    ),
    "chair_rise_cls": TaskConfig(
        name="chair_rise_cls",
        task_kind="classification",
        target_column_gt="chair_rise",
        signature_filename="signature_baseline_ct.csv",
    ),
}


# Model block feature definitions
@dataclass(frozen=True)
class ModelBlockConfig:
    """Feature column names for each modelling block."""
    clinical_features: Tuple[str, ...] = ("sex",)
    gt_smi_feature: str = "gt_smi"
    gt_mra_feature: str = "gt_mra"
    auto_smi_feature: str = "auto_smi"
    auto_mra_feature: str = "auto_mra"

    @property
    def gt_smi_block(self) -> Tuple[str, ...]:
        return (self.gt_smi_feature,)

    @property
    def gt_mra_block(self) -> Tuple[str, ...]:
        return (self.gt_mra_feature,)

    @property
    def gt_smi_mra_block(self) -> Tuple[str, ...]:
        return (self.gt_smi_feature, self.gt_mra_feature)

    @property
    def auto_smi_block(self) -> Tuple[str, ...]:
        return (self.auto_smi_feature,)

    @property
    def auto_mra_block(self) -> Tuple[str, ...]:
        return (self.auto_mra_feature,)

    @property
    def auto_smi_mra_block(self) -> Tuple[str, ...]:
        return (self.auto_smi_feature, self.auto_mra_feature)


# Top-level run configuration
@dataclass(frozen=True)
class RunConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    enet: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decision_curve: DecisionCurveConfig = field(default_factory=DecisionCurveConfig)
    blocks: ModelBlockConfig = field(default_factory=ModelBlockConfig)

RCFG = RunConfig()
