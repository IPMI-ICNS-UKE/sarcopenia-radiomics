import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

ExtractionMode = Literal["2d", "3d"]


@dataclass(frozen=True)
class CohortConfig:
    table_path: Path
    images_root: Path
    labels_root: Path
    cohort_name: str


@dataclass(frozen=True)
class CommonRunConfig:
    hu_min: float = -29.0
    hu_max: float = 150.0

    # Extraction mode: "2d" (single L3 / annotated slice) or "3d" (whole muscle volume).
    # Defaults to "2d" to preserve the original behaviour.
    mode: ExtractionMode = "2d"

    abdominal_wall: str = "abdominal_wall.nii.gz"
    paravertebral: str = "paravertebral.nii.gz"
    l3_mask: str = "vertebra_l3.nii.gz"

    abdominal_wall_i: str = "abdominal_wall_i.nii.gz"
    paravertebral_i: str = "paravertebral_i.nii.gz"
    abdominal_wall_j: str = "abdominal_wall_j.nii.gz"
    paravertebral_j: str = "paravertebral_j.nii.gz"

    # Conventional CT NIfTI expected as: <patient_id>-HU.nii.gz
    ct_nifti_suffix: str = "-HU.nii.gz"

    filters: Tuple[Tuple[str, int], ...] = (
        ("use", 1),
        ("image_present", 1),
        ("labels_present", 1),
    )


@dataclass(frozen=True)
class RadiomicsRunConfig(CommonRunConfig):
    crop_margin_px: int = 10

    map_file_names: Dict[str, str] = field(default_factory=lambda: {
        "ct": "HU",
        "vnc": "VNC",
        "musclefat": "MuscleFat",
        "iodine": "Iodine",
        "electrondensity": "ElectronDensity",
        "effectivez": "EffectiveZ",
        "40kev": "40keV",
        "60kev": "60keV",
        "80kev": "80keV",
        "100kev": "100keV",
        "120kev": "120keV",

    })

    pyradiomics_yaml: Dict[str, Path] = field(default_factory=lambda: {
        "ct": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "vnc": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "musclefat": REPO_ROOT / "configs" / "pyradiomics_2d_2bw_allfilt_allfeat.yaml",
        "electrondensity": REPO_ROOT / "configs" / "pyradiomics_2d_1bw_allfilt_allfeat.yaml",
        "iodine": REPO_ROOT / "configs" / "pyradiomics_2d_025bw_allfilt_allfeat.yaml",
        "effectivez": REPO_ROOT / "configs" / "pyradiomics_2d_01bw_allfilt_allfeat.yaml",
        "40kev": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "60kev": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "80kev": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "100kev": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
        "120kev": REPO_ROOT / "configs" / "pyradiomics_2d_5bw_allfilt_allfeat.yaml",
    })


@dataclass(frozen=True)
class MapPreprocSpec:
    clip_min: float
    clip_max: float
    fill_value: float
    mean: float
    std: float


@dataclass(frozen=True)
class DeepRadiomicsRunConfig(CommonRunConfig):
    backbone_name: str = "meddinov3"

    # Trained-model file shipped alongside this repo
    meddinov3_weights: Path = REPO_ROOT / "models" / "model_meddinov3.pth"

    # Local checkouts of the MedDINOv3 required because MedDINOv3
    # reuses the dinov3 vision-transformer architecture (see utils/deep_features.py).
    # Override via the MEDDINOV3_REPO_ROOT environment variables
    meddinov3_repo_root: Path = Path(
        os.environ.get("MEDDINOV3_REPO_ROOT", str(REPO_ROOT / "external" / "MedDINOv3"))
    )
    dinov3_repo_root: Path = Path(
        os.environ.get("DINOV3_REPO_ROOT", str(REPO_ROOT / "external" / "dinov3"))
    )

    eps: float = 1e-8

    map_file_names: Dict[str, str] = field(default_factory=lambda: {
        "ct": "HU",
        "vnc": "VNC",
        "musclefat": "MuscleFat",
        "iodine": "Iodine",
        "electrondensity": "ElectronDensity",
        "effectivez": "EffectiveZ",
        "40kev": "40keV",
        "60kev": "60keV",
        "80kev": "80keV",
        "100kev": "100keV",
        "120kev": "120keV",

    })

    meddinov3_map_specs: Dict[str, MapPreprocSpec] = field(default_factory=lambda: {
        "ct": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "vnc": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "musclefat": MapPreprocSpec(0.0, 100.0, 0.0, 20.0, 35.0),
        "effectivez": MapPreprocSpec(0.0, 30.0, 0.0, 4.0, 6.0),
        "electrondensity": MapPreprocSpec(0.0, 500.0, 0.0, 100.0, 150.0),
        "iodine": MapPreprocSpec(0.0, 40.0, 0.0, 0.50, 1.0),
        "40kev": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "60kev": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "80kev": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "100kev": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
        "120kev": MapPreprocSpec(-1000.0, 1000.0, -1024.0, 65.0, 180.0),
    })


COHORT1 = CohortConfig(
    table_path=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"),
    images_root=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/imagesAll"),
    labels_root=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/labelsAll"),
    cohort_name="cohort1",
)

COHORT2 = CohortConfig(
    table_path=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"),
    images_root=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll"),
    labels_root=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/labelsAll"),
    cohort_name="cohort2",
)
