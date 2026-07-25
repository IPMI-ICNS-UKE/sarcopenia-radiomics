from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math
import traceback
from tqdm import tqdm

import pandas as pd
import SimpleITK as sitk


# Configuration
IMAGES_ROOT = Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll")
LABELS_ROOT = Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/labelsAll")
OUTPUT_DIR = Path(
    "/home/gkolokolnikov/PhD_project/spectral_ct_sarcopenia/SarcoSpectRadiomics/code_radiology/output/01_data_statistics"
)

MAP_TYPES: Sequence[str] = (
    "HU",
    "MuscleFat",
)

LABEL_CANDIDATES: Dict[str, Sequence[str]] = {
    "abdominal_wall": ("abdominal_wall.nii.gz",),
    "paravertebral": ("paravertebral.nii.gz",),
    "vertebra_l2": ("vertebra_l2.nii.gz",),
}

SPACING_TOL = 1e-5


# Data structures
@dataclass
class GeometryInfo:
    size_xyz: Tuple[int, int, int]
    spacing_xyz: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]


@dataclass
class QCRecord:
    patient_id: str
    map_type: str
    image_path: str
    label_name: str
    label_path: str

    image_exists: bool
    label_exists: bool
    read_success: bool
    error_message: str

    image_size_xyz: str
    label_size_xyz: str
    image_spacing_xyz: str
    label_spacing_xyz: str

    image_origin_xyz: str
    label_origin_xyz: str

    shape_match: Optional[bool]
    spacing_match: Optional[bool]
    origin_match: Optional[bool]
    direction_match: Optional[bool]
    geometry_match: Optional[bool]

    expected_label_size_at_image_spacing_xyz: str
    would_shape_match_after_spacing_adjustment: Optional[bool]

    label_physical_extent_mm_xyz: str
    image_physical_extent_mm_xyz: str


# Utility functions
def tuple_to_str(values: Optional[Sequence], precision: int = 6) -> str:
    if values is None:
        return ""
    formatted = []
    for v in values:
        if isinstance(v, float):
            formatted.append(f"{v:.{precision}f}")
        else:
            formatted.append(str(v))
    return "(" + ", ".join(formatted) + ")"


def find_patient_dirs(images_root: Path) -> List[Path]:
    patient_dirs = [p for p in images_root.iterdir() if p.is_dir() and p.name.startswith("Pat")]
    return sorted(patient_dirs, key=lambda p: natural_patient_sort_key(p.name))


def natural_patient_sort_key(patient_name: str) -> Tuple[str, int]:
    # "Pat30" -> ("Pat", 30)
    prefix = "".join([c for c in patient_name if not c.isdigit()])
    digits = "".join([c for c in patient_name if c.isdigit()])
    num = int(digits) if digits else -1
    return prefix, num


def get_image_path(patient_dir: Path, patient_id: str, map_type: str) -> Path:
    return patient_dir / f"{patient_id}-{map_type}.nii.gz"


def find_label_path(label_patient_dir: Path, label_key: str) -> Optional[Path]:
    for filename in LABEL_CANDIDATES[label_key]:
        candidate = label_patient_dir / filename
        if candidate.is_file():
            return candidate
    return None


def read_geometry(path: Path) -> GeometryInfo:
    img = sitk.ReadImage(str(path))
    return GeometryInfo(
        size_xyz=tuple(int(v) for v in img.GetSize()),
        spacing_xyz=tuple(float(v) for v in img.GetSpacing()),
        origin_xyz=tuple(float(v) for v in img.GetOrigin()),
        direction=tuple(float(v) for v in img.GetDirection()),
    )


def compare_float_tuples(a: Sequence[float], b: Sequence[float], tol: float = SPACING_TOL) -> bool:
    if len(a) != len(b):
        return False
    return all(math.isclose(x, y, rel_tol=0.0, abs_tol=tol) for x, y in zip(a, b))


def compare_direction(a: Sequence[float], b: Sequence[float], tol: float = SPACING_TOL) -> bool:
    return compare_float_tuples(a, b, tol=tol)


def physical_extent_mm(
    size_xyz: Sequence[int], spacing_xyz: Sequence[float]
) -> Tuple[float, float, float]:
    return tuple(float(s) * float(sp) for s, sp in zip(size_xyz, spacing_xyz))


def expected_size_after_spacing_change(
    source_size_xyz: Sequence[int],
    source_spacing_xyz: Sequence[float],
    target_spacing_xyz: Sequence[float],
) -> Tuple[int, int, int]:
    extent = physical_extent_mm(source_size_xyz, source_spacing_xyz)
    expected = []
    for e, sp in zip(extent, target_spacing_xyz):
        if sp <= 0:
            expected.append(-1)
        else:
            expected.append(int(round(e / sp)))
    return tuple(expected)


def make_qc_record_for_missing(
    patient_id: str,
    map_type: str,
    image_path: Path,
    label_name: str,
    label_path: Optional[Path],
    image_exists: bool,
    label_exists: bool,
    error_message: str = "",
) -> QCRecord:
    return QCRecord(
        patient_id=patient_id,
        map_type=map_type,
        image_path=str(image_path),
        label_name=label_name,
        label_path=str(label_path) if label_path is not None else "",
        image_exists=image_exists,
        label_exists=label_exists,
        read_success=False,
        error_message=error_message,
        image_size_xyz="",
        label_size_xyz="",
        image_spacing_xyz="",
        label_spacing_xyz="",
        image_origin_xyz="",
        label_origin_xyz="",
        shape_match=None,
        spacing_match=None,
        origin_match=None,
        direction_match=None,
        geometry_match=None,
        expected_label_size_at_image_spacing_xyz="",
        would_shape_match_after_spacing_adjustment=None,
        label_physical_extent_mm_xyz="",
        image_physical_extent_mm_xyz="",
    )


# Main QC logic
def evaluate_pair(
    patient_id: str,
    map_type: str,
    image_path: Path,
    label_name: str,
    label_path: Path,
) -> QCRecord:
    try:
        image_geom = read_geometry(image_path)
        label_geom = read_geometry(label_path)

        shape_match = image_geom.size_xyz == label_geom.size_xyz
        spacing_match = compare_float_tuples(
            image_geom.spacing_xyz, label_geom.spacing_xyz, tol=SPACING_TOL
        )
        origin_match = compare_float_tuples(
            image_geom.origin_xyz, label_geom.origin_xyz, tol=SPACING_TOL
        )
        direction_match = compare_direction(
            image_geom.direction, label_geom.direction, tol=SPACING_TOL
        )
        geometry_match = shape_match and spacing_match and origin_match and direction_match

        expected_label_size = expected_size_after_spacing_change(
            source_size_xyz=label_geom.size_xyz,
            source_spacing_xyz=label_geom.spacing_xyz,
            target_spacing_xyz=image_geom.spacing_xyz,
        )
        would_shape_match_after_spacing_adjustment = expected_label_size == image_geom.size_xyz

        label_extent = physical_extent_mm(label_geom.size_xyz, label_geom.spacing_xyz)
        image_extent = physical_extent_mm(image_geom.size_xyz, image_geom.spacing_xyz)

        return QCRecord(
            patient_id=patient_id,
            map_type=map_type,
            image_path=str(image_path),
            label_name=label_name,
            label_path=str(label_path),
            image_exists=True,
            label_exists=True,
            read_success=True,
            error_message="",
            image_size_xyz=tuple_to_str(image_geom.size_xyz, precision=0),
            label_size_xyz=tuple_to_str(label_geom.size_xyz, precision=0),
            image_spacing_xyz=tuple_to_str(image_geom.spacing_xyz),
            label_spacing_xyz=tuple_to_str(label_geom.spacing_xyz),
            image_origin_xyz=tuple_to_str(image_geom.origin_xyz),
            label_origin_xyz=tuple_to_str(label_geom.origin_xyz),
            shape_match=shape_match,
            spacing_match=spacing_match,
            origin_match=origin_match,
            direction_match=direction_match,
            geometry_match=geometry_match,
            expected_label_size_at_image_spacing_xyz=tuple_to_str(expected_label_size, precision=0),
            would_shape_match_after_spacing_adjustment=would_shape_match_after_spacing_adjustment,
            label_physical_extent_mm_xyz=tuple_to_str(label_extent),
            image_physical_extent_mm_xyz=tuple_to_str(image_extent),
        )

    except Exception as exc:
        return QCRecord(
            patient_id=patient_id,
            map_type=map_type,
            image_path=str(image_path),
            label_name=label_name,
            label_path=str(label_path),
            image_exists=True,
            label_exists=True,
            read_success=False,
            error_message=f"{type(exc).__name__}: {exc}",
            image_size_xyz="",
            label_size_xyz="",
            image_spacing_xyz="",
            label_spacing_xyz="",
            image_origin_xyz="",
            label_origin_xyz="",
            shape_match=None,
            spacing_match=None,
            origin_match=None,
            direction_match=None,
            geometry_match=None,
            expected_label_size_at_image_spacing_xyz="",
            would_shape_match_after_spacing_adjustment=None,
            label_physical_extent_mm_xyz="",
            image_physical_extent_mm_xyz="",
        )


def run_qc(
    images_root: Path,
    labels_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[QCRecord] = []
    patient_dirs = find_patient_dirs(images_root)

    for patient_dir in tqdm(patient_dirs):
        patient_id = patient_dir.name
        label_patient_dir = labels_root / patient_id

        for map_type in MAP_TYPES:
            image_path = get_image_path(patient_dir, patient_id, map_type)
            image_exists = image_path.is_file()

            for label_name in LABEL_CANDIDATES.keys():
                label_path = find_label_path(label_patient_dir, label_name)
                label_exists = label_path is not None and label_path.is_file()

                if not image_exists or not label_exists:
                    records.append(
                        make_qc_record_for_missing(
                            patient_id=patient_id,
                            map_type=map_type,
                            image_path=image_path,
                            label_name=label_name,
                            label_path=label_path,
                            image_exists=image_exists,
                            label_exists=label_exists,
                            error_message=(
                                "Missing image file"
                                if not image_exists and label_exists
                                else (
                                    "Missing label file"
                                    if image_exists and not label_exists
                                    else "Missing image and label files"
                                )
                            ),
                        )
                    )
                    continue

                records.append(
                    evaluate_pair(
                        patient_id=patient_id,
                        map_type=map_type,
                        image_path=image_path,
                        label_name=label_name,
                        label_path=label_path,
                    )
                )

    df = pd.DataFrame([asdict(r) for r in records])

    csv_path = output_dir / "spectral_ct_label_alignment_qc.csv"
    xlsx_path = output_dir / "spectral_ct_label_alignment_qc.xlsx"
    summary_csv_path = output_dir / "spectral_ct_label_alignment_qc_summary.csv"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    summary_df = build_summary(df)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved detailed QC table to: {csv_path}")
    print(f"Saved detailed QC table to: {xlsx_path}")
    print(f"Saved summary table to: {summary_csv_path}")

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary_rows = []

    group_cols = ["map_type", "label_name"]
    for (map_type, label_name), subdf in df.groupby(group_cols, dropna=False):
        n_total = len(subdf)
        n_read_success = int(subdf["read_success"].fillna(False).sum())
        n_image_exists = int(subdf["image_exists"].fillna(False).sum())
        n_label_exists = int(subdf["label_exists"].fillna(False).sum())

        def count_true(col: str) -> int:
            if col not in subdf.columns:
                return 0
            return int(subdf[col].fillna(False).sum())

        summary_rows.append(
            {
                "map_type": map_type,
                "label_name": label_name,
                "n_total_pairs": n_total,
                "n_image_exists": n_image_exists,
                "n_label_exists": n_label_exists,
                "n_read_success": n_read_success,
                "n_shape_match": count_true("shape_match"),
                "n_spacing_match": count_true("spacing_match"),
                "n_origin_match": count_true("origin_match"),
                "n_direction_match": count_true("direction_match"),
                "n_geometry_match": count_true("geometry_match"),
                "n_would_shape_match_after_spacing_adjustment": count_true(
                    "would_shape_match_after_spacing_adjustment"
                ),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["map_type", "label_name"]).reset_index(drop=True)


def main() -> None:
    try:
        run_qc(
            images_root=IMAGES_ROOT,
            labels_root=LABELS_ROOT,
            output_dir=OUTPUT_DIR,
        )
    except Exception:
        print("QC failed with an unexpected error:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
