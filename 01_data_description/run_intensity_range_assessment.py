import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# Configuration
@dataclass(frozen=True)
class Config:
    table_path: Path = Path(
        "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
    )
    images_root: Path = Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/imagesAll")
    labels_root: Path = Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/labelsAll")
    output_xlsx: Path = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "01_data_statistics"
        / "intensity_ranges.xlsx"
    )

    # Approved cohort filters
    filters: Tuple[Tuple[str, int], ...] = (
        ("use", 1),
        ("image_present", 1),
        ("labels_present", 1),
    )

    # Map names and filenames relative to patient folder
    map_defs: Tuple[Tuple[str, str], ...] = (
        ("conventional_ct", "ConventionalCT"),
        ("40keV", "{patient_id}-40keV.nii.gz"),
        ("60keV", "{patient_id}-60keV.nii.gz"),
        ("80keV", "{patient_id}-80keV.nii.gz"),
        ("100keV", "{patient_id}-100keV.nii.gz"),
        ("120keV", "{patient_id}-120keV.nii.gz"),
        ("vnc", "{patient_id}-VNC.nii.gz"),
        ("muscle_fat", "{patient_id}-MuscleFat.nii.gz"),
        ("iodine_density", "{patient_id}-Iodine.nii.gz"),
        ("electron_density", "{patient_id}-ElectronDensity.nii.gz"),
        ("effective_z", "{patient_id}-EffectiveZ.nii.gz"),
    )

    abdominal_wall_name: str = "abdominal_wall.nii.gz"
    paravertebral_name: str = "paravertebral.nii.gz"


CFG = Config()


# Table / path helpers
def load_ground_truth(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.table_path)

    for col, val in cfg.filters:
        if col not in df.columns:
            raise KeyError(f"Required filter column missing: {col}")
        df = df[df[col] == val]

    if "Pat ID" not in df.columns:
        raise KeyError("Required column missing: 'Pat ID'")

    return df.reset_index(drop=True)


def natural_sort_patient_ids(patient_ids: List[str]) -> List[str]:
    def key(s: str):
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 0, s)

    return sorted(patient_ids, key=key)


def build_map_path(patient_dir: Path, map_name: str, patient_id: str) -> Path:
    if map_name == "conventional_ct":
        return patient_dir / "ConventionalCT"

    template = dict(CFG.map_defs)[map_name]
    return patient_dir / template.format(patient_id=patient_id)


# SimpleITK IO
def read_image(path: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return sitk.ReadImage(str(path), pixel_type)


def read_dicom_series(dicom_dir: Path, pixel_type=sitk.sitkFloat32) -> sitk.Image:
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetOutputPixelType(pixel_type)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if series_ids:
        series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_files)
        return reader.Execute()

    all_files = sorted(str(p) for p in dicom_dir.iterdir() if p.is_file())
    if not all_files:
        raise FileNotFoundError(f"No files found in DICOM directory: {dicom_dir}")

    reader.SetFileNames(all_files)
    return reader.Execute()


def resample_label_to_image(label_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    """
    Force resampling to image geometry, even if size/spacing already look similar,
    since label and image grids are not guaranteed to be pixel-identical.
    """
    resampled = sitk.Resample(
        label_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk.Cast(sitk.NotEqual(resampled, 0), sitk.sitkUInt8)


# Mask helpers
def load_combined_muscle_mask(patient_id: str, cfg: Config) -> sitk.Image:
    label_dir = cfg.labels_root / patient_id

    abw = read_image(label_dir / cfg.abdominal_wall_name, sitk.sitkInt16)
    para = read_image(label_dir / cfg.paravertebral_name, sitk.sitkInt16)

    abw_bin = sitk.Cast(sitk.NotEqual(abw, 0), sitk.sitkUInt8)
    para_bin = sitk.Cast(sitk.NotEqual(para, 0), sitk.sitkUInt8)

    return sitk.Or(abw_bin, para_bin)


# Statistics
def compute_basic_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
        }

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
    }


def extract_image_and_mask_stats(
    img: sitk.Image,
    combined_mask_native: sitk.Image,
) -> Dict[str, float]:
    """
    Returns stats for:
    - all voxels in image
    - voxels inside abdominal + paravertebral mask
    Mask is always force-resampled to image grid.
    """
    mask = resample_label_to_image(combined_mask_native, img)

    img_arr_full = np.asarray(sitk.GetArrayViewFromImage(img), dtype=np.float32)  # z, y, x
    mask_arr = sitk.GetArrayViewFromImage(mask).astype(bool)

    img_stats = compute_basic_stats(img_arr_full)
    mask_stats = compute_basic_stats(img_arr_full[mask_arr])

    out = {
        "image_min": img_stats["min"],
        "image_max": img_stats["max"],
        "image_mean": img_stats["mean"],
        "image_q1": img_stats["q1"],
        "image_q3": img_stats["q3"],
        "image_iqr": img_stats["iqr"],
        "mask_min": mask_stats["min"],
        "mask_max": mask_stats["max"],
        "mask_mean": mask_stats["mean"],
        "mask_q1": mask_stats["q1"],
        "mask_q3": mask_stats["q3"],
        "mask_iqr": mask_stats["iqr"],
        "mask_voxel_count": int(mask_arr.sum()),
    }
    return out


# Per-patient processing
def load_map_image(patient_id: str, map_name: str, cfg: Config) -> sitk.Image:
    patient_dir = cfg.images_root / patient_id
    map_path = build_map_path(patient_dir, map_name, patient_id)

    if map_name == "conventional_ct":
        return read_dicom_series(map_path, pixel_type=sitk.sitkFloat32)

    return read_image(map_path, pixel_type=sitk.sitkFloat32)


def process_patient_map(
    patient_row: pd.Series,
    map_name: str,
    cfg: Config,
) -> Dict[str, object]:
    patient_id = str(patient_row["Pat ID"])

    try:
        img = load_map_image(patient_id, map_name, cfg)

        if map_name == "muscle_fat":
            img = sitk.Clamp(img, lowerBound=0.0)

        combined_mask_native = load_combined_muscle_mask(patient_id, cfg)
        stats = extract_image_and_mask_stats(img, combined_mask_native)

        return {
            "patient_id": patient_id,
            "map_name": map_name,
            "status": "ok",
            "image_size_x": int(img.GetSize()[0]),
            "image_size_y": int(img.GetSize()[1]),
            "image_size_z": int(img.GetSize()[2]),
            "spacing_x": float(img.GetSpacing()[0]),
            "spacing_y": float(img.GetSpacing()[1]),
            "spacing_z": float(img.GetSpacing()[2]),
            "origin_x": float(img.GetOrigin()[0]),
            "origin_y": float(img.GetOrigin()[1]),
            "origin_z": float(img.GetOrigin()[2]),
            **stats,
            "error": "",
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "map_name": map_name,
            "status": "failed",
            "image_size_x": np.nan,
            "image_size_y": np.nan,
            "image_size_z": np.nan,
            "spacing_x": np.nan,
            "spacing_y": np.nan,
            "spacing_z": np.nan,
            "origin_x": np.nan,
            "origin_y": np.nan,
            "origin_z": np.nan,
            "image_min": np.nan,
            "image_max": np.nan,
            "image_mean": np.nan,
            "image_q1": np.nan,
            "image_q3": np.nan,
            "image_iqr": np.nan,
            "mask_min": np.nan,
            "mask_max": np.nan,
            "mask_mean": np.nan,
            "mask_q1": np.nan,
            "mask_q3": np.nan,
            "mask_iqr": np.nan,
            "mask_voxel_count": np.nan,
            "error": repr(e),
        }


# Aggregation
def summarize_series(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "max": np.nan,
        }

    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "min": float(x.min()),
        "q1": q1,
        "median": float(x.median()),
        "q3": q3,
        "iqr": float(q3 - q1),
        "max": float(x.max()),
    }


def build_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = raw_df[raw_df["status"] == "ok"].copy()

    metrics = [
        "image_min",
        "image_max",
        "image_mean",
        "image_q1",
        "image_q3",
        "image_iqr",
        "mask_min",
        "mask_max",
        "mask_mean",
        "mask_q1",
        "mask_q3",
        "mask_iqr",
        "mask_voxel_count",
    ]

    rows = []
    for map_name, df_map in ok_df.groupby("map_name", sort=False):
        for metric in metrics:
            stats = summarize_series(df_map[metric])
            rows.append(
                {
                    "map_name": map_name,
                    "metric": metric,
                    **stats,
                }
            )

    out = pd.DataFrame(rows)

    map_order = [name for name, _ in CFG.map_defs]
    out["map_order"] = out["map_name"].map({m: i for i, m in enumerate(map_order)})
    out = out.sort_values(["map_order", "metric"]).drop(columns="map_order").reset_index(drop=True)

    return out


def build_status_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["map_name", "status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["map_name", "status"])
        .reset_index(drop=True)
    )


# Main
def main() -> None:
    cfg = CFG
    cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    df = load_ground_truth(cfg)
    patient_ids = natural_sort_patient_ids(df["Pat ID"].astype(str).tolist())
    df = df.set_index("Pat ID").loc[patient_ids].reset_index()

    rows: List[Dict[str, object]] = []
    total_steps = len(df) * len(cfg.map_defs)

    with tqdm(total=total_steps, desc="Processing patient-map pairs") as pbar:
        for _, patient_row in df.iterrows():
            for map_name, _ in cfg.map_defs:
                rows.append(process_patient_map(patient_row, map_name, cfg))
                pbar.update(1)

    raw_df = pd.DataFrame(rows)
    summary_df = build_summary_table(raw_df)
    status_df = build_status_table(raw_df)

    with pd.ExcelWriter(cfg.output_xlsx, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_patient_stats", index=False)
        summary_df.to_excel(writer, sheet_name="summary_by_map", index=False)
        status_df.to_excel(writer, sheet_name="status_counts", index=False)

    print(f"Saved Excel file to: {cfg.output_xlsx}")


if __name__ == "__main__":
    main()
