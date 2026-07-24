import json
from pathlib import Path
from typing import Dict, Iterable, Tuple
import pandas as pd

from .config import BaseStage05Config, SpectralSignatureConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_method_dirs(output_root: Path, map_name: str | None = None) -> Dict[str, Path]:
    root = ensure_dir(output_root)
    tables_dir = ensure_dir(root / "tables")
    if map_name is None:
        return {"root": root, "tables": tables_dir}

    map_dir = ensure_dir(root / map_name)
    return {
        "root": root,
        "map_dir": map_dir,
        "tables_dir": ensure_dir(map_dir / "tables"),
        "plots_dir": ensure_dir(map_dir / "plots"),
    }


def save_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_ground_truth(
    cfg: BaseStage05Config,
    cohort_key: str,
) -> pd.DataFrame:
    path = cfg.get_ground_truth_path(cohort_key)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found for {cohort_key}: {path}")

    df = pd.read_excel(path)
    required = [cfg.id_col_gt, cfg.temporal_flag_col, cfg.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in ground truth for {cohort_key}: {missing}")

    df = df.rename(columns={cfg.id_col_gt: cfg.patient_id_col}).copy()
    df[cfg.patient_id_col] = df[cfg.patient_id_col].astype(str)
    df[cfg.cohort_col] = cfg.get_cohort_value(cohort_key)

    if cohort_key == cfg.cohort_2_key:
        # Cohort 2 is the external hold-out cohort for feature-selection outputs.
        df[cfg.split_col] = "test"
    else:
        df[cfg.split_col] = df[cfg.temporal_flag_col].map({0: "train", 1: "test"})
        if df[cfg.split_col].isna().any():
            bad = df.loc[df[cfg.split_col].isna(), [cfg.patient_id_col, cfg.temporal_flag_col]]
            raise ValueError(
                f"Could not derive split from {cfg.temporal_flag_col} for rows:\n"
                f"{bad.to_string(index=False)}"
            )

    return df.reset_index(drop=True)


def load_csv(path: Path, required_cols: Iterable[str], id_col: str = "patient_id") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    df = df.copy()
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str)
    return df.reset_index(drop=True)


def merge_features_with_ground_truth(
    features_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    cfg: BaseStage05Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gt_cols = [cfg.patient_id_col, cfg.temporal_flag_col, cfg.split_col, cfg.target_col]
    merged = features_df.merge(
        gt_df[gt_cols],
        on=cfg.patient_id_col,
        how="inner",
        suffixes=("", "_gt"),
    )
    if merged.empty:
        raise ValueError("No overlapping patients after feature/ground-truth merge.")

    if cfg.split_col in features_df.columns:
        mismatch_mask = merged[cfg.split_col].astype(str) != merged[f"{cfg.split_col}_gt"].astype(str)
        split_mismatches = merged.loc[
            mismatch_mask,
            [cfg.patient_id_col, cfg.split_col, f"{cfg.split_col}_gt"],
        ].copy()
        merged[cfg.split_col] = merged[f"{cfg.split_col}_gt"]
        merged = merged.drop(columns=[f"{cfg.split_col}_gt"])
    else:
        split_mismatches = pd.DataFrame(
            columns=[cfg.patient_id_col, cfg.split_col, f"{cfg.split_col}_gt"]
        )

    merged = merged.rename(columns={cfg.target_col: "task_target"}).copy()

    duplicated = merged[merged.duplicated(subset=[cfg.patient_id_col], keep=False)]
    if not duplicated.empty:
        ids = duplicated[[cfg.patient_id_col]].drop_duplicates().to_string(index=False)
        raise ValueError(f"Duplicate patient rows found after merge:\n{ids}")

    return merged.reset_index(drop=True), split_mismatches.reset_index(drop=True)


def resolve_map_dir(filtered_root: Path, map_name: str) -> Path:
    direct = filtered_root / map_name
    if direct.exists():
        return direct
    if not filtered_root.exists():
        raise FileNotFoundError(f"Filtered root does not exist: {filtered_root}")

    candidates = {p.name.casefold(): p for p in filtered_root.iterdir() if p.is_dir()}
    lookup_keys = {
        map_name.casefold(),
        map_name.replace("keV", "kev").casefold(),
        map_name.replace("kev", "keV").casefold(),
    }
    for key in lookup_keys:
        if key in candidates:
            return candidates[key]

    raise FileNotFoundError(f"Map directory not found for map={map_name!r} under {filtered_root}")


def resolve_file_from_template(map_dir: Path, template: str, map_name: str) -> Path:
    expected = {
        template.format(map_name=map_name),
        template.format(map_name=map_name.replace("keV", "kev")),
        template.format(map_name=map_name.replace("kev", "keV")),
    }
    for name in expected:
        candidate = map_dir / name
        if candidate.exists():
            return candidate

    files_by_name = {p.name.casefold(): p for p in map_dir.iterdir() if p.is_file()}
    for name in expected:
        candidate = files_by_name.get(name.casefold())
        if candidate is not None:
            return candidate

    raise FileNotFoundError(f"Feature file not found in {map_dir}. Tried: {sorted(expected)}")


def get_spectral_filtered_feature_path(
    cfg: SpectralSignatureConfig,
    map_name: str,
    cohort_key: str,
) -> Path:
    map_dir = resolve_map_dir(cfg.filtered_root, map_name)
    template = (
        cfg.feature_file_template
        if cohort_key == cfg.cohort_1_key
        else cfg.feature_file_template_cohort_2
    )
    return resolve_file_from_template(map_dir, template, map_name)


def load_spectral_filtered_features(
    cfg: SpectralSignatureConfig,
    map_name: str,
    cohort_key: str,
) -> pd.DataFrame:
    path = get_spectral_filtered_feature_path(cfg, map_name, cohort_key)
    required = [cfg.patient_id_col]
    df = load_csv(path, required_cols=required, id_col=cfg.patient_id_col)
    if cfg.cohort_col not in df.columns:
        df[cfg.cohort_col] = cfg.get_cohort_value(cohort_key)
    return df
