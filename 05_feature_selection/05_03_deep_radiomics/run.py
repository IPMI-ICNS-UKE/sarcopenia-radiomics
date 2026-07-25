from pathlib import Path
import sys
from typing import Iterable, List
import pandas as pd

STAGE_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

from utils.config import DeepRadiomicsConfig
from utils.io import (
    ensure_method_dirs,
    load_csv,
    load_ground_truth,
    merge_features_with_ground_truth,
    save_json,
)
from utils.signature import make_signature_dataset, summarize_signature


def _filter_raw_features(df: pd.DataFrame, cfg: DeepRadiomicsConfig) -> pd.DataFrame:
    required = [cfg.status_col, cfg.mask_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required raw deep-radiomics columns: {missing}")

    mask = (df[cfg.status_col].astype(str) == cfg.required_status_value) & (
        df[cfg.mask_col].astype(str) == cfg.required_mask_value
    )

    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(
            f"No raw deep-radiomics rows found with "
            f"{cfg.status_col}={cfg.required_status_value!r}, "
            f"{cfg.mask_col}={cfg.required_mask_value!r}"
        )

    return out.reset_index(drop=True)


def _deep_feature_columns(df: pd.DataFrame, source_map_name: str) -> List[str]:
    suffix = f"_{source_map_name}"
    feature_cols = [
        c for c in df.columns if c.endswith(suffix) and (c.startswith("d") or c.startswith("pc"))
    ]
    if not feature_cols:
        raise ValueError(f"No deep-radiomics feature columns found for map suffix {suffix!r}.")
    return sorted(feature_cols)


def _write_signature_outputs(
    cfg: DeepRadiomicsConfig,
    output_root: Path,
    map_name: str,
    cohort_key: str,
    signature_template: str,
    signature_df: pd.DataFrame,
    split_mismatches: pd.DataFrame,
    feature_cols: Iterable[str],
    input_path: Path,
    source_kind: str,
) -> None:
    dirs = ensure_method_dirs(output_root, map_name)
    suffix = cfg.get_output_suffix(cohort_key)
    signature_path = dirs["map_dir"] / signature_template.format(map_name=map_name).replace(
        ".csv", f"{suffix}.csv"
    )
    table_stem = signature_path.stem

    signature_df.to_csv(signature_path, index=False)
    split_mismatches.to_csv(dirs["tables_dir"] / f"split_mismatches_{table_stem}.csv", index=False)
    save_json(
        dirs["tables_dir"] / f"summary_{table_stem}.json",
        summarize_signature(
            signature_df=signature_df,
            selected_features=list(feature_cols),
            cfg=cfg,
            cohort_key=cohort_key,
            extra={
                "source_kind": source_kind,
                "input_path": str(input_path),
                "map_name": map_name,
            },
        ),
    )

    if cfg.verbose:
        print(f"Saved signature: {signature_path}")
        print(f"Rows={len(signature_df)} | Features={len(list(feature_cols))}")


def run_raw_one_map_one_cohort(
    cfg: DeepRadiomicsConfig,
    map_name: str,
    cohort_key: str,
) -> None:
    if cfg.verbose:
        print(f"\n=== 05_03 deep radiomics RAW | map={map_name} | cohort={cohort_key} ===")

    input_path = cfg.get_raw_features_path(cohort_key)
    gt_df = load_ground_truth(cfg, cohort_key)
    raw_df = load_csv(
        input_path,
        required_cols=[cfg.patient_id_col, cfg.status_col, cfg.mask_col],
        id_col=cfg.patient_id_col,
    )
    raw_df = _filter_raw_features(raw_df, cfg)

    source_map_name = cfg.get_raw_source_map_name(map_name)
    feature_cols = _deep_feature_columns(raw_df, source_map_name)
    feature_df = raw_df[[cfg.patient_id_col, *feature_cols]].copy()

    merged_df, split_mismatches = merge_features_with_ground_truth(feature_df, gt_df, cfg)
    signature_df = make_signature_dataset(merged_df, feature_cols, cfg)

    _write_signature_outputs(
        cfg=cfg,
        output_root=cfg.output_root_raw,
        map_name=map_name,
        cohort_key=cohort_key,
        signature_template=cfg.raw_signature_template,
        signature_df=signature_df,
        split_mismatches=split_mismatches,
        feature_cols=feature_cols,
        input_path=input_path,
        source_kind="raw",
    )


def main() -> None:
    cfg = DeepRadiomicsConfig()

    for map_name in cfg.maps:
        for cohort_key in (cfg.cohort_1_key, cfg.cohort_2_key):
            try:
                run_raw_one_map_one_cohort(cfg, map_name, cohort_key)
            except Exception as exc:
                print(f"Failed RAW for map={map_name}, cohort={cohort_key}: {exc!r}")


if __name__ == "__main__":
    main()
