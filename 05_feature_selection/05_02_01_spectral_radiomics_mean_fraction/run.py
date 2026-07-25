from pathlib import Path
import sys

STAGE_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))
from utils.config import SpectralMeanFractionConfig
from utils.io import (
    ensure_method_dirs,
    load_csv,
    load_ground_truth,
    merge_features_with_ground_truth,
    save_json,
)
from utils.signature import make_signature_dataset, summarize_signature


def _filter_main_mask(df, cfg: SpectralMeanFractionConfig):
    out = df[df[cfg.mask_col].astype(str) == cfg.required_mask_value].copy()
    if out.empty:
        raise ValueError(f"No rows found with {cfg.mask_col} == {cfg.required_mask_value!r}.")
    return out.reset_index(drop=True)


def _mean_feature_name(map_name: str) -> str:
    return f"original_firstorder_Mean_{map_name}"


def run_one_map_one_cohort(
    cfg: SpectralMeanFractionConfig,
    map_name: str,
    cohort_key: str,
) -> None:
    if cfg.verbose:
        print(f"\n=== 05_02_01 mean_fraction | map={map_name} | cohort={cohort_key} ===")

    gt_df = load_ground_truth(cfg, cohort_key)
    raw_df = load_csv(
        cfg.get_raw_features_path(cohort_key),
        required_cols=[cfg.patient_id_col, cfg.cohort_col, cfg.mask_col],
        id_col=cfg.patient_id_col,
    )
    raw_df = _filter_main_mask(raw_df, cfg)

    source_col = _mean_feature_name(map_name)
    if source_col not in raw_df.columns:
        raise KeyError(f"Required mean fraction source feature not found: {source_col}")

    features_df = raw_df[[cfg.patient_id_col, source_col]].rename(
        columns={source_col: cfg.exported_feature_name}
    )

    merged_df, split_mismatches = merge_features_with_ground_truth(features_df, gt_df, cfg)
    signature_df = make_signature_dataset(
        merged_df,
        feature_cols=[cfg.exported_feature_name],
        cfg=cfg,
    )

    dirs = ensure_method_dirs(cfg.output_root, map_name)
    suffix = cfg.get_output_suffix(cohort_key)
    signature_path = dirs["map_dir"] / cfg.signature_template.format(map_name=map_name).replace(
        ".csv", f"{suffix}.csv"
    )
    mismatch_path = (
        dirs["tables_dir"]
        / f"split_mismatches_spectral_radiomics_mean_fraction_{map_name}{suffix}.csv"
    )
    summary_path = (
        dirs["tables_dir"] / f"summary_spectral_radiomics_mean_fraction_{map_name}{suffix}.json"
    )

    signature_df.to_csv(signature_path, index=False)
    split_mismatches.to_csv(mismatch_path, index=False)
    save_json(
        summary_path,
        summarize_signature(
            signature_df=signature_df,
            selected_features=[cfg.exported_feature_name],
            cfg=cfg,
            cohort_key=cohort_key,
            extra={
                "map_name": map_name,
                "mask_used": cfg.required_mask_value,
                "source_feature": source_col,
                "input_raw_features_path": str(cfg.get_raw_features_path(cohort_key)),
            },
        ),
    )

    if cfg.verbose:
        print(f"Saved signature: {signature_path}")
        print(f"Rows={len(signature_df)} | Features=1")


def main() -> None:
    cfg = SpectralMeanFractionConfig()
    for map_name in cfg.maps:
        for cohort_key in (cfg.cohort_1_key, cfg.cohort_2_key):
            try:
                run_one_map_one_cohort(cfg, map_name, cohort_key)
            except Exception as exc:
                print(f"Failed for map={map_name}, cohort={cohort_key}: {exc!r}")


if __name__ == "__main__":
    main()
