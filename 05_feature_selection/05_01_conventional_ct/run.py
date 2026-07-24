from pathlib import Path
import sys

STAGE_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))
from utils.config import ConventionalCTConfig
from utils.io import ensure_method_dirs, load_csv, load_ground_truth, merge_features_with_ground_truth, save_json
from utils.signature import make_signature_dataset, summarize_signature


def run_one_cohort(cfg: ConventionalCTConfig, cohort_key: str) -> None:
    if cfg.verbose:
        print(f"\n=== 05_01 conventional CT | cohort={cohort_key} ===")

    gt_df = load_ground_truth(cfg, cohort_key)
    feature_df = load_csv(
        cfg.get_feature_path(cohort_key),
        required_cols=[cfg.patient_id_col, *cfg.feature_cols],
        id_col=cfg.patient_id_col,
    )

    merged_df, split_mismatches = merge_features_with_ground_truth(feature_df, gt_df, cfg)
    signature_df = make_signature_dataset(merged_df, cfg.feature_cols, cfg)

    dirs = ensure_method_dirs(cfg.output_root)
    suffix = cfg.get_output_suffix(cohort_key)
    signature_name = cfg.signature_filename.replace(".csv", f"{suffix}.csv")
    signature_path = dirs["root"] / signature_name
    mismatch_path = dirs["tables"] / f"split_mismatches_baseline_ct{suffix}.csv"
    summary_path = dirs["tables"] / f"summary_baseline_ct{suffix}.json"

    signature_df.to_csv(signature_path, index=False)
    split_mismatches.to_csv(mismatch_path, index=False)
    save_json(
        summary_path,
        summarize_signature(
            signature_df=signature_df,
            selected_features=list(cfg.feature_cols),
            cfg=cfg,
            cohort_key=cohort_key,
            extra={"input_feature_path": str(cfg.get_feature_path(cohort_key))},
        ),
    )

    if cfg.verbose:
        print(f"Saved signature: {signature_path}")
        print(f"Rows={len(signature_df)} | Features={len(cfg.feature_cols)}")


def main() -> None:
    cfg = ConventionalCTConfig()
    for cohort_key in (cfg.cohort_1_key, cfg.cohort_2_key):
        try:
            run_one_cohort(cfg, cohort_key)
        except Exception as exc:
            print(f"Failed for cohort={cohort_key}: {exc!r}")


if __name__ == "__main__":
    main()
