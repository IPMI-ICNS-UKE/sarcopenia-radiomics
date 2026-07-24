from dataclasses import replace

from config import COHORT1, COHORT2, RCFG
from extraction import run_cohort


def save_cohort_output(cohort_cfg, run_cfg, out_csv) -> None:
    out_df = run_cohort(cohort_cfg, run_cfg)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Backbone: {run_cfg.backbone_name}")
    print(f"Saved {cohort_cfg.cohort_name} [{run_cfg.mode}]: {out_csv}")
    print(out_df.head())


def main() -> None:
    cfg_2d = replace(RCFG, mode="2d")
    cfg_3d = replace(RCFG, mode="3d")

    # 2D extraction (single L3 / annotated slice)
    save_cohort_output(COHORT1, cfg_2d, RCFG.out_csv)
    save_cohort_output(COHORT2, cfg_2d, RCFG.out_csv_cohort2)

    # 3D extraction (whole muscle volume, slice-wise CLS embeddings
    # mean-pooled per map). Compute-heavy; comment out if not needed.
    save_cohort_output(COHORT1, cfg_3d, RCFG.out_csv_3d)
    save_cohort_output(COHORT2, cfg_3d, RCFG.out_csv_cohort2_3d)


if __name__ == "__main__":
    main()
