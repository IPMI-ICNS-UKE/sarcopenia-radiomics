import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import (
    ICFG,
    MODEL_NAME,
    MODEL_DISPLAY_NAME,
    InterpretabilityConfig,
)
from dataset import load_splits
from forest import (
    load_fitted_model,
    compute_score_level_or,
    compute_feature_level_or,
)
from shap_analysis import compute_shap
from plots import plot_forest, plot_shap_beeswarm


# Helpers
def _save_csv(df: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.select_dtypes(include=["floating"]).columns:
        out[col] = out[col].round(decimals)
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")


# Forest plots
def run_forest(
    fitted: dict,
    splits: dict,
    cfg: InterpretabilityConfig,
    run_feature_level: bool,
) -> None:
    df_train = splits["train"]
    out_dir = cfg.paths.plots_dir
    tables_dir = cfg.paths.tables_dir
    show_p = not cfg.bootstrap.omit_pvalues

    # ---- Score level ----
    print("\n[Forest] Score-level OR computation ...")
    df_or_score = compute_score_level_or(fitted, df_train, cfg)
    _save_csv(df_or_score, tables_dir / "multivariable_or_train_score.csv")

    plot_forest(
        df_or_score,
        out_path=out_dir / "interpretability_train_forest_score.png",
        title=f"{MODEL_DISPLAY_NAME} — Forest plot (train set, score level)",
        level="score",
        show_pvalues=show_p,
        cfg=cfg,
    )

    # ---- Feature level ----
    if run_feature_level:
        print("\n[Forest] Feature-level OR computation ...")
        df_or_feat = compute_feature_level_or(fitted, df_train, cfg)
        _save_csv(df_or_feat, tables_dir / "multivariable_or_train_feature.csv")

        plot_forest(
            df_or_feat,
            out_path=out_dir / "interpretability_train_forest_feature.png",
            title=f"{MODEL_DISPLAY_NAME} — Forest plot (train set, feature level)",
            level="feature",
            show_pvalues=show_p,
            cfg=cfg,
        )


# SHAP bee swarm plots
def run_shap(
    fitted: dict,
    splits: dict,
    cfg: InterpretabilityConfig,
    run_feature_level: bool,
) -> None:
    out_dir = cfg.paths.plots_dir
    tables_dir = cfg.paths.tables_dir

    shap_splits = {k: v for k, v in splits.items() if k in ("test1", "test2")}

    for level in ["score", "feature"] if run_feature_level else ["score"]:
        print(f"\n[SHAP] {level.capitalize()}-level computation ...")
        shap_results = compute_shap(fitted, shap_splits, level=level)

        for split_label, df_shap in shap_results.items():
            # Save raw SHAP values
            _save_csv(
                df_shap,
                tables_dir / f"shap_values_{split_label}_{level}.csv",
            )

            # Determine output filename
            fname = f"interpretability_{split_label}_shap_{level}.png"
            title = (
                f"{MODEL_DISPLAY_NAME} — SHAP bee swarm "
                f"({'Cohort 1 test' if split_label == 'test1' else 'Cohort 2'}, "
                f"{level} level)"
            )

            plot_shap_beeswarm(
                df_shap,
                out_path=out_dir / fname,
                title=title,
                cfg=cfg,
            )


# Main
def main(
    run_feature_level: bool = True,
    n_boot_override: int = 0,
) -> None:
    cfg = ICFG

    # Optionally override n_boot from CLI
    if n_boot_override > 0:
        boot_cfg = replace(cfg.bootstrap, n_boot=n_boot_override)
        cfg = replace(cfg, bootstrap=boot_cfg)

    cfg.paths.make_all()

    print("=" * 70)
    print(f"Stage 07_07_interpretability — model: {MODEL_NAME}")
    print(f"  Display name: {MODEL_DISPLAY_NAME}")
    print(f"  Bootstrap iterations: {cfg.bootstrap.n_boot}")
    print(f"  Omit p-values: {cfg.bootstrap.omit_pvalues}")
    print(f"  Feature-level analysis: {run_feature_level}")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data splits ...")
    splits = load_splits(cfg)

    # 2. Load fitted model
    print("\n[2/4] Loading fitted model ...")
    fitted = load_fitted_model(cfg)
    print(f"  Model kind: {fitted['spec'].kind}")

    # 3. Forest plots (train set)
    print("\n[3/4] Forest plots ...")
    run_forest(fitted, splits, cfg, run_feature_level=run_feature_level)

    # 4. SHAP bee swarm (test sets)
    print("\n[4/4] SHAP bee swarm plots ...")
    run_shap(fitted, splits, cfg, run_feature_level=run_feature_level)

    print("\n" + "=" * 70)
    print("Stage 07_07_interpretability — DONE")
    print(f"  Outputs saved to: {cfg.paths.output_root}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 07_07_interpretability analysis.")
    parser.add_argument(
        "--no-feature-level",
        action="store_true",
        help="Skip feature-level analysis (faster).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=0,
        help="Override number of bootstrap iterations (0 = use config default).",
    )
    args = parser.parse_args()
    main(
        run_feature_level=not args.no_feature_level,
        n_boot_override=args.n_boot,
    )
