import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from config import RCFG, TASKS, RunConfig, TaskName
from dataset import build_dataset, COHORT_1, COHORT_2
from models import build_group_lookup, get_model_specs
from cv_base import _build_pred_df, run_cv_for_specs
from models_base import predict_model, resolve_model_specs
from stats import (
    bootstrap_calibration,
    bootstrap_classification_metrics,
    bootstrap_decision_curves,
    bootstrap_regression_metrics,
    classification_metrics,
    compute_decision_curves,
    evaluate_calibration,
    make_threshold_grid,
    regression_metrics,
    summarize_bootstrap_decision_curves,
    summarize_calibration_bootstrap,
)
from io_utils import (
    build_classification_metrics_table,
    build_regression_metrics_table,
    save_df,
    save_excel,
    save_pickle,
)
from plots import generate_classification_plots, generate_regression_plots


# Evaluation
def _evaluate_predictions(
    df_pred: pd.DataFrame,
    task_kind: str,
    n_features_by_model: Dict[str, int],
    run_cfg: RunConfig,
    n_boot: int,
) -> Dict:
    outputs: Dict = {}
    if task_kind == "classification":
        for model_name, g in df_pred.groupby("model_name"):
            thr = (
                float(g["selected_threshold"].dropna().iloc[0])
                if "selected_threshold" in g.columns and g["selected_threshold"].notna().any()
                else run_cfg.cv.classification_threshold
            )
            df_pred.loc[g.index, "selected_threshold"] = thr
            df_pred.loc[g.index, "pred_label"] = (g["pred_proba"] >= thr).astype(int)
        rows = []
        for mn, g in df_pred.groupby("model_name"):
            thr = float(g["selected_threshold"].dropna().iloc[0])
            m = classification_metrics(g["target"].to_numpy(), g["pred_proba"].to_numpy(), thr)
            m["n_features"] = n_features_by_model.get(mn, 1)
            rows.append({"model_name": mn, **m})
        point = pd.DataFrame(rows)
        boot = bootstrap_classification_metrics(
            df_pred,
            n_boot=n_boot,
            threshold=run_cfg.cv.classification_threshold,
            random_state=run_cfg.cv.random_state,
        )
        outputs["test_metrics"] = point.merge(boot.summary, on="model_name", how="left")
        outputs["test_bootstrap_samples"] = boot.samples
        outputs["test_bootstrap_summary"] = boot.summary

        cal = evaluate_calibration(
            df_pred, n_bins=run_cfg.calibration.n_bins, strategy=run_cfg.calibration.strategy
        )
        cal_boot_raw = bootstrap_calibration(
            df_pred, n_boot=n_boot, random_state=run_cfg.cv.random_state
        )
        outputs["test_calibration_metrics"] = cal.metrics
        outputs["test_calibration_curve"] = cal.curve
        outputs["test_calibration_bootstrap"] = summarize_calibration_bootstrap(cal_boot_raw)

        thr_grid = make_threshold_grid(
            run_cfg.decision_curve.threshold_start,
            run_cfg.decision_curve.threshold_stop,
            run_cfg.decision_curve.threshold_step,
        )
        dca = compute_decision_curves(df_pred, thresholds=thr_grid)
        dca_boot_raw = bootstrap_decision_curves(
            df_pred, thresholds=thr_grid, n_boot=n_boot, random_state=run_cfg.cv.random_state
        )
        outputs["test_decision_curve"] = dca.curves
        outputs["test_decision_curve_summary"] = dca.summary
        outputs["test_decision_curve_bootstrap"] = summarize_bootstrap_decision_curves(dca_boot_raw)
    else:
        rows = []
        for mn, g in df_pred.groupby("model_name"):
            p = n_features_by_model.get(mn, 1)
            m = regression_metrics(g["target"].to_numpy(), g["pred_value"].to_numpy(), p)
            m["n_features"] = p
            rows.append({"model_name": mn, **m})
        point = pd.DataFrame(rows)
        boot = bootstrap_regression_metrics(
            df_pred,
            n_boot=n_boot,
            n_features_by_model=n_features_by_model,
            random_state=run_cfg.cv.random_state,
        )
        outputs["test_metrics"] = point.merge(boot.summary, on="model_name", how="left")
        outputs["test_bootstrap_samples"] = boot.samples
        outputs["test_bootstrap_summary"] = boot.summary
    outputs["test_predictions"] = df_pred
    return outputs


# Saving
def _save_outputs(outputs, *, task, task_kind, postfix, run_cfg):
    dec = run_cfg.output.metric_decimals
    paths = run_cfg.paths
    kind_tag = "cls" if task_kind == "classification" else "reg"
    suffix = f"_{task}{postfix}"
    pred = outputs.get("test_predictions", pd.DataFrame())
    if not pred.empty:
        save_df(pred, paths.predictions_dir / f"test_{kind_tag}_predictions{suffix}.csv", dec)
    metrics = outputs.get("test_metrics", pd.DataFrame())
    if not metrics.empty:
        builder = (
            build_classification_metrics_table
            if task_kind == "classification"
            else build_regression_metrics_table
        )
        save_excel(
            builder(metrics, decimals=dec),
            paths.metrics_dir / f"test_{kind_tag}_metrics{suffix}.xlsx",
            sheet_name=f"{kind_tag}_metrics",
            decimals=dec,
        )
    for key in [
        "test_bootstrap_samples",
        "test_bootstrap_summary",
        "test_calibration_metrics",
        "test_calibration_curve",
        "test_calibration_bootstrap",
        "test_decision_curve",
        "test_decision_curve_summary",
        "test_decision_curve_bootstrap",
    ]:
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            save_df(df, paths.tables_dir / f"{key}{suffix}.csv", dec)


def _save_oof_outputs(outputs, *, task, task_kind, run_cfg, n_boot, all_specs):
    dec = run_cfg.output.metric_decimals
    paths = run_cfg.paths
    kind_tag = "cls" if task_kind == "classification" else "reg"
    oof_agg = outputs.get("oof_predictions_aggregated", pd.DataFrame())
    if not oof_agg.empty:
        save_df(
            oof_agg, paths.predictions_dir / f"train_oof_{kind_tag}_predictions_{task}.csv", dec
        )
    for key in ["cv_fold_metrics", "cv_summary", "coefficients", "model_info", "errors"]:
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            save_df(df, paths.tables_dir / f"{key}_{task}.csv", dec)
    if not oof_agg.empty:
        n_feat = {s.name: s.n_features for s in all_specs}
        if task_kind == "classification":
            rows = []
            for mn, g in oof_agg.groupby("model_name"):
                thr = (
                    float(g["selected_threshold"].dropna().iloc[0])
                    if "selected_threshold" in g.columns and g["selected_threshold"].notna().any()
                    else run_cfg.cv.classification_threshold
                )
                m = classification_metrics(g["target"].to_numpy(), g["pred_proba"].to_numpy(), thr)
                m["n_features"] = n_feat.get(mn, 1)
                rows.append({"model_name": mn, **m})
            pt = pd.DataFrame(rows)
            boot = bootstrap_classification_metrics(
                oof_agg,
                n_boot=n_boot,
                threshold=run_cfg.cv.classification_threshold,
                random_state=run_cfg.cv.random_state,
            )
            merged = pt.merge(boot.summary, on="model_name", how="left")
            pub = build_classification_metrics_table(merged, decimals=dec)
        else:
            rows = []
            for mn, g in oof_agg.groupby("model_name"):
                p = n_feat.get(mn, 1)
                m = regression_metrics(g["target"].to_numpy(), g["pred_value"].to_numpy(), p)
                m["n_features"] = p
                rows.append({"model_name": mn, **m})
            pt = pd.DataFrame(rows)
            boot = bootstrap_regression_metrics(
                oof_agg,
                n_boot=n_boot,
                n_features_by_model=n_feat,
                random_state=run_cfg.cv.random_state,
            )
            merged = pt.merge(boot.summary, on="model_name", how="left")
            pub = build_regression_metrics_table(merged, decimals=dec)
        save_excel(
            pub,
            paths.metrics_dir / f"train_oof_{kind_tag}_metrics_{task}.xlsx",
            sheet_name=f"oof_{kind_tag}_metrics",
            decimals=dec,
        )


# Main runner
def run_task(
    task: TaskName,
    run_cfg: RunConfig = RCFG,
    save_outputs: bool = True,
    n_boot: int = 2000,
) -> Dict:
    if task not in TASKS:
        raise KeyError(f"Unknown task: {task}")
    if run_cfg.debug.enabled:
        n_boot = run_cfg.debug.n_boot_override

    task_cfg = TASKS[task]
    task_kind = task_cfg.task_kind
    run_cfg.paths.make_all()

    print("=" * 80)
    print(f"Running 06_03_deep_radiomics — task: {task}")

    df = build_dataset(task=task, run_cfg=run_cfg, save_csv=True)
    n_rep = (
        run_cfg.debug.n_repeats_outer_override
        if run_cfg.debug.enabled
        else run_cfg.cv.n_repeats_outer
    )

    # Cohort-1 for training/test
    df_c1 = df[df["cohort"] == COHORT_1].reset_index(drop=True)

    # Build group lookup from actual dataframe columns
    group_lookup = build_group_lookup(df_c1, run_cfg)
    specs = get_model_specs(task=task, run_cfg=run_cfg, group_lookup=group_lookup)
    specs = resolve_model_specs(specs, group_lookup)
    n_features_by_model = {s.name: s.n_features for s in specs}

    outputs = run_cv_for_specs(
        df=df_c1,
        specs=specs,
        train_split_value="train",
        test_split_value="test",
        n_splits_outer=run_cfg.cv.n_splits_outer,
        n_repeats_outer=n_rep,
        n_splits_inner=run_cfg.cv.n_splits_inner,
        random_state=run_cfg.cv.random_state,
        l1_ratios=run_cfg.enet.l1_ratios,
        logistic_cs=run_cfg.enet.logistic_cs,
        linear_alphas=run_cfg.enet.linear_alphas,
        max_iter=run_cfg.enet.max_iter,
        convergence_tol=run_cfg.enet.convergence_tol,
        default_cls_threshold=run_cfg.cv.classification_threshold,
    )

    if save_outputs:
        _save_oof_outputs(
            outputs, task=task, task_kind=task_kind, run_cfg=run_cfg, n_boot=n_boot, all_specs=specs
        )

    # Cohort-1 test evaluation
    test_pred = outputs.get("test_predictions", pd.DataFrame())
    if not test_pred.empty:
        test_eval = _evaluate_predictions(
            test_pred.copy(), task_kind, n_features_by_model, run_cfg, n_boot
        )
        outputs.update(test_eval)
        if save_outputs:
            _save_outputs(outputs, task=task, task_kind=task_kind, postfix="", run_cfg=run_cfg)
            for mn, fitted in outputs.get("refit_models", {}).items():
                kind_tag = "cls" if task_kind == "classification" else "reg"
                save_pickle(fitted, run_cfg.paths.models_dir / kind_tag / f"{task}__{mn}.joblib")
            coefficients = outputs.get("coefficients", pd.DataFrame())
            if task_kind == "classification":
                generate_classification_plots(
                    test_predictions=test_eval["test_predictions"],
                    calibration_curve=test_eval.get("test_calibration_curve", pd.DataFrame()),
                    decision_curve=test_eval.get("test_decision_curve", pd.DataFrame()),
                    coefficients=coefficients,
                    plots_dir=run_cfg.paths.plots_dir,
                    task_name=task,
                    key_models=None,
                    interactive_plots_dir=(
                        run_cfg.paths.interactive_plots_dir
                        if run_cfg.output.save_interactive_plots
                        else None
                    ),
                    dca_xlim=run_cfg.decision_curve.plot_xlim,
                )
            else:
                generate_regression_plots(
                    test_predictions=test_eval["test_predictions"],
                    coefficients=coefficients,
                    plots_dir=run_cfg.paths.plots_dir,
                    task_name=task,
                    key_models=None,
                )

    # Cohort-2 evaluation
    df_c2 = df[df["cohort"] == COHORT_2].reset_index(drop=True)
    if not df_c2.empty:
        refit_models = outputs.get("refit_models", {})
        pred_tables: List[pd.DataFrame] = []
        for spec in specs:
            fitted = refit_models.get(spec.name)
            if fitted is None:
                continue
            try:
                pred = predict_model(fitted, df_c2)
                thr = (
                    fitted.get("selected_threshold", run_cfg.cv.classification_threshold)
                    if task_kind == "classification"
                    else np.nan
                )
                pred_tables.append(_build_pred_df(spec, df_c2, pred, thr))
            except Exception as exc:
                print(f"  Warning: cohort-2 prediction failed for {spec.name}: {exc}")

        if pred_tables:
            c2_pred = pd.concat(pred_tables, axis=0, ignore_index=True)
            c2_eval = _evaluate_predictions(
                c2_pred, task_kind, n_features_by_model, run_cfg, n_boot
            )
            outputs["cohort_2"] = c2_eval
            if save_outputs:
                _save_outputs(
                    c2_eval, task=task, task_kind=task_kind, postfix="_cohort_2", run_cfg=run_cfg
                )
                if task_kind == "classification":
                    generate_classification_plots(
                        test_predictions=c2_pred,
                        calibration_curve=c2_eval.get("test_calibration_curve", pd.DataFrame()),
                        decision_curve=c2_eval.get("test_decision_curve", pd.DataFrame()),
                        coefficients=outputs.get("coefficients", pd.DataFrame()),
                        plots_dir=run_cfg.paths.plots_dir,
                        task_name=f"{task}_cohort_2",
                        key_models=None,
                        interactive_plots_dir=(
                            run_cfg.paths.interactive_plots_dir
                            if run_cfg.output.save_interactive_plots
                            else None
                        ),
                        dca_xlim=run_cfg.decision_curve.plot_xlim,
                    )
                else:
                    generate_regression_plots(
                        test_predictions=c2_pred,
                        coefficients=outputs.get("coefficients", pd.DataFrame()),
                        plots_dir=run_cfg.paths.plots_dir,
                        task_name=f"{task}_cohort_2",
                        key_models=None,
                    )

    print(f"  Finished task: {task}")
    return outputs


if __name__ == "__main__":
    run_task("sarcopenia_composite_cls")
