from pathlib import Path
import sys
import pandas as pd

STAGE_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

from utils.config import SpectralSignatureConfig
from utils.io import (
    ensure_method_dirs,
    load_ground_truth,
    load_spectral_filtered_features,
    merge_features_with_ground_truth,
    save_json,
)
from utils.plots import (
    save_coefficient_path_plot,
    save_cv_tuning_plot,
    save_selected_coefficients_plot,
    save_selection_frequency_plot,
)
from utils.selection import (
    compute_coefficient_paths,
    compute_selection_frequency,
    extract_selected_coefficients,
    prepare_data_for_selection,
    summarize_selection,
    tune_l1_logistic,
)
from utils.signature import make_signature_dataset


def _select_features_from_training(
    prepared, search, coef_df, freq_df, cfg: SpectralSignatureConfig
):
    selected_features = coef_df.loc[coef_df["selected"], "feature"].tolist()
    if cfg.selection_frequency_threshold > 0:
        stable = set(
            freq_df.loc[
                freq_df["selection_frequency"] >= cfg.selection_frequency_threshold,
                "feature",
            ].tolist()
        )
        selected_features = [f for f in selected_features if f in stable]
    return selected_features


def _build_cohort_signature(
    cfg: SpectralSignatureConfig,
    map_name: str,
    cohort_key: str,
    selected_features,
):
    gt_df = load_ground_truth(cfg, cohort_key)
    feature_df = load_spectral_filtered_features(cfg, map_name, cohort_key)
    merged_df, split_mismatches = merge_features_with_ground_truth(feature_df, gt_df, cfg)
    signature_df = make_signature_dataset(merged_df, selected_features, cfg)
    return signature_df, split_mismatches, merged_df


def run_one_map(cfg: SpectralSignatureConfig, map_name: str) -> None:
    if cfg.verbose:
        print(f"\n=== 05_02_02 spectral signature | map={map_name} ===")

    gt_df = load_ground_truth(cfg, cfg.cohort_1_key)
    feature_df = load_spectral_filtered_features(cfg, map_name, cfg.cohort_1_key)
    merged_df, split_mismatches = merge_features_with_ground_truth(feature_df, gt_df, cfg)

    prepared = prepare_data_for_selection(merged_df, cfg)
    search = tune_l1_logistic(prepared.X_train, prepared.y_train, cfg)
    coef_df = extract_selected_coefficients(search, prepared.feature_cols_after)

    freq_df = compute_selection_frequency(
        prepared.X_train,
        prepared.y_train,
        prepared.feature_cols_after,
        best_c=float(search.best_params_["model__C"]),
        cfg=cfg,
    )

    selected_features = _select_features_from_training(prepared, search, coef_df, freq_df, cfg)

    cohort_1_signature = make_signature_dataset(
        pd.concat([prepared.train_df, prepared.test_df], axis=0, ignore_index=True),
        selected_features,
        cfg,
    )
    cohort_2_signature, split_mismatches_cohort_2, cohort_2_merged = _build_cohort_signature(
        cfg,
        map_name,
        cfg.cohort_2_key,
        selected_features,
    )

    path_df = compute_coefficient_paths(
        prepared.X_train, prepared.y_train, prepared.feature_cols_after, cfg
    )
    cv_results = pd.DataFrame(search.cv_results_).copy()
    cv_results["C"] = cv_results["param_model__C"].astype(float)

    dirs = ensure_method_dirs(cfg.output_root, map_name)
    suffix = cfg.get_output_suffix(cfg.cohort_2_key)
    signature_base = cfg.signature_template.format(map_name=map_name)

    cohort_1_signature.to_csv(dirs["map_dir"] / signature_base, index=False)
    cohort_2_signature.to_csv(
        dirs["map_dir"] / signature_base.replace(".csv", f"{suffix}.csv"), index=False
    )

    coef_df.to_csv(
        dirs["tables_dir"] / f"coefficients_spectral_radiomics_signature_{map_name}.csv",
        index=False,
    )
    freq_df.to_csv(
        dirs["tables_dir"] / f"selection_frequency_spectral_radiomics_signature_{map_name}.csv",
        index=False,
    )
    cv_results.to_csv(
        dirs["tables_dir"] / f"cv_results_spectral_radiomics_signature_{map_name}.csv", index=False
    )
    path_df.to_csv(
        dirs["tables_dir"] / f"coefficient_paths_spectral_radiomics_signature_{map_name}.csv",
        index=False,
    )
    split_mismatches.to_csv(
        dirs["tables_dir"] / f"split_mismatches_spectral_radiomics_signature_{map_name}.csv",
        index=False,
    )
    split_mismatches_cohort_2.to_csv(
        dirs["tables_dir"]
        / f"split_mismatches_spectral_radiomics_signature_{map_name}{suffix}.csv",
        index=False,
    )

    summary = summarize_selection(prepared, search, coef_df, freq_df, cfg, map_name)
    summary.update(
        {
            "n_signature_rows_cohort_1": int(len(cohort_1_signature)),
            "n_signature_rows_cohort_2": int(len(cohort_2_signature)),
            "n_train_cohort_2": int((cohort_2_merged[cfg.split_col] == "train").sum()),
            "n_test_cohort_2": int((cohort_2_merged[cfg.split_col] == "test").sum()),
            "selected_features_after_stability_filter": list(selected_features),
        }
    )
    save_json(dirs["tables_dir"] / f"summary_spectral_radiomics_signature_{map_name}.json", summary)

    best_c = float(search.best_params_["model__C"])
    n_nonzero_at_best_c = int(coef_df["selected"].sum())

    save_cv_tuning_plot(
        cv_results[["C", "mean_test_score", "std_test_score", "rank_test_score"]].copy(),
        dirs["plots_dir"] / f"tuning_spectral_radiomics_signature_{map_name}.png",
    )
    save_coefficient_path_plot(
        path_df,
        dirs["plots_dir"] / f"coefficient_paths_spectral_radiomics_signature_{map_name}.png",
        best_c=best_c,
        n_selected_features=n_nonzero_at_best_c,
    )
    save_selected_coefficients_plot(
        coef_df.loc[coef_df["selected"]].copy(),
        dirs["plots_dir"] / f"selected_coefficients_spectral_radiomics_signature_{map_name}.png",
    )
    save_selection_frequency_plot(
        freq_df,
        dirs["plots_dir"] / f"selection_frequency_spectral_radiomics_signature_{map_name}.png",
    )

    print(
        f"Initial filtered features: {len(prepared.feature_cols_before)} | "
        f"After preprocessing: {len(prepared.feature_cols_after)} | "
        f"Selected signature features: {len(selected_features)}"
    )
    print(f"Best C: {best_c} | Best CV {cfg.scoring}: {search.best_score_:.4f}")


def main() -> None:
    cfg = SpectralSignatureConfig()
    for map_name in cfg.maps:
        try:
            run_one_map(cfg, map_name)
        except Exception as exc:
            print(f"Failed for map={map_name}: {exc!r}")


if __name__ == "__main__":
    main()
