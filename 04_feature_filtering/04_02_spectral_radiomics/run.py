import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RCFG
from utils.filtering_core import run_spectral_filtering_pipeline, subset_with_ids
from utils.io_utils import (
    align_to_reference_features,
    attach_split_labels,
    ensure_cohort_column,
    ensure_output_dirs,
    identify_feature_columns,
    load_ground_truth_with_split,
    read_csv_checked,
)
from utils.plotting import (
    plot_clustered_abs_spearman_heatmap,
    plot_feature_counts,
    plot_spearman_dendrogram,
    plot_variance_distribution,
)


def build_map_paths(map_name: str):
    map_dir = RCFG.stable_root / map_name
    return (
        map_dir / RCFG.stable_features_pattern_cohort_1.format(map_name=map_name),
        map_dir / RCFG.stable_features_pattern_cohort_2.format(map_name=map_name),
        map_dir / "tables" / RCFG.stability_summary_pattern.format(map_name=map_name),
        RCFG.output_root / map_name,
    )


def process_map(map_name: str) -> pd.DataFrame:
    print("=" * 80)
    print(f"Processing spectral radiomics map: {map_name}")

    stable_c1, stable_c2, stability_csv, out_dir = build_map_paths(map_name)
    out_dir, tables_dir, plots_dir = ensure_output_dirs(out_dir)

    gt_c1 = load_ground_truth_with_split(
        RCFG.ground_truth_path_cohort_1, RCFG.patient_id_col_gt, RCFG.test_flag_col_gt
    )
    gt_c2 = load_ground_truth_with_split(
        RCFG.ground_truth_path_cohort_2, RCFG.patient_id_col_gt, all_test=True
    )

    df_c1 = read_csv_checked(stable_c1, required_columns=[RCFG.patient_id_col_data])
    df_c2 = read_csv_checked(stable_c2, required_columns=[RCFG.patient_id_col_data])
    df_stability = read_csv_checked(stability_csv, required_columns=["feature", "mean_train_icc"])

    df_c1 = ensure_cohort_column(df_c1, RCFG.cohort_col_data, RCFG.cohort_1_label)
    df_c2 = ensure_cohort_column(df_c2, RCFG.cohort_col_data, RCFG.cohort_2_label)

    df_c1 = attach_split_labels(df_c1, gt_c1, RCFG.patient_id_col_data, RCFG.patient_id_col_gt)
    df_c2 = attach_split_labels(df_c2, gt_c2, RCFG.patient_id_col_data, RCFG.patient_id_col_gt)

    id_columns = [c for c in RCFG.id_columns if c in df_c1.columns]
    df_c2 = align_to_reference_features(
        df_c1, df_c2, id_columns, label=f"spectral/{map_name}/cohort2"
    )
    feature_cols = identify_feature_columns(df_c1, id_columns)

    df_all_c1 = df_c1.copy()
    result = run_spectral_filtering_pipeline(
        df_all=df_all_c1,
        feature_cols=feature_cols,
        id_columns=id_columns,
        df_stability=df_stability,
        map_name=map_name,
        variance_threshold=RCFG.variance_threshold,
        min_unique_values=RCFG.min_unique_values,
        rho_threshold=RCFG.spearman_rho_threshold,
    )

    df_filtered_c2 = subset_with_ids(df_c2, id_columns, result.selected_features).reset_index(
        drop=True
    )

    result.df_all.to_csv(
        out_dir / RCFG.filtered_features_pattern.format(map_name=map_name), index=False
    )
    df_filtered_c2.to_csv(
        out_dir / RCFG.filtered_features_pattern_cohort_2.format(map_name=map_name), index=False
    )

    result.variance_summary.to_csv(
        tables_dir / f"near_zero_variance_summary_{map_name}.csv", index=False
    )
    result.correlation_summary.to_csv(
        tables_dir / f"correlation_clustering_summary_{map_name}.csv", index=False
    )
    result.overall_summary.to_csv(
        tables_dir / f"filtering_overall_summary_{map_name}.csv", index=False
    )
    pd.DataFrame({"selected_feature": result.selected_features}).to_csv(
        tables_dir / f"selected_feature_list_{map_name}.csv", index=False
    )
    if not result.corr_matrix_train.empty:
        result.corr_matrix_train.to_csv(
            tables_dir / f"spearman_corr_matrix_train_after_nzv_{map_name}.csv"
        )

    if RCFG.make_plots:
        row = result.overall_summary.iloc[0]
        plot_feature_counts(
            [
                int(row["n_initial_stable_features"]),
                int(row["n_after_nzv"]),
                int(row["n_after_correlation_filtering"]),
            ],
            ["Stable\nfeatures", "After NZV\nfiltering", "After correlation\nfiltering"],
            plots_dir / f"feature_counts_filtering_{map_name}.png",
            map_name,
        )
        plot_variance_distribution(
            result.variance_summary,
            plots_dir / f"variance_distribution_{map_name}.png",
            map_name,
            variance_threshold=RCFG.variance_threshold,
        )
        leaf = plot_spearman_dendrogram(
            result.corr_matrix_train,
            RCFG.spearman_rho_threshold,
            plots_dir / f"spearman_dendrogram_{map_name}.png",
            map_name,
            force_labels=True,
        )
        if leaf is not None:
            leaf.to_csv(tables_dir / f"dendrogram_leaf_order_{map_name}.csv", index=False)
        heat = plot_clustered_abs_spearman_heatmap(
            result.corr_matrix_train,
            RCFG.spearman_rho_threshold,
            plots_dir / f"clustered_abs_spearman_heatmap_{map_name}.png",
            map_name,
            max_features_for_plot=RCFG.heatmap_max_features,
        )
        if heat is not None:
            heat.to_csv(tables_dir / f"clustered_heatmap_order_{map_name}.csv", index=False)

    row = result.overall_summary.iloc[0]
    print(f"Initial stable features: {int(row['n_initial_stable_features'])}")
    print(f"After NZV: {int(row['n_after_nzv'])}")
    print(f"After correlation filtering: {int(row['n_after_correlation_filtering'])}")
    print(f"Saved outputs to: {out_dir}")
    return result.overall_summary


def main() -> None:
    summaries = [process_map(m) for m in RCFG.map_names]
    if summaries:
        RCFG.output_root.mkdir(parents=True, exist_ok=True)
        pd.concat(summaries, axis=0, ignore_index=True).to_csv(
            RCFG.output_root / "filtering_summary_all_maps.csv", index=False
        )


if __name__ == "__main__":
    main()
