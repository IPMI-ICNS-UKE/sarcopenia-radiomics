import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RCFG
from utils.filtering_core import run_analysis_only_pipeline
from utils.io_utils import (
    add_postfix_to_filename,
    attach_split_labels,
    ensure_output_dirs,
    load_ground_truth_with_split,
    read_csv_checked,
)
from utils.plotting import (
    plot_clustered_abs_spearman_heatmap,
    plot_feature_counts,
    plot_spearman_dendrogram,
    plot_variance_distribution,
)


def process_dataset(dataset_key: str, input_csv: Path, ground_truth_path: Path, all_test: bool, postfix: str) -> None:
    out_dir, tables_dir, plots_dir = ensure_output_dirs(RCFG.output_root)
    required = [*RCFG.id_columns, *RCFG.feature_columns]
    df = read_csv_checked(input_csv, required_columns=required)

    gt = load_ground_truth_with_split(
        ground_truth_path, RCFG.patient_id_col_gt, RCFG.test_flag_col_gt, all_test=all_test
    )
    df = attach_split_labels(df, gt, RCFG.patient_id_col_data, RCFG.patient_id_col_gt)

    id_columns = [c for c in RCFG.id_columns if c in df.columns]
    feature_cols = list(RCFG.feature_columns)

    result = run_analysis_only_pipeline(
        df_all=df,
        feature_cols=feature_cols,
        id_columns=id_columns,
        rho_threshold=RCFG.spearman_rho_threshold,
        dataset_name=f"{RCFG.dataset_name}{postfix}",
        split_col=RCFG.split_col,
    )

    result.df_all.to_csv(out_dir / add_postfix_to_filename(RCFG.filtered_features_filename, postfix), index=False)

    result.variance_summary.to_csv(
        tables_dir / add_postfix_to_filename("variance_summary_baseline_ct.csv", postfix), index=False
    )
    result.correlation_summary.to_csv(
        tables_dir / add_postfix_to_filename("correlation_summary_baseline_ct.csv", postfix), index=False
    )
    result.overall_summary.to_csv(
        tables_dir / add_postfix_to_filename("analysis_overall_summary_baseline_ct.csv", postfix), index=False
    )
    if not result.corr_matrix_train.empty:
        result.corr_matrix_train.to_csv(
            tables_dir / add_postfix_to_filename("spearman_corr_matrix_train_baseline_ct.csv", postfix)
        )

    if RCFG.make_plots:
        title = f"{RCFG.dataset_name}{postfix}"
        plot_feature_counts(
            [len(feature_cols), len(feature_cols)],
            ["Input\nfeatures", "Retained\nfeatures"],
            plots_dir / add_postfix_to_filename("feature_counts_filtering_baseline_ct.png", postfix),
            title,
        )
        plot_variance_distribution(
            result.variance_summary,
            plots_dir / add_postfix_to_filename("variance_distribution_baseline_ct.png", postfix),
            title,
        )
        leaf = plot_spearman_dendrogram(
            result.corr_matrix_train,
            RCFG.spearman_rho_threshold,
            plots_dir / add_postfix_to_filename("spearman_dendrogram_baseline_ct.png", postfix),
            title,
        )
        if leaf is not None:
            leaf.to_csv(tables_dir / add_postfix_to_filename("dendrogram_leaf_order_baseline_ct.csv", postfix), index=False)
        heat = plot_clustered_abs_spearman_heatmap(
            result.corr_matrix_train,
            RCFG.spearman_rho_threshold,
            plots_dir / add_postfix_to_filename("clustered_abs_spearman_heatmap_baseline_ct.png", postfix),
            title,
        )
        if heat is not None:
            heat.to_csv(tables_dir / add_postfix_to_filename("clustered_heatmap_order_baseline_ct.csv", postfix), index=False)

    print("=" * 80)
    print(f"Processed 04_01_conventional_ct: {dataset_key}")
    print(f"Input file: {input_csv}")
    print(f"Retained features: {len(feature_cols)} / {len(feature_cols)}")
    print(f"Saved outputs to: {out_dir}")


def main() -> None:
    datasets = {
        "cohort_1": (
            RCFG.stable_root / RCFG.stable_features_filename,
            RCFG.ground_truth_path_cohort_1,
            False,
            "",
        ),
        "cohort_2": (
            RCFG.stable_root / RCFG.stable_features_filename_cohort_2,
            RCFG.ground_truth_path_cohort_2,
            True,
            RCFG.cohort_2_postfix,
        ),
    }
    for key, (path, ground_truth_path, all_test, postfix) in datasets.items():
        process_dataset(key, path, ground_truth_path, all_test, postfix)


if __name__ == "__main__":
    main()
