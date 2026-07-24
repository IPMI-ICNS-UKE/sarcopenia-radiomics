import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RCFG
from utils.icc import (
    choose_top_profile_features,
    compute_manual_vs_model_icc,
    compute_simulated_mask_icc,
)
from utils.io import (
    assign_constant_split,
    attach_splits,
    build_feature_groups_by_map_type,
    detect_feature_columns,
    load_features_table,
    load_ground_truth,
    map_type_to_slug,
    save_csv,
)
from utils.plots import plot_icc_heatmap, plot_profile_manual, plot_profile_simulated
from utils.tables import build_original_mask_feature_table


def build_stable_table(df_all: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Deep radiomics stage analyzes stability but does not select/drop features.
    return build_original_mask_feature_table(
        df=df_all,
        feature_cols=feature_cols,
        keep_id_cols=RCFG.keep_id_cols_in_stable_table,
        split_col=RCFG.split_col,
    )


def run_one_map_type(
    df_cohort1: pd.DataFrame,
    df_cohort2: pd.DataFrame,
    map_type: str,
    feature_cols: List[str],
) -> Dict[str, object]:
    map_slug = map_type_to_slug(map_type)
    map_out_dir = RCFG.output_dir / map_slug
    tables_dir = map_out_dir / "tables"
    plots_dir = map_out_dir / "plots"
    map_out_dir.mkdir(parents=True, exist_ok=True)

    df_train = df_cohort1[df_cohort1[RCFG.split_col] == RCFG.cohort1_train_label].copy()
    df_test = df_cohort1[df_cohort1[RCFG.split_col] == RCFG.cohort1_test_label].copy()
    df_test_cohort_2 = df_cohort2[df_cohort2[RCFG.split_col] == RCFG.cohort2_test_label].copy()

    icc_sim_train = compute_simulated_mask_icc(
        df_train,
        feature_cols,
        RCFG.split_col,
        (RCFG.cohort1_train_label,),
        RCFG.simulated_masks,
    )
    icc_sim_test = compute_simulated_mask_icc(
        df_test,
        feature_cols,
        RCFG.split_col,
        (RCFG.cohort1_test_label,),
        RCFG.simulated_masks,
    )
    icc_sim_test_cohort_2 = compute_simulated_mask_icc(
        df_test_cohort_2,
        feature_cols,
        RCFG.split_col,
        (RCFG.cohort2_test_label,),
        RCFG.simulated_masks,
    )

    icc_manual_train = compute_manual_vs_model_icc(
        df_train,
        feature_cols,
        RCFG.split_col,
        (RCFG.cohort1_train_label,),
        RCFG.manual_level_order,
        RCFG.manual_rater_order,
        by_level=False,
    )
    icc_manual_test = compute_manual_vs_model_icc(
        df_test,
        feature_cols,
        RCFG.split_col,
        (RCFG.cohort1_test_label,),
        RCFG.manual_level_order,
        RCFG.manual_rater_order,
        by_level=False,
    )

    save_csv(icc_sim_train, tables_dir / f"icc_simulated_masks_{RCFG.cohort1_train_label}_{map_slug}.csv")
    save_csv(icc_sim_test, tables_dir / f"icc_simulated_masks_{RCFG.cohort1_test_label}_{map_slug}.csv")
    save_csv(
        icc_sim_test_cohort_2,
        tables_dir / f"icc_simulated_masks_{RCFG.cohort2_test_label}_{map_slug}_cohort_2.csv",
    )
    save_csv(icc_manual_train, tables_dir / f"icc_manual_vs_model_{RCFG.cohort1_train_label}_{map_slug}.csv")
    save_csv(icc_manual_test, tables_dir / f"icc_manual_vs_model_{RCFG.cohort1_test_label}_{map_slug}.csv")

    # No feature filtering for deep radiomics: all original-mask features are retained.
    stable_table_cohort1 = build_stable_table(df_cohort1, feature_cols)
    stable_table_cohort2 = build_stable_table(df_cohort2, feature_cols)
    save_csv(stable_table_cohort1, map_out_dir / f"stable_deep_radiomics_{map_slug}.csv")
    save_csv(stable_table_cohort2, map_out_dir / f"stable_deep_radiomics_{map_slug}_cohort_2.csv")

    top_features = choose_top_profile_features(
        feature_cols=feature_cols,
        icc_sim_train=icc_sim_train,
        icc_manual_train=icc_manual_train,
        top_n=RCFG.top_n_profile_features,
    )
    save_csv(pd.DataFrame({"feature": top_features}), tables_dir / f"top_profile_features_{map_slug}.csv")

    plot_icc_heatmap(
        icc_sim_train,
        f"ICC - {map_type} - simulated masks - train",
        plots_dir / f"heatmap_icc_simulated_masks_{RCFG.cohort1_train_label}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_sim_test,
        f"ICC - {map_type} - simulated masks - test",
        plots_dir / f"heatmap_icc_simulated_masks_{RCFG.cohort1_test_label}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_sim_test_cohort_2,
        f"ICC - {map_type} - simulated masks - cohort 2",
        plots_dir / f"heatmap_icc_simulated_masks_{RCFG.cohort2_test_label}_{map_slug}_cohort_2.png",
    )
    plot_icc_heatmap(
        icc_manual_train,
        f"ICC - {map_type} - manual vs model - train",
        plots_dir / f"heatmap_icc_manual_vs_model_{RCFG.cohort1_train_label}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_manual_test,
        f"ICC - {map_type} - manual vs model - test",
        plots_dir / f"heatmap_icc_manual_vs_model_{RCFG.cohort1_test_label}_{map_slug}.png",
    )

    for feature in top_features:
        plot_profile_simulated(df_train, feature, RCFG.cohort1_train_label, RCFG.simulated_masks, plots_dir)
        plot_profile_simulated(df_test, feature, RCFG.cohort1_test_label, RCFG.simulated_masks, plots_dir)
        plot_profile_simulated(df_test_cohort_2, feature, RCFG.cohort2_test_label, RCFG.simulated_masks, plots_dir)
        plot_profile_manual(
            df_train, feature, RCFG.cohort1_train_label, RCFG.manual_level_order, RCFG.manual_rater_order, plots_dir
        )
        plot_profile_manual(
            df_test, feature, RCFG.cohort1_test_label, RCFG.manual_level_order, RCFG.manual_rater_order, plots_dir
        )

    return {
        "map_type": map_type,
        "map_type_slug": map_slug,
        "n_features_detected": len(feature_cols),
        "n_rows_stable_table_cohort1": len(stable_table_cohort1),
        "n_rows_stable_table_cohort2": len(stable_table_cohort2),
    }


def validate_feature_sets(df_cohort1: pd.DataFrame, df_cohort2: pd.DataFrame) -> Dict[str, List[str]]:
    feature_cols_cohort1 = detect_feature_columns(
        df_cohort1,
        metadata_cols=RCFG.metadata_cols,
        split_col=RCFG.split_col,
        prefix="d",
    )
    feature_cols_cohort2 = detect_feature_columns(
        df_cohort2,
        metadata_cols=RCFG.metadata_cols,
        split_col=RCFG.split_col,
        prefix="d",
    )

    if set(feature_cols_cohort1) != set(feature_cols_cohort2):
        only_in_cohort1 = sorted(set(feature_cols_cohort1) - set(feature_cols_cohort2))
        only_in_cohort2 = sorted(set(feature_cols_cohort2) - set(feature_cols_cohort1))
        raise ValueError(
            "Cohort 1 and cohort 2 deep-radiomics feature sets do not match. "
            f"Only in cohort 1: {only_in_cohort1[:10]}. "
            f"Only in cohort 2: {only_in_cohort2[:10]}."
        )

    return build_feature_groups_by_map_type(feature_cols_cohort1, RCFG.target_map_types)


def main() -> None:
    RCFG.output_dir.mkdir(parents=True, exist_ok=True)

    df_gt_cohort1 = load_ground_truth(
        RCFG.ground_truth_xlsx,
        RCFG.filters,
        RCFG.patient_id_col_gt,
        RCFG.test_col_gt,
        split_col=RCFG.split_col,
        train_label=RCFG.cohort1_train_label,
        test_label=RCFG.cohort1_test_label,
    )

    df_features_cohort1 = load_features_table(
        RCFG.features_csv_cohort1,
        required_cols=RCFG.metadata_cols,
        table_name="cohort 1 deep radiomics features table",
    )
    df_features_cohort2 = load_features_table(
        RCFG.features_csv_cohort2,
        required_cols=RCFG.metadata_cols,
        table_name="cohort 2 deep radiomics features table",
    )

    df_cohort1 = attach_splits(df_features_cohort1, df_gt_cohort1, split_col=RCFG.split_col)
    df_cohort2 = assign_constant_split(df_features_cohort2, RCFG.split_col, RCFG.cohort2_test_label)

    feature_groups = validate_feature_sets(df_cohort1, df_cohort2)
    if not feature_groups:
        raise ValueError("No map-type feature groups were detected from target_map_types.")

    summary_rows: List[Dict[str, object]] = []
    for map_type in RCFG.target_map_types:
        if map_type not in feature_groups:
            continue
        print(f"Running deep radiomics stability for map type: {map_type}")
        summary_rows.append(run_one_map_type(df_cohort1, df_cohort2, map_type, feature_groups[map_type]))

    save_csv(pd.DataFrame(summary_rows), RCFG.output_dir / "tables" / "map_type_stability_summary.csv")
    print(f"Processed map types: {len(summary_rows)}")
    print(f"Saved outputs to: {RCFG.output_dir}")


if __name__ == "__main__":
    main()
