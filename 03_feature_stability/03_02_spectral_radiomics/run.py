import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import RCFG
from utils.icc import (
    build_stability_summary,
    choose_top_profile_features,
    compute_manual_vs_model_icc,
    compute_simulated_mask_icc,
    resolve_feature_selection,
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
from utils.plots import (
    plot_icc_heatmap,
    plot_icc_histogram,
    plot_profile_manual,
    plot_profile_simulated,
)
from utils.tables import build_original_mask_feature_table


def build_stable_table(df_all: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    return build_original_mask_feature_table(
        df=df_all,
        feature_cols=selected_features,
        keep_id_cols=RCFG.keep_id_cols_in_stable_table,
        split_col=RCFG.split_col,
    )


def run_one_map_type(
    df_cohort1: pd.DataFrame,
    df_cohort2: Optional[pd.DataFrame],
    map_type: str,
    feature_cols: List[str],
) -> Dict[str, object]:
    map_slug = map_type_to_slug(map_type)
    map_out_dir = RCFG.output_dir / map_slug
    tables_dir = map_out_dir / "tables"
    plots_dir = map_out_dir / "plots"
    map_out_dir.mkdir(parents=True, exist_ok=True)

    df_train = df_cohort1[df_cohort1[RCFG.split_col] == RCFG.train_split_name].copy()
    df_test = df_cohort1[df_cohort1[RCFG.split_col] == RCFG.temporal_test_split_name].copy()

    icc_sim_train = compute_simulated_mask_icc(
        df_train,
        feature_cols,
        RCFG.split_col,
        (RCFG.train_split_name,),
        RCFG.simulated_masks,
    )
    icc_sim_test = compute_simulated_mask_icc(
        df_test,
        feature_cols,
        RCFG.split_col,
        (RCFG.temporal_test_split_name,),
        RCFG.simulated_masks,
    )
    icc_manual_train = compute_manual_vs_model_icc(
        df_train,
        feature_cols,
        RCFG.split_col,
        (RCFG.train_split_name,),
        RCFG.manual_level_order,
        RCFG.manual_rater_order,
        by_level=False,
    )
    icc_manual_test = compute_manual_vs_model_icc(
        df_test,
        feature_cols,
        RCFG.split_col,
        (RCFG.temporal_test_split_name,),
        RCFG.manual_level_order,
        RCFG.manual_rater_order,
        by_level=False,
    )

    save_csv(
        icc_sim_train, tables_dir / f"icc_simulated_masks_{RCFG.train_split_name}_{map_slug}.csv"
    )
    save_csv(
        icc_sim_test,
        tables_dir / f"icc_simulated_masks_{RCFG.temporal_test_split_name}_{map_slug}.csv",
    )
    save_csv(
        icc_manual_train, tables_dir / f"icc_manual_vs_model_{RCFG.train_split_name}_{map_slug}.csv"
    )
    save_csv(
        icc_manual_test,
        tables_dir / f"icc_manual_vs_model_{RCFG.temporal_test_split_name}_{map_slug}.csv",
    )

    icc_sim_cohort2 = pd.DataFrame()
    if df_cohort2 is not None and not df_cohort2.empty:
        icc_sim_cohort2 = compute_simulated_mask_icc(
            df_cohort2,
            feature_cols,
            RCFG.split_col,
            (RCFG.external_test_split_name,),
            RCFG.simulated_masks,
        )
        save_csv(
            icc_sim_cohort2,
            tables_dir
            / f"icc_simulated_masks_{RCFG.external_test_split_name}_{map_slug}_cohort_2.csv",
        )

    stable_sets = build_stability_summary(
        icc_sim_train=icc_sim_train,
        icc_manual_train=icc_manual_train,
        threshold=RCFG.icc_threshold,
    )
    save_csv(stable_sets["simulated"], tables_dir / f"stable_features_simulated_{map_slug}.csv")
    save_csv(stable_sets["manual"], tables_dir / f"stable_features_manual_{map_slug}.csv")
    save_csv(stable_sets["both"], tables_dir / f"stable_features_both_{map_slug}.csv")
    save_csv(stable_sets["all"], tables_dir / f"stability_summary_all_features_{map_slug}.csv")

    final_selection_df = resolve_feature_selection(stable_sets, RCFG.selection_mode)
    final_features = final_selection_df["feature"].astype(str).tolist()

    save_csv(
        build_stable_table(df_cohort1, final_features),
        map_out_dir / f"stable_spectral_radiomics_{map_slug}.csv",
    )
    if df_cohort2 is not None and not df_cohort2.empty:
        save_csv(
            build_stable_table(df_cohort2, final_features),
            map_out_dir / f"stable_spectral_radiomics_{map_slug}_cohort_2.csv",
        )

    top_features = choose_top_profile_features(
        feature_cols=final_features if final_features else feature_cols,
        icc_sim_train=icc_sim_train,
        icc_manual_train=icc_manual_train,
        top_n=RCFG.top_n_profile_features,
    )
    save_csv(
        pd.DataFrame({"feature": top_features}), tables_dir / f"top_profile_features_{map_slug}.csv"
    )

    plot_icc_heatmap(
        icc_sim_train,
        f"ICC - {map_type} - simulated masks - train",
        plots_dir / f"heatmap_icc_simulated_masks_{RCFG.train_split_name}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_sim_test,
        f"ICC - {map_type} - simulated masks - test",
        plots_dir / f"heatmap_icc_simulated_masks_{RCFG.temporal_test_split_name}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_manual_train,
        f"ICC - {map_type} - manual vs model - train",
        plots_dir / f"heatmap_icc_manual_vs_model_{RCFG.train_split_name}_{map_slug}.png",
    )
    plot_icc_heatmap(
        icc_manual_test,
        f"ICC - {map_type} - manual vs model - test",
        plots_dir / f"heatmap_icc_manual_vs_model_{RCFG.temporal_test_split_name}_{map_slug}.png",
    )
    if not icc_sim_cohort2.empty:
        plot_icc_heatmap(
            icc_sim_cohort2,
            f"ICC - {map_type} - simulated masks - cohort 2",
            plots_dir
            / f"heatmap_icc_simulated_masks_{RCFG.external_test_split_name}_{map_slug}_cohort_2.png",
        )

    # ICC histograms (train set only, one PNG per scenario per spectral map).
    plot_icc_histogram(
        icc_df=icc_sim_train,
        map_type=map_type,
        scenario_label="Simulated masks ICC",
        threshold=RCFG.icc_threshold,
        out_path=plots_dir
        / f"histogram_icc_simulated_masks_{RCFG.train_split_name}_{map_slug}.png",
    )
    plot_icc_histogram(
        icc_df=icc_manual_train,
        map_type=map_type,
        scenario_label="Manual vs model ICC",
        threshold=RCFG.icc_threshold,
        out_path=plots_dir
        / f"histogram_icc_manual_vs_model_{RCFG.train_split_name}_{map_slug}.png",
    )

    for feature in top_features:
        plot_profile_simulated(
            df_train, feature, RCFG.train_split_name, RCFG.simulated_masks, plots_dir
        )
        plot_profile_simulated(
            df_test, feature, RCFG.temporal_test_split_name, RCFG.simulated_masks, plots_dir
        )
        plot_profile_manual(
            df_train,
            feature,
            RCFG.train_split_name,
            RCFG.manual_level_order,
            RCFG.manual_rater_order,
            plots_dir,
        )
        plot_profile_manual(
            df_test,
            feature,
            RCFG.temporal_test_split_name,
            RCFG.manual_level_order,
            RCFG.manual_rater_order,
            plots_dir,
        )
        if df_cohort2 is not None and not df_cohort2.empty:
            plot_profile_simulated(
                df_cohort2, feature, RCFG.external_test_split_name, RCFG.simulated_masks, plots_dir
            )

    return {
        "map_type": map_type,
        "map_type_slug": map_slug,
        "n_features_detected": len(feature_cols),
        "n_stable_simulated": len(stable_sets["simulated"]),
        "n_stable_manual": len(stable_sets["manual"]),
        "n_stable_both": len(stable_sets["both"]),
        "n_final_selected": len(final_features),
        "cohort_2_available": int(df_cohort2 is not None and not df_cohort2.empty),
    }


def main() -> None:
    RCFG.output_dir.mkdir(parents=True, exist_ok=True)

    df_gt = load_ground_truth(
        RCFG.ground_truth_xlsx,
        RCFG.filters,
        RCFG.patient_id_col_gt,
        RCFG.test_col_gt,
        split_col=RCFG.split_col,
        train_label=RCFG.train_split_name,
        test_label=RCFG.temporal_test_split_name,
    )

    required_cols = list(RCFG.metadata_cols)

    df_features_cohort1 = load_features_table(
        RCFG.cohort1_features_csv, required_cols, table_name="cohort 1 spectral radiomics"
    )
    df_features_cohort2 = load_features_table(
        RCFG.cohort2_features_csv, required_cols, table_name="cohort 2 spectral radiomics"
    )

    df_cohort1 = attach_splits(df_features_cohort1, df_gt, split_col=RCFG.split_col)
    df_cohort2 = assign_constant_split(
        df_features_cohort2, RCFG.split_col, RCFG.external_test_split_name
    )

    features_c1 = detect_feature_columns(df_cohort1, RCFG.metadata_cols, RCFG.split_col)
    features_c2 = detect_feature_columns(df_cohort2, RCFG.metadata_cols, RCFG.split_col)
    if set(features_c1) != set(features_c2):
        raise ValueError("Cohort 1 and cohort 2 spectral radiomics feature columns do not match.")

    feature_groups = build_feature_groups_by_map_type(features_c1, RCFG.target_map_types)

    summary_rows = []
    for map_type in RCFG.target_map_types:
        print(f"Processing map type: {map_type}...")
        if map_type not in feature_groups:
            continue
        summary_rows.append(
            run_one_map_type(df_cohort1, df_cohort2, map_type, feature_groups[map_type])
        )

    save_csv(
        pd.DataFrame(summary_rows), RCFG.output_dir / "tables" / "map_type_selection_summary.csv"
    )


if __name__ == "__main__":
    main()
