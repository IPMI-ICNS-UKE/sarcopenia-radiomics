import sys
from pathlib import Path
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
    load_features_table,
    load_ground_truth,
    save_csv,
)
from utils.plots import plot_icc_heatmap, plot_profile_manual, plot_profile_simulated
from utils.tables import build_original_mask_feature_table


def run_single_dataset(
    df: pd.DataFrame,
    stable_csv_name: str,
    split_names: tuple[str, ...],
    file_postfix: str,
    include_manual: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stable_df = build_original_mask_feature_table(
        df=df,
        feature_cols=RCFG.feature_cols,
        keep_id_cols=RCFG.keep_id_cols,
        split_col=RCFG.split_col,
    )
    save_csv(stable_df, RCFG.out_dir / stable_csv_name)

    tables_dir = RCFG.out_dir / "tables"
    plots_dir = RCFG.out_dir / "plots"

    icc_sim = compute_simulated_mask_icc(
        df=df,
        feature_cols=RCFG.feature_cols,
        split_col=RCFG.split_col,
        split_names=split_names,
        simulated_masks=RCFG.simulated_masks,
        patient_id_col=RCFG.patient_id_col_features,
    )

    icc_manual = pd.DataFrame()
    if include_manual:
        icc_manual = compute_manual_vs_model_icc(
            df=df,
            feature_cols=RCFG.feature_cols,
            split_col=RCFG.split_col,
            split_names=split_names,
            manual_levels=RCFG.manual_levels,
            manual_raters=RCFG.manual_raters,
            patient_id_col=RCFG.patient_id_col_features,
            by_level=True,
        )

    for split_name in split_names:
        save_csv(
            icc_sim[icc_sim["split"] == split_name].reset_index(drop=True),
            tables_dir / f"icc_simulated_masks_{split_name}{file_postfix}.csv",
        )
        plot_icc_heatmap(
            icc_sim[icc_sim["split"] == split_name],
            title=f"Conventional CT ICC - simulated masks - {split_name}{file_postfix}",
            out_path=plots_dir / f"heatmap_icc_simulated_masks_{split_name}{file_postfix}.png",
        )

        top_features = choose_top_profile_features(
            feature_cols=RCFG.feature_cols,
            icc_sim_train=icc_sim[icc_sim["split"] == split_name],
            top_n=len(RCFG.feature_cols),
        )
        df_split = df[df[RCFG.split_col] == split_name].copy()
        for feature in top_features:
            plot_profile_simulated(
                df_split,
                feature=feature,
                split_name=f"{split_name}{file_postfix}",
                simulated_masks=RCFG.simulated_masks,
                out_dir=plots_dir,
                patient_id_col=RCFG.patient_id_col_features,
            )

        if include_manual:
            save_csv(
                icc_manual[icc_manual["split"] == split_name].reset_index(drop=True),
                tables_dir / f"icc_manual_vs_model_{split_name}{file_postfix}.csv",
            )
            plot_icc_heatmap(
                icc_manual[icc_manual["split"] == split_name],
                title=f"Conventional CT ICC - manual vs model - {split_name}{file_postfix}",
                out_path=plots_dir / f"heatmap_icc_manual_vs_model_{split_name}{file_postfix}.png",
            )
            for feature in top_features:
                plot_profile_manual(
                    df_split,
                    feature=feature,
                    split_name=f"{split_name}{file_postfix}",
                    manual_levels=RCFG.manual_levels,
                    manual_raters=RCFG.manual_raters,
                    out_dir=plots_dir,
                    patient_id_col=RCFG.patient_id_col_features,
                )

    return icc_sim, icc_manual


def main() -> None:
    RCFG.out_dir.mkdir(parents=True, exist_ok=True)

    df_splits = load_ground_truth(
        path=RCFG.ground_truth_xlsx,
        filters=RCFG.filters,
        patient_id_col_gt=RCFG.patient_id_col_gt,
        split_source_col=RCFG.split_source_col,
        patient_id_col=RCFG.patient_id_col_features,
        split_col=RCFG.split_col,
        train_label=RCFG.train_label,
        test_label=RCFG.test_label,
    )

    required_cols = [
        RCFG.patient_id_col_features,
        "cohort",
        "status",
        "mask",
        *RCFG.feature_cols,
    ]

    df_features_c1 = load_features_table(
        RCFG.input_features_csv,
        required_cols=required_cols,
        feature_cols=RCFG.feature_cols,
        keep_status_value=RCFG.keep_status_value,
        patient_id_col=RCFG.patient_id_col_features,
        table_name="cohort 1 conventional CT features table",
    )
    df_c1 = attach_splits(df_features_c1, df_splits, RCFG.patient_id_col_features, RCFG.split_col)
    run_single_dataset(
        df=df_c1,
        stable_csv_name=RCFG.stable_csv_name,
        split_names=(RCFG.train_label, RCFG.test_label),
        file_postfix="",
        include_manual=True,
    )

    df_features_c2 = load_features_table(
        RCFG.input_features_csv_cohort_2,
        required_cols=required_cols,
        feature_cols=RCFG.feature_cols,
        keep_status_value=RCFG.keep_status_value,
        patient_id_col=RCFG.patient_id_col_features,
        table_name="cohort 2 conventional CT features table",
    )
    df_c2 = assign_constant_split(df_features_c2, RCFG.split_col, RCFG.cohort_2_label)
    run_single_dataset(
        df=df_c2,
        stable_csv_name=RCFG.stable_csv_name_cohort_2,
        split_names=(RCFG.cohort_2_label,),
        file_postfix="_cohort_2",
        include_manual=False,
    )


if __name__ == "__main__":
    main()
