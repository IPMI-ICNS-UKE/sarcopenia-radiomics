from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm


def icc2_1_from_wide(df_wide: pd.DataFrame) -> Dict[str, float]:
    """
    ICC(2,1): two-way random-effects, absolute agreement, single measurement.

    Rows are subjects/cases and columns are raters/repeated measurements.
    Missing rows are removed before ICC calculation.
    """
    df_wide = df_wide.dropna(axis=0, how="any")
    n, k = df_wide.shape

    if n < 2 or k < 2:
        return {
            "icc": np.nan,
            "n_subjects": int(n),
            "n_raters": int(k),
            "ms_rows": np.nan,
            "ms_cols": np.nan,
            "ms_err": np.nan,
        }

    x = df_wide.to_numpy(dtype=float)
    grand_mean = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((x - grand_mean) ** 2)
    ss_err = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_err = (n - 1) * (k - 1)

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_err = ss_err / df_err

    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    icc = np.nan if np.isclose(denom, 0.0) else (ms_rows - ms_err) / denom

    return {
        "icc": float(icc),
        "n_subjects": int(n),
        "n_raters": int(k),
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_err": float(ms_err),
    }


def build_complete_wide_table(
    df: pd.DataFrame,
    subject_col: str,
    rater_col: str,
    value_col: str,
    expected_raters: Sequence[str],
) -> pd.DataFrame:
    """Build a complete-case subject x rater table for one feature."""
    df_use = df[[subject_col, rater_col, value_col]].copy()
    df_use = df_use[df_use[rater_col].isin(expected_raters)].dropna(subset=[value_col])

    counts = (
        df_use.groupby([subject_col, rater_col], dropna=False)[value_col]
        .size()
        .reset_index(name="n_rows")
    )
    duplicated = counts[counts["n_rows"] > 1]
    if not duplicated.empty:
        raise ValueError(
            f"Found duplicated subject-rater rows for feature '{value_col}'. "
            "Expected one value per subject/case and mask/rater."
        )

    wide = df_use.pivot(index=subject_col, columns=rater_col, values=value_col)
    wide = wide.reindex(columns=list(expected_raters))
    return wide.dropna(axis=0, how="any")


def add_manual_case_columns(
    df: pd.DataFrame,
    manual_levels: Sequence[str],
    manual_raters: Sequence[str],
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    """Convert masks such as i_l2, j_l2, a_l2 into level, rater, and subject-case columns."""
    expected_masks = {f"{r}_{level}" for level in manual_levels for r in manual_raters}
    out = df[df["mask"].isin(expected_masks)].copy()
    if out.empty:
        out["level"] = pd.Series(dtype=str)
        out["manual_rater"] = pd.Series(dtype=str)
        out["subject_case_id"] = pd.Series(dtype=str)
        return out

    level_map = {f"{r}_{level}": level for level in manual_levels for r in manual_raters}
    rater_map = {f"{r}_{level}": r for level in manual_levels for r in manual_raters}

    out["level"] = out["mask"].map(level_map)
    out["manual_rater"] = out["mask"].map(rater_map)
    out["subject_case_id"] = out[patient_id_col].astype(str) + "__" + out["level"].astype(str)
    return out


def compute_simulated_mask_icc(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    split_col: str,
    split_names: Sequence[str],
    simulated_masks: Sequence[str],
    patient_id_col: str = "patient_id",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for split_name in split_names:
        print(f"Processing split: {split_name}...")
        df_split = df[df[split_col] == split_name].copy()
        df_mask = df_split[df_split["mask"].isin(simulated_masks)].copy()

        for feature_col in tqdm(feature_cols):
            wide = build_complete_wide_table(
                df=df_mask,
                subject_col=patient_id_col,
                rater_col="mask",
                value_col=feature_col,
                expected_raters=simulated_masks,
            )
            row: Dict[str, object] = {
                "scenario": "simulated_masks",
                "split": split_name,
                "group_name": "all_masks",
                "feature": feature_col,
                "raters": "|".join(simulated_masks),
                "icc_type": "ICC(2,1)",
            }
            stats = icc2_1_from_wide(wide)
            row.update(stats)
            row["status"] = "ok" if stats["n_subjects"] >= 2 else "insufficient_subjects"
            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["split", "icc", "feature"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def compute_manual_vs_model_icc(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    split_col: str,
    split_names: Sequence[str],
    manual_levels: Sequence[str],
    manual_raters: Sequence[str],
    patient_id_col: str = "patient_id",
    by_level: bool = False,
) -> pd.DataFrame:
    """
    Compute manual-vs-model ICC.

    by_level=False treats l2/l3/l4 as separate subject-cases and pools them.
    by_level=True computes separate ICCs for each level.
    """
    rows: List[Dict[str, object]] = []

    for split_name in split_names:
        df_split = df[df[split_col] == split_name].copy()
        df_manual = add_manual_case_columns(
            df_split,
            manual_levels=manual_levels,
            manual_raters=manual_raters,
            patient_id_col=patient_id_col,
        )

        groups = [(None, df_manual)]
        if by_level:
            groups = [
                (level, df_manual[df_manual["level"] == level].copy()) for level in manual_levels
            ]

        for level, df_group in groups:
            for feature_col in feature_cols:
                wide = build_complete_wide_table(
                    df=df_group,
                    subject_col="subject_case_id",
                    rater_col="manual_rater",
                    value_col=feature_col,
                    expected_raters=manual_raters,
                )
                row: Dict[str, object] = {
                    "scenario": "manual_vs_model",
                    "split": split_name,
                    "group_name": "all_levels" if level is None else level,
                    "feature": feature_col,
                    "level": "all_levels" if level is None else level,
                    "raters": "|".join(manual_raters),
                    "icc_type": "ICC(2,1)",
                }
                stats = icc2_1_from_wide(wide)
                row.update(stats)
                row["status"] = "ok" if stats["n_subjects"] >= 2 else "insufficient_subjects"
                rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["split", "group_name", "icc", "feature"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )


def build_stability_summary(
    icc_sim_train: pd.DataFrame,
    icc_manual_train: pd.DataFrame,
    threshold: float,
) -> Dict[str, pd.DataFrame]:
    sim = icc_sim_train[["feature", "icc"]].rename(columns={"icc": "icc_simulated_train"}).copy()
    man = icc_manual_train[["feature", "icc"]].rename(columns={"icc": "icc_manual_train"}).copy()

    merged = sim.merge(man, on="feature", how="outer")
    merged["stable_simulated"] = merged["icc_simulated_train"] > threshold
    merged["stable_manual"] = merged["icc_manual_train"] > threshold
    merged["stable_both"] = merged["stable_simulated"] & merged["stable_manual"]
    merged["mean_train_icc"] = merged[["icc_simulated_train", "icc_manual_train"]].mean(
        axis=1, skipna=False
    )

    return {
        "all": merged.sort_values(
            ["stable_both", "mean_train_icc", "feature"],
            ascending=[False, False, True],
        ).reset_index(drop=True),
        "simulated": merged[merged["stable_simulated"]]
        .sort_values(
            ["icc_simulated_train", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True),
        "manual": merged[merged["stable_manual"]]
        .sort_values(
            ["icc_manual_train", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True),
        "both": merged[merged["stable_both"]]
        .sort_values(
            ["mean_train_icc", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True),
    }


def resolve_feature_selection(
    stable_sets: Dict[str, pd.DataFrame], selection_mode: str
) -> pd.DataFrame:
    valid = {"simulated", "manual", "both"}
    if selection_mode not in valid:
        raise ValueError(f"selection_mode must be one of {sorted(valid)}, got: {selection_mode}")
    return stable_sets[selection_mode].copy()


def choose_top_profile_features(
    feature_cols: Sequence[str],
    icc_sim_train: pd.DataFrame,
    icc_manual_train: pd.DataFrame | None = None,
    top_n: int = 5,
) -> List[str]:
    """Choose profile features by mean available training ICC without filtering features out."""
    cols = ["feature", "icc"]
    rank = icc_sim_train[cols].rename(columns={"icc": "icc_simulated_train"}).copy()

    if icc_manual_train is not None and not icc_manual_train.empty:
        manual = icc_manual_train[cols].rename(columns={"icc": "icc_manual_train"}).copy()
        rank = rank.merge(manual, on="feature", how="outer")
        rank["mean_train_icc"] = rank[["icc_simulated_train", "icc_manual_train"]].mean(
            axis=1, skipna=True
        )
    else:
        rank["mean_train_icc"] = rank["icc_simulated_train"]

    rank = rank[rank["feature"].isin(feature_cols)]
    rank = rank.sort_values(["mean_train_icc", "feature"], ascending=[False, True])
    return rank["feature"].astype(str).head(top_n).tolist()
