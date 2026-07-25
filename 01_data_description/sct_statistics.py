from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind


CONTINUOUS_VARS = [
    "age",
    "bmi",
    "ct_muscle_mass_2d",  # SMI 2D
    "ct_muscle_density_2d",  # MRA 2D
    "ct_muscle_fat_segm_2d",  # FF 2D
]

DISPLAY_NAMES = {
    "age": "Age (years)",
    "bmi": "BMI",
    "ct_muscle_mass_2d": "SMI 2D",
    "ct_muscle_density_2d": "MRA 2D",
    "ct_muscle_fat_segm_2d": "FF 2D",
}

ECOG_LEVELS = [0, 1, 2]


def _to_numeric_clean(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").dropna()


def _format_mean_sd(x: pd.Series, decimals: int = 1) -> str:
    x = _to_numeric_clean(x)
    if len(x) == 0:
        return "NA"

    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan

    if np.isnan(sd):
        return f"{mean:.{decimals}f} ± NA"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def _format_range(x: pd.Series, decimals: int = 1) -> str:
    x = _to_numeric_clean(x)
    if len(x) == 0:
        return "NA"
    return f"{float(x.min()):.{decimals}f}–{float(x.max()):.{decimals}f}"


def _format_iqr(x: pd.Series, decimals: int = 1) -> str:
    x = _to_numeric_clean(x)
    if len(x) == 0:
        return "NA"

    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return f"{q1:.{decimals}f}–{q3:.{decimals}f}"


def _format_n_pct(n: int, total: int, decimals: int = 1) -> str:
    if total == 0:
        return "NA"
    pct = 100.0 * n / total
    return f"{n} ({pct:.{decimals}f}%)"


def format_p_value(p: Optional[float]) -> str:
    if p is None or pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def infer_sex_mapping(sex_series: pd.Series) -> Tuple[object, object]:
    """
    Infer sex coding.
    Expected:
    - 0 = female
    - 1 = male
    """
    s = sex_series.dropna()

    if len(s) == 0:
        return 0, 1

    if s.dtype == object:
        s_low = s.astype(str).str.strip().str.lower()
        uniq = sorted(s_low.unique().tolist())

        if "female" in uniq and "male" in uniq:
            return "female", "male"
        if "f" in uniq and "m" in uniq:
            return "f", "m"

        return uniq[0], uniq[-1]

    uniq = sorted(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
    if 0 in uniq and 1 in uniq:
        return 0, 1
    if len(uniq) >= 2:
        return uniq[0], uniq[-1]

    return 0, 1


def count_sex(df: pd.DataFrame, sex_col: str, female_value, male_value) -> Dict[str, int]:
    s = df[sex_col]

    if isinstance(female_value, str) and isinstance(male_value, str):
        s = s.astype(str).str.strip().str.lower()
        female_n = int((s == str(female_value).lower()).sum())
        male_n = int((s == str(male_value).lower()).sum())
    else:
        female_n = int((s == female_value).sum())
        male_n = int((s == male_value).sum())

    return {
        "female": female_n,
        "male": male_n,
        "total": len(df),
    }


def shapiro_is_normal(x: pd.Series, alpha: float = 0.05) -> Optional[bool]:
    """
    Returns:
        True  -> distribution is compatible with normality
        False -> non-normal
        None  -> insufficient data or test failed
    """
    x = _to_numeric_clean(x)

    if len(x) < 3:
        return None

    if len(x) > 5000:
        x = x.sample(5000, random_state=42)

    try:
        _, p = shapiro(x)
        return bool(p >= alpha)
    except Exception:
        return None


def compare_continuous(x1: pd.Series, x2: pd.Series) -> Tuple[Optional[float], str]:
    """
    Two-group comparison for continuous variables:
    - Welch t-test if both groups look normal
    - Mann-Whitney U otherwise
    """
    x1 = _to_numeric_clean(x1)
    x2 = _to_numeric_clean(x2)

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, "NA"

    normal1 = shapiro_is_normal(x1)
    normal2 = shapiro_is_normal(x2)

    if normal1 is True and normal2 is True:
        _, p = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
        return float(p), "Welch t-test"

    try:
        _, p = mannwhitneyu(x1, x2, alternative="two-sided")
        return float(p), "Mann-Whitney U"
    except Exception:
        return np.nan, "NA"


def compare_binary_from_counts(
    pos1: int,
    neg1: int,
    pos2: int,
    neg2: int,
) -> Tuple[Optional[float], str]:
    """
    Two-group comparison for binary categorical variables.
    Uses:
    - Chi-square if expected counts are adequate
    - Fisher exact otherwise
    """
    table = np.array(
        [
            [pos1, neg1],
            [pos2, neg2],
        ],
        dtype=int,
    )

    if table.sum() == 0:
        return np.nan, "NA"

    try:
        _, p_chi, _, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            _, p_fisher = fisher_exact(table)
            return float(p_fisher), "Fisher exact"
        return float(p_chi), "Chi-square"
    except Exception:
        return np.nan, "NA"


def compare_ecog(
    ecog1: pd.Series,
    ecog2: pd.Series,
    levels: List[int] = ECOG_LEVELS,
) -> Tuple[Optional[float], str]:
    """
    Compares ECOG distribution between two groups.

    Preferred:
    - Chi-square across ECOG 0/1/2

    Fallback for sparse counts:
    - Collapse to ECOG 0 vs ECOG >=1 and use Fisher exact
    """
    ecog1 = pd.to_numeric(ecog1, errors="coerce").dropna()
    ecog2 = pd.to_numeric(ecog2, errors="coerce").dropna()

    if len(ecog1) == 0 or len(ecog2) == 0:
        return np.nan, "NA"

    table = np.array(
        [
            [(ecog1 == lvl).sum() for lvl in levels],
            [(ecog2 == lvl).sum() for lvl in levels],
        ],
        dtype=int,
    )

    try:
        _, p_chi, _, expected = chi2_contingency(table)
        if (expected < 5).any():
            collapsed = np.array(
                [
                    [int((ecog1 == 0).sum()), int((ecog1 >= 1).sum())],
                    [int((ecog2 == 0).sum()), int((ecog2 >= 1).sum())],
                ],
                dtype=int,
            )
            _, p_fisher = fisher_exact(collapsed)
            return float(p_fisher), "Fisher exact (ECOG 0 vs ≥1)"
        return float(p_chi), "Chi-square"
    except Exception:
        return np.nan, "NA"


def summarize_continuous(x: pd.Series, decimals: int = 1) -> Dict[str, str]:
    return {
        "mean_sd": _format_mean_sd(x, decimals=decimals),
        "range": _format_range(x, decimals=decimals),
        "iqr": _format_iqr(x, decimals=decimals),
    }


def summarize_ecog_levels(
    df: pd.DataFrame,
    ecog_col: str = "ecog",
    levels: List[int] = ECOG_LEVELS,
    decimals: int = 1,
) -> Dict[int, str]:
    if ecog_col not in df.columns:
        return {lvl: "NA" for lvl in levels}

    s = pd.to_numeric(df[ecog_col], errors="coerce")
    total = int(s.notna().sum())

    out = {}
    for lvl in levels:
        n = int((s == lvl).sum())
        out[lvl] = _format_n_pct(n, total, decimals=decimals)

    return out


def _empty_stat_block() -> Dict[str, str]:
    return {"mean_sd": "NA", "range": "NA", "iqr": "NA"}


def _group_columns(group_name: str) -> Tuple[str, str, str]:
    return (
        f"{group_name} mean ± SD",
        f"{group_name} range",
        f"{group_name} IQR",
    )


def _comparison_columns(left_name: str, right_name: str) -> Tuple[str, str]:
    return (
        f"P-value: {left_name} vs {right_name}",
        f"Statistical test: {left_name} vs {right_name}",
    )


def _set_group_summary(row: Dict[str, str], group_name: str, stats: Dict[str, str]) -> None:
    mean_sd_col, range_col, iqr_col = _group_columns(group_name)
    row[mean_sd_col] = stats["mean_sd"]
    row[range_col] = stats["range"]
    row[iqr_col] = stats["iqr"]


def _set_group_value(row: Dict[str, str], group_name: str, value: str) -> None:
    mean_sd_col, range_col, iqr_col = _group_columns(group_name)
    row[mean_sd_col] = value
    row[range_col] = ""
    row[iqr_col] = ""


def _set_comparison(
    row: Dict[str, str], left_name: str, right_name: str, p_value, test_name: str
) -> None:
    p_col, test_col = _comparison_columns(left_name, right_name)
    row[p_col] = format_p_value(p_value)
    row[test_col] = test_name


def build_data_description_table_three_groups(
    df_group1: pd.DataFrame,
    df_group2: pd.DataFrame,
    df_group3: pd.DataFrame,
    group1_name: str,
    group2_name: str,
    group3_name: str,
    sex_col: str = "sex",
    ecog_col: str = "ecog",
    female_value=None,
    male_value=None,
    group3_has_ecog: bool = False,
    decimals: int = 1,
) -> pd.DataFrame:
    """
    Builds a long-format baseline table for three groups.

    Output includes:
    - descriptive statistics for all three datasets
    - pairwise p-values and test names for all dataset comparisons
    - ECOG rows marked as NA for group 3 when ECOG is unavailable
    """
    rows = []

    if female_value is None or male_value is None:
        female_value, male_value = infer_sex_mapping(
            pd.concat(
                [df_group1[sex_col], df_group2[sex_col], df_group3[sex_col]],
                axis=0,
            )
        )

    comparisons = [
        (df_group1, df_group2, group1_name, group2_name),
        (df_group1, df_group3, group1_name, group3_name),
        (df_group2, df_group3, group2_name, group3_name),
    ]

    # Continuous variables
    for var in CONTINUOUS_VARS:
        if not all(var in df.columns for df in [df_group1, df_group2, df_group3]):
            continue

        stats1 = summarize_continuous(df_group1[var], decimals=decimals)
        stats2 = summarize_continuous(df_group2[var], decimals=decimals)
        stats3 = summarize_continuous(df_group3[var], decimals=decimals)

        row = {"Parameter": DISPLAY_NAMES.get(var, var)}
        _set_group_summary(row, group1_name, stats1)
        _set_group_summary(row, group2_name, stats2)
        _set_group_summary(row, group3_name, stats3)

        for left_df, right_df, left_name, right_name in comparisons:
            p_value, test_name = compare_continuous(left_df[var], right_df[var])
            _set_comparison(row, left_name, right_name, p_value, test_name)

        rows.append(row)

    # Sex rows
    sex1 = count_sex(df_group1, sex_col, female_value, male_value)
    sex2 = count_sex(df_group2, sex_col, female_value, male_value)
    sex3 = count_sex(df_group3, sex_col, female_value, male_value)

    row_female = {"Parameter": "Sex: female, n (%)"}
    _set_group_value(
        row_female, group1_name, _format_n_pct(sex1["female"], sex1["total"], decimals)
    )
    _set_group_value(
        row_female, group2_name, _format_n_pct(sex2["female"], sex2["total"], decimals)
    )
    _set_group_value(
        row_female, group3_name, _format_n_pct(sex3["female"], sex3["total"], decimals)
    )

    p_sex_12, test_sex_12 = compare_binary_from_counts(
        pos1=sex1["female"], neg1=sex1["male"], pos2=sex2["female"], neg2=sex2["male"]
    )
    p_sex_13, test_sex_13 = compare_binary_from_counts(
        pos1=sex1["female"], neg1=sex1["male"], pos2=sex3["female"], neg2=sex3["male"]
    )
    p_sex_23, test_sex_23 = compare_binary_from_counts(
        pos1=sex2["female"], neg1=sex2["male"], pos2=sex3["female"], neg2=sex3["male"]
    )

    _set_comparison(row_female, group1_name, group2_name, p_sex_12, test_sex_12)
    _set_comparison(row_female, group1_name, group3_name, p_sex_13, test_sex_13)
    _set_comparison(row_female, group2_name, group3_name, p_sex_23, test_sex_23)
    rows.append(row_female)

    row_male = {"Parameter": "Sex: male, n (%)"}
    _set_group_value(row_male, group1_name, _format_n_pct(sex1["male"], sex1["total"], decimals))
    _set_group_value(row_male, group2_name, _format_n_pct(sex2["male"], sex2["total"], decimals))
    _set_group_value(row_male, group3_name, _format_n_pct(sex3["male"], sex3["total"], decimals))
    for left_name, right_name in [
        (group1_name, group2_name),
        (group1_name, group3_name),
        (group2_name, group3_name),
    ]:
        _set_comparison(row_male, left_name, right_name, np.nan, "")
    rows.append(row_male)

    # ECOG rows
    ecog_stats_1 = summarize_ecog_levels(
        df_group1, ecog_col=ecog_col, levels=ECOG_LEVELS, decimals=decimals
    )
    ecog_stats_2 = summarize_ecog_levels(
        df_group2, ecog_col=ecog_col, levels=ECOG_LEVELS, decimals=decimals
    )
    ecog_stats_3 = (
        summarize_ecog_levels(df_group3, ecog_col=ecog_col, levels=ECOG_LEVELS, decimals=decimals)
        if group3_has_ecog
        else {lvl: "NA" for lvl in ECOG_LEVELS}
    )

    p_ecog_12, test_ecog_12 = compare_ecog(
        df_group1[ecog_col], df_group2[ecog_col], levels=ECOG_LEVELS
    )

    for idx, level in enumerate(ECOG_LEVELS):
        row = {"Parameter": f"ECOG {level}, n (%)"}
        _set_group_value(row, group1_name, ecog_stats_1[level])
        _set_group_value(row, group2_name, ecog_stats_2[level])
        _set_group_value(row, group3_name, ecog_stats_3[level])

        if idx == 0:
            _set_comparison(row, group1_name, group2_name, p_ecog_12, test_ecog_12)
        else:
            _set_comparison(row, group1_name, group2_name, np.nan, "")

        _set_comparison(row, group1_name, group3_name, np.nan, "NA (ECOG unavailable in cohort 2)")
        _set_comparison(row, group2_name, group3_name, np.nan, "NA (ECOG unavailable in cohort 2)")
        rows.append(row)

    out_df = pd.DataFrame(rows)

    order = [
        "Age (years)",
        "BMI",
        "SMI 2D",
        "MRA 2D",
        "FF 2D",
        "Sex: female, n (%)",
        "Sex: male, n (%)",
        "ECOG 0, n (%)",
        "ECOG 1, n (%)",
        "ECOG 2, n (%)",
    ]
    out_df["__order"] = out_df["Parameter"].map({k: i for i, k in enumerate(order)})
    out_df = (
        out_df.sort_values("__order", na_position="last")
        .drop(columns="__order")
        .reset_index(drop=True)
    )

    return out_df
