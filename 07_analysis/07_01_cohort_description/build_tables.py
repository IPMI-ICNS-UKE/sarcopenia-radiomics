import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    KIND_COHORT,
    KIND_CONT,
    KIND_DATA,
    KIND_SECTION,
    KIND_SUB,
    StatResult,
    TableModel,
    categorical_test,
    continuous_test,
    count_percent,
    format_pvalue,
    mean_sd,
    valid_denominator,
    value_range,
)

import config as C

logger = logging.getLogger(__name__)

COL = C.COLUMNS


# Small data helpers
def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        logger.warning("Column `%s` missing - treated as all-missing.", col)
        return pd.Series([np.nan] * len(df))
    return df[col]


def _binary_counts(series: pd.Series, positive) -> Tuple[int, int, int]:
    """Return (n_positive, n_negative, n_valid) for a binary column."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    pos = int((s[valid] == positive).sum())
    neg = int(valid.sum() - pos)
    return pos, neg, int(valid.sum())


def _continuous_pvalue(groups: List[pd.DataFrame], col: str, cfg: C.RunConfig) -> StatResult:
    data = [pd.to_numeric(_series(g, col), errors="coerce").to_numpy() for g in groups]
    return continuous_test(data, mode=cfg.continuous_test, alpha=C.NORMALITY_ALPHA)


def _binary_pvalue(groups: List[pd.DataFrame], col: str, positive, cfg: C.RunConfig) -> StatResult:
    table = []
    for g in groups:
        pos, neg, _ = _binary_counts(_series(g, col), positive)
        table.append([pos, neg])
    return categorical_test(
        table,
        mode=cfg.categorical_test,
        min_expected=C.MIN_EXPECTED_COUNT,
        n_resamples=C.MONTE_CARLO_RESAMPLES,
        seed=cfg.random_seed,
    )


def _sex_pvalue(groups: List[pd.DataFrame], cfg: C.RunConfig) -> StatResult:
    table = []
    for g in groups:
        s = pd.to_numeric(_series(g, COL.sex), errors="coerce")
        male = int((s == C.SEX_MALE_VALUE).sum())
        female = int((s == C.SEX_FEMALE_VALUE).sum())
        table.append([male, female])
    return categorical_test(
        table,
        mode=cfg.categorical_test,
        min_expected=C.MIN_EXPECTED_COUNT,
        n_resamples=C.MONTE_CARLO_RESAMPLES,
        seed=cfg.random_seed,
    )


def _tumor_label(value) -> Optional[str]:
    if pd.isna(value):
        return C.TUMOR_CUP_LABEL if C.TUMOR_MISSING_AS_CUP else None
    try:
        code = int(round(float(value)))
    except (TypeError, ValueError):
        return C.TUMOR_CUP_LABEL if C.TUMOR_MISSING_AS_CUP else None
    return C.TUMOR_TYPE_MAP.get(code, C.TUMOR_CUP_LABEL)


# Table 1 - Patient characteristics
def build_table1(
    groups: Dict[str, pd.DataFrame], cfg: C.RunConfig
) -> Tuple[TableModel, List[Tuple[str, str, StatResult]]]:
    g_tr = groups[C.GROUP_TRAINING]
    g_t1 = groups[C.GROUP_TEST1]
    g_t2 = groups[C.GROUP_TEST2]
    ordered = [g_tr, g_t1, g_t2]
    sizes = [len(g_tr), len(g_t1), len(g_t2)]

    header = [
        "Characteristic",
        f"{C.GROUP_LABELS[C.GROUP_TRAINING]}\n(n={sizes[0]})",
        f"{C.GROUP_LABELS[C.GROUP_TEST1]}\n(n={sizes[1]})",
        f"{C.GROUP_LABELS[C.GROUP_TEST2]}\n(n={sizes[2]})",
        "p-value",
    ]
    t = TableModel(
        title="Table 1. Patient characteristics.",
        header=header,
        sheet_name="Table 1 - Patient char",
    )
    methods: List[Tuple[str, str, StatResult]] = []

    d, pdec, cut = cfg.decimals, cfg.pvalue_decimals, C.PVALUE_SMALL_CUTOFF

    # Cohort row
    t.add(
        ["Cohort", C.COHORT1_DESCRIPTION, C.COHORT1_DESCRIPTION, C.COHORT2_DESCRIPTION, ""],
        KIND_COHORT,
    )

    # ---- Patient demographics ------------------------------------------ #
    t.add(["Patient demographics", "", "", "", ""], KIND_SECTION)

    def _continuous_block(label: str, col: str) -> None:
        res = _continuous_pvalue(ordered, col, cfg)
        methods.append((label, "Training vs Test 1 vs Test 2", res))
        t.add(
            [label]
            + [mean_sd(_series(g, col), d) for g in ordered]
            + [format_pvalue(res.p_value, pdec, cut)],
            KIND_DATA,
        )
        t.add(
            [""] + [value_range(_series(g, col), d) for g in ordered] + [""],
            KIND_CONT,
        )

    _continuous_block("Age (years)", COL.age)

    # Sex
    res_sex = _sex_pvalue(ordered, cfg)
    methods.append(("Sex", "Training vs Test 1 vs Test 2", res_sex))
    t.add(
        ["Sex, n (%)", "", "", "", format_pvalue(res_sex.p_value, pdec, cut)],
        KIND_DATA,
    )
    for sub_label, sex_value in (("Male", C.SEX_MALE_VALUE), ("Female", C.SEX_FEMALE_VALUE)):
        cells = [sub_label]
        for g in ordered:
            s = pd.to_numeric(_series(g, COL.sex), errors="coerce")
            denom = valid_denominator(s, len(g), cfg.percent_denominator)
            cells.append(count_percent(int((s == sex_value).sum()), denom, d, C.PERCENT_SYMBOL))
        cells.append("")
        t.add(cells, KIND_SUB)

    _continuous_block("BMI (kg / m2)", COL.bmi)

    # ---- Functional tests ---------------------------------------------- #
    t.add(["Functional tests", "", "", "", ""], KIND_SECTION)
    _continuous_block("Hand grip strength, kg", COL.hand_grip_cont)
    _continuous_block("Chair rise test time, s", COL.chair_rise_cont)

    def _binary_row(label: str, col: str, with_p: bool = True) -> None:
        res = _binary_pvalue(ordered, col, C.POSITIVE_VALUE, cfg)
        if with_p:
            methods.append((label, "Training vs Test 1 vs Test 2", res))
        cells = [label]
        for g in ordered:
            s = pd.to_numeric(_series(g, col), errors="coerce")
            denom = valid_denominator(s, len(g), cfg.percent_denominator)
            pos = int((s == C.POSITIVE_VALUE).sum())
            cells.append(count_percent(pos, denom, d, C.PERCENT_SYMBOL))
        cells.append(format_pvalue(res.p_value, pdec, cut) if with_p else "")
        t.add(cells, KIND_DATA)

    _binary_row("Sarcopenia according to hand grip test", COL.hand_grip_bin)
    _binary_row("Sarcopenia according to chair rise test", COL.chair_rise_bin)
    _binary_row(
        "Sarcopenia according to the composite definition",
        COL.sarcopenia_composite,
        with_p=cfg.compute_composite_pvalue,
    )

    return t, methods


# Table 2 - Clinical characteristics of cohort 1
def build_table2(groups: Dict[str, pd.DataFrame], cfg: C.RunConfig) -> TableModel:
    g_tr = groups[C.GROUP_TRAINING]
    g_t1 = groups[C.GROUP_TEST1]
    ordered = [g_tr, g_t1]
    sizes = [len(g_tr), len(g_t1)]
    d = cfg.decimals

    header = [
        "Characteristic",
        f"{C.GROUP_LABELS[C.GROUP_TRAINING]}\n(n={sizes[0]})",
        f"{C.GROUP_LABELS[C.GROUP_TEST1]}\n(n={sizes[1]})",
    ]
    t = TableModel(
        title="Table 2. Clinical characteristics of cohort 1.",
        header=header,
        sheet_name="Table 2 - Cohort 1",
    )

    # ECOG
    t.add(["ECOG at time of inclusion, n (%)", "", ""], KIND_SECTION)
    for cat in C.ECOG_CATEGORIES:
        cells = [str(cat)]
        for g in ordered:
            s = pd.to_numeric(_series(g, COL.ecog), errors="coerce")
            denom = int(s.notna().sum())
            cells.append(count_percent(int((s == cat).sum()), denom, d, C.PERCENT_SYMBOL))
        t.add(cells, KIND_SUB)

    # Tumor type
    t.add(["Tumor type, n (%)", "", ""], KIND_SECTION)
    tumor_counts = []
    for g in ordered:
        labels = _series(g, COL.tumor_type).map(_tumor_label)
        tumor_counts.append(labels.value_counts(dropna=True))
    for label in C.TUMOR_TYPE_ORDER:
        cells = [label]
        for vc in tumor_counts:
            denom = int(vc.sum())
            cells.append(count_percent(int(vc.get(label, 0)), denom, d, C.PERCENT_SYMBOL))
        t.add(cells, KIND_SUB)

    # Metastasis flags
    for label, col in (
        ("Primary metastasized", COL.primary_metastasized),
        ("Secondary metastasized", COL.secondary_metastasized),
    ):
        cells = [label]
        for g in ordered:
            s = pd.to_numeric(_series(g, col), errors="coerce")
            denom = valid_denominator(s, len(g), cfg.percent_denominator)
            cells.append(
                count_percent(int((s == C.POSITIVE_VALUE).sum()), denom, d, C.PERCENT_SYMBOL)
            )
        t.add(cells, KIND_DATA)

    # Chemo cycles (mean +- SD only, per template - no range row)
    t.add(
        ["Number of chemotherapeutic cycles"]
        + [mean_sd(_series(g, COL.num_chem_cycles), d) for g in ordered],
        KIND_DATA,
    )

    return t


# Table 3 - Clinical characteristics of cohort 2
def build_table3(groups: Dict[str, pd.DataFrame], cfg: C.RunConfig) -> TableModel:
    g_t2 = groups[C.GROUP_TEST2]
    d = cfg.decimals

    header = [
        "Characteristic",
        f"{C.GROUP_LABELS[C.GROUP_TEST2]}\n(n={len(g_t2)})",
    ]
    t = TableModel(
        title="Table 3. Clinical characteristics of cohort 2.",
        header=header,
        sheet_name="Table 3 - Cohort 2",
    )

    # MELD score
    t.add(["Meld score", mean_sd(_series(g_t2, COL.meld_score), d)], KIND_DATA)

    # Child-Pugh grade
    t.add(["Child Pugh grade", ""], KIND_SECTION)
    cp = _series(g_t2, COL.child_pugh_grade).astype("string").str.strip().str.upper()
    cp_denom = int(cp.notna().sum())
    for grade in C.CHILD_PUGH_CATEGORIES:
        t.add(
            [grade, count_percent(int((cp == grade).sum()), cp_denom, d, C.PERCENT_SYMBOL)],
            KIND_SUB,
        )

    # HCC / transplant listing
    for label, col in (
        ("Hepatocellular carcinoma", COL.hcc),
        ("Listed for the transplantation", COL.listed_for_trans),
    ):
        s = pd.to_numeric(_series(g_t2, col), errors="coerce")
        denom = valid_denominator(s, len(g_t2), cfg.percent_denominator)
        t.add(
            [label, count_percent(int((s == C.POSITIVE_VALUE).sum()), denom, d, C.PERCENT_SYMBOL)],
            KIND_DATA,
        )

    return t


# Statistical-methods sheet
def build_methods_table(methods: List[Tuple[str, str, StatResult]], cfg: C.RunConfig) -> TableModel:
    t = TableModel(
        title="Statistical methods (Table 1 group comparisons).",
        header=["Variable", "Comparison", "Statistical test", "p-value", "n used"],
        sheet_name="Statistical methods",
    )
    for var, comparison, res in methods:
        p_txt = "" if res.p_value is None or np.isnan(res.p_value) else f"{res.p_value:.6f}"
        t.add([var, comparison, res.test_name, p_txt, str(res.n_used)], KIND_DATA)
    note = (
        "Continuous: Shapiro-Wilk + Levene -> one-way ANOVA if normal & "
        "equal-variance, otherwise Kruskal-Wallis. Categorical: Pearson "
        "chi-square; Fisher exact (2x2) or seeded Monte-Carlo chi-square "
        "(r x c) when any expected count < %d." % C.MIN_EXPECTED_COUNT
    )
    t.add([note, "", "", "", ""], KIND_CONT)
    return t
