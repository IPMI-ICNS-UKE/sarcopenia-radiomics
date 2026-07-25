import logging
import os
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


def _read_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground-truth table not found: {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _filter_used(df: pd.DataFrame, use_col: str, use_value) -> pd.DataFrame:
    if use_col not in df.columns:
        logger.warning("`%s` column missing - keeping all rows.", use_col)
        return df.copy()
    used = pd.to_numeric(df[use_col], errors="coerce")
    return df.loc[used == use_value].copy()


def load_groups(cfg, columns, gc) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict with keys 'training', 'test1', 'test2'.

    Parameters
    ----------
    cfg     : RunConfig instance (paths)
    columns : Columns dataclass instance (column names)
    gc      : GroupConst instance (use / split values, group keys)
    """
    # ---- cohort 1 -------------------------------------------------------- #
    c1 = _read_table(cfg.cohort1_table)
    c1 = _coerce_numeric(
        c1,
        [
            columns.age,
            columns.sex,
            columns.bmi,
            columns.hand_grip_bin,
            columns.hand_grip_cont,
            columns.chair_rise_bin,
            columns.chair_rise_cont,
            columns.sarcopenia_composite,
            columns.test_temporal,
            columns.use,
            columns.ecog,
            columns.tumor_type,
            columns.primary_metastasized,
            columns.secondary_metastasized,
            columns.num_chem_cycles,
        ],
    )
    c1 = _filter_used(c1, columns.use, gc.use_flag_value)

    if columns.test_temporal not in c1.columns:
        raise KeyError(
            f"`{columns.test_temporal}` is required to split cohort 1 "
            f"into training / testing set 1."
        )
    tt = pd.to_numeric(c1[columns.test_temporal], errors="coerce")
    training = c1.loc[tt == gc.training_value].copy()
    test1 = c1.loc[tt == gc.test1_value].copy()

    # ---- cohort 2 -------------------------------------------------------- #
    c2 = _read_table(cfg.cohort2_table)
    c2 = _coerce_numeric(
        c2,
        [
            columns.age,
            columns.sex,
            columns.bmi,
            columns.hand_grip_bin,
            columns.hand_grip_cont,
            columns.chair_rise_bin,
            columns.chair_rise_cont,
            columns.sarcopenia_composite,
            columns.use,
            columns.meld_score,
            columns.hcc,
            columns.listed_for_trans,
        ],
    )
    test2 = _filter_used(c2, columns.use, gc.use_flag_value)

    logger.info(
        "Group sizes - training: %d | test1: %d | test2: %d",
        len(training),
        len(test1),
        len(test2),
    )

    return {
        gc.group_training: training.reset_index(drop=True),
        gc.group_test1: test1.reset_index(drop=True),
        gc.group_test2: test2.reset_index(drop=True),
    }
