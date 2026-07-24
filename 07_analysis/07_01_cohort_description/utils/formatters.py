import math
from typing import Sequence

import numpy as np
import pandas as pd

MISSING = ""  # text used when a value cannot be computed


def _num(x: float, decimals: int) -> str:
    return f"{round(float(x), decimals):.{decimals}f}"


def mean_sd(values: Sequence[float], decimals: int = 1) -> str:
    """'mean ± SD' over the non-missing values; '' if none."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return MISSING
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return f"{_num(np.mean(v), decimals)} \u00b1 {_num(sd, decimals)}"


def value_range(values: Sequence[float], decimals: int = 1) -> str:
    """'(min - max)' over the non-missing values; '' if none."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return MISSING
    return f"({_num(v.min(), decimals)} - {_num(v.max(), decimals)})"


def count_percent(
    count: int,
    denominator: int,
    decimals: int = 1,
    percent_symbol: str = "%",
) -> str:
    """'n (pp.p%)'; '' if the denominator is zero."""
    if denominator is None or denominator <= 0:
        return MISSING
    pct = 100.0 * count / denominator
    return f"{int(count)} ({_num(pct, decimals)}{percent_symbol})"


def format_pvalue(
    p: float,
    decimals: int = 3,
    small_cutoff: float = 0.001,
) -> str:
    """Round a p-value; show '<cutoff' below the cutoff; '' if NaN."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return MISSING
    if p < small_cutoff:
        return f"<{small_cutoff:g}"
    return f"{round(float(p), decimals):.{decimals}f}"


def valid_denominator(
    series: pd.Series,
    group_size: int,
    mode: str = "valid",
) -> int:
    """Denominator for n (%) cells.

    mode == 'valid' -> number of non-missing observations (default)
    mode == 'group' -> full group size
    """
    if mode == "group":
        return int(group_size)
    return int(pd.to_numeric(series, errors="coerce").notna().sum())
