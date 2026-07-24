import logging
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatResult:
    p_value: float
    test_name: str
    n_used: int


# Continuous
def continuous_test(
    groups: Sequence[Sequence[float]],
    mode: str = "auto",
    alpha: float = 0.05,
) -> StatResult:
    """Compare a continuous variable across >=2 groups."""
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[~np.isnan(g)] for g in clean]
    clean = [g for g in clean if g.size > 0]
    n_used = int(sum(g.size for g in clean))

    if len(clean) < 2 or any(g.size < 2 for g in clean):
        return StatResult(np.nan, "not tested (insufficient data)", n_used)

    if mode == "anova":
        use_anova = True
        reason = "forced"
    elif mode == "kruskal":
        use_anova = False
        reason = "forced"
    else:  # auto
        normal = True
        for g in clean:
            if g.size < 3 or np.ptp(g) == 0:
                normal = False
                break
            try:
                if stats.shapiro(g).pvalue < alpha:
                    normal = False
                    break
            except Exception:
                normal = False
                break
        equal_var = True
        if normal:
            try:
                equal_var = stats.levene(*clean, center="median").pvalue >= alpha
            except Exception:
                equal_var = False
        use_anova = normal and equal_var
        reason = "auto"

    try:
        if use_anova:
            p = stats.f_oneway(*clean).pvalue
            name = "One-way ANOVA"
        else:
            p = stats.kruskal(*clean).pvalue
            name = "Kruskal-Wallis"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Continuous test failed: %s", exc)
        return StatResult(np.nan, "not tested (error)", n_used)

    if reason == "auto":
        name += " (auto)"
    return StatResult(float(p), name, n_used)


# Categorical
def _monte_carlo_chi2(
    table: np.ndarray, n_resamples: int, seed: int
) -> float:
    """Permutation p-value for an r x c contingency table.

    The group labels are permuted across all observations while the overall
    category distribution is held fixed; the chi-square statistic is
    recomputed each time.  p = P(stat_perm >= stat_observed).
    """
    rng = np.random.default_rng(seed)
    obs_stat = stats.chi2_contingency(table, correction=False)[0]

    # Reconstruct the raw (group, category) sample from the counts.
    n_groups, n_cats = table.shape
    group_labels, cat_labels = [], []
    for gi in range(n_groups):
        for ci in range(n_cats):
            cnt = int(round(table[gi, ci]))
            group_labels.extend([gi] * cnt)
            cat_labels.extend([ci] * cnt)
    group_labels = np.asarray(group_labels)
    cat_labels = np.asarray(cat_labels)
    n = group_labels.size
    if n == 0:
        return np.nan

    ge = 0
    for _ in range(n_resamples):
        perm = rng.permutation(group_labels)
        perm_tab = np.zeros_like(table, dtype=float)
        for gi in range(n_groups):
            mask = perm == gi
            for ci in range(n_cats):
                perm_tab[gi, ci] = np.count_nonzero(cat_labels[mask] == ci)
        if perm_tab.sum(axis=0).min() == 0 or perm_tab.sum(axis=1).min() == 0:
            stat = obs_stat  # degenerate resample -> count as >=
        else:
            stat = stats.chi2_contingency(perm_tab, correction=False)[0]
        if stat >= obs_stat - 1e-12:
            ge += 1
    return (ge + 1) / (n_resamples + 1)


def categorical_test(
    table: Sequence[Sequence[int]],
    mode: str = "auto",
    min_expected: int = 5,
    n_resamples: int = 10000,
    seed: int = 2025,
) -> StatResult:
    """Compare a categorical variable across groups.

    Parameters
    ----------
    table : 2-D counts, rows = groups, columns = categories.
    """
    arr = np.asarray(table, dtype=float)
    # Drop empty rows / columns so the test stays well defined.
    arr = arr[arr.sum(axis=1) > 0]
    if arr.size:
        arr = arr[:, arr.sum(axis=0) > 0]
    n_used = int(arr.sum()) if arr.size else 0

    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2 or n_used == 0:
        return StatResult(np.nan, "not tested (insufficient data)", n_used)

    try:
        chi2, p_chi2, _, expected = stats.chi2_contingency(arr, correction=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("chi-square failed: %s", exc)
        return StatResult(np.nan, "not tested (error)", n_used)

    low_expected = expected.min() < min_expected

    if mode == "chi2":
        return StatResult(float(p_chi2), "Pearson chi-square", n_used)

    if mode == "auto" and not low_expected:
        return StatResult(float(p_chi2), "Pearson chi-square", n_used)

    # Exact / Monte-Carlo branch (auto with low counts, or forced).
    if arr.shape == (2, 2):
        p = stats.fisher_exact(arr.astype(int))[1]
        return StatResult(float(p), "Fisher exact", n_used)

    p = _monte_carlo_chi2(arr, n_resamples=n_resamples, seed=seed)
    return StatResult(
        float(p), f"Chi-square (Monte-Carlo, B={n_resamples})", n_used
    )
