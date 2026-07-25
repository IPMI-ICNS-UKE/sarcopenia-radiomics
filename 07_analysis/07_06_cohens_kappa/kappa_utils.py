import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_kappa_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Cohen's Kappa for all column pairs in *df*.

    Parameters
    ----------
    df : DataFrame
        Rows = patients, columns = binary classifier labels.
        Any row with NaN in *either* compared column is excluded for that pair.

    Returns
    -------
    kappa_matrix : DataFrame
        Square symmetric DataFrame indexed and columned by df.columns.
        Diagonal is 1.0 (perfect self-agreement).
        Off-diagonal values are Cohen's Kappa ∈ [-1, 1].
        NaN when too few overlapping patients exist.
    """
    cols = df.columns.tolist()
    n = len(cols)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        matrix[i, i] = 1.0  # self-agreement
        for j in range(i + 1, n):
            pair = df[[cols[i], cols[j]]].dropna()
            if len(pair) < 2:
                continue
            y1 = pair.iloc[:, 0].astype(int)
            y2 = pair.iloc[:, 1].astype(int)
            # cohen_kappa_score requires at least two classes to be present
            # in the union; fall back to NaN if only one class is observed.
            try:
                kappa = cohen_kappa_score(y1, y2)
            except ValueError:
                kappa = np.nan
            matrix[i, j] = kappa
            matrix[j, i] = kappa

    return pd.DataFrame(matrix, index=cols, columns=cols)
