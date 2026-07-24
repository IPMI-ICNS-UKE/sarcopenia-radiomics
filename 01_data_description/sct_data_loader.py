from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


class DataLoader:
    """
    Loads cohort tables, applies study filters, and performs cohort splitting.

    Current behavior:
    - Cohort 1:
        - train = test_temporal == 0
        - internal test = test_temporal == 1
    - Cohort 2:
        - whole filtered cohort is returned as a single test set

    Approved filters:
    - use == 1
    - image_present == 1
    - labels_present == 1
    """

    def __init__(self, cohort1_path: Path, cohort2_path: Optional[Path] = None):
        self.cohort1_path = Path(cohort1_path)
        self.cohort2_path = Path(cohort2_path) if cohort2_path is not None else None

        self.filters = {
            "use": 1,
            "image_present": 1,
            "labels_present": 1,
        }

    def _read_excel(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        return pd.read_excel(path)

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filtered = df.copy()

        for col, val in self.filters.items():
            if col not in df_filtered.columns:
                raise KeyError(f"Required filter column is missing: {col}")
            df_filtered = df_filtered[df_filtered[col] == val]

        return df_filtered.reset_index(drop=True)

    def load_cohort1_full(self) -> pd.DataFrame:
        df = self._read_excel(self.cohort1_path)
        return self._apply_filters(df)

    def load_cohort1(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        train_df : pd.DataFrame
            Cohort 1 training set (test_temporal == 0)
        test_df : pd.DataFrame
            Cohort 1 internal testing set (test_temporal == 1)
        """
        df = self.load_cohort1_full()

        if "test_temporal" not in df.columns:
            raise KeyError("Required column is missing: test_temporal")

        train_df = df[df["test_temporal"] == 0].reset_index(drop=True)
        test_df = df[df["test_temporal"] == 1].reset_index(drop=True)

        return train_df, test_df

    def load_cohort2(self) -> Optional[pd.DataFrame]:
        """
        Returns the full filtered cohort 2 table, or None if cohort 2 is not provided.
        """
        if self.cohort2_path is None:
            return None

        df = self._read_excel(self.cohort2_path)
        return self._apply_filters(df)
