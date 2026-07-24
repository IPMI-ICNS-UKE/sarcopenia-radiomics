from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


@dataclass
class RocData:
    """Container for one model's ROC curve data and associated metrics."""

    model_name: str
    display_name: str
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    auc_lo: Optional[float] = None
    auc_hi: Optional[float] = None

    @property
    def auc_label(self) -> str:
        """
        Returns AUC formatted for legend:
        'AUC = 0.XX (0.YY–0.ZZ)' when CI is available,
        'AUC = 0.XX' otherwise.
        Two decimal places as required.
        """
        auc_str = f"{self.auc:.2f}"
        if self.auc_lo is not None and self.auc_hi is not None:
            return f"AUC = {auc_str} ({self.auc_lo:.2f}–{self.auc_hi:.2f})"
        return f"AUC = {auc_str}"


def compute_roc_curves(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_order: List[str],
    model_display_names: Dict[str, str],
) -> List[RocData]:
    """
    Compute ROC curves for each model in model_order.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: model_name, target, pred_score.
    metrics_df : pd.DataFrame
        Must contain columns: model_name, AUC, AUC_lo, AUC_hi.
    model_order : list of str
        Defines the order in which models appear in the output.
    model_display_names : dict
        Mapping from model_name to human-readable display name.

    Returns
    -------
    List[RocData], one entry per model found in predictions_df (in model_order).
    """
    roc_list: List[RocData] = []

    metrics_indexed = (
        metrics_df.set_index("model_name") if not metrics_df.empty else pd.DataFrame()
    )

    for model_name in model_order:
        sub = predictions_df[predictions_df["model_name"] == model_name]
        if sub.empty:
            continue

        y_true = sub["target"].astype(int).values
        y_score = sub["pred_score"].astype(float).values

        # Guard: need at least both classes present
        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)

        # Pull AUC and CI from pre-computed metrics if available; otherwise
        # fall back to sklearn (no CI).
        if (
            not metrics_indexed.empty
            and model_name in metrics_indexed.index
        ):
            row = metrics_indexed.loc[model_name]
            auc = float(row["AUC"])
            auc_lo = row.get("AUC_lo", None)
            auc_hi = row.get("AUC_hi", None)
            auc_lo = float(auc_lo) if auc_lo is not None and not pd.isna(auc_lo) else None
            auc_hi = float(auc_hi) if auc_hi is not None and not pd.isna(auc_hi) else None
        else:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_score))
            auc_lo = None
            auc_hi = None

        display_name = model_display_names.get(model_name, model_name)

        roc_list.append(
            RocData(
                model_name=model_name,
                display_name=display_name,
                fpr=fpr,
                tpr=tpr,
                auc=auc,
                auc_lo=auc_lo,
                auc_hi=auc_hi,
            )
        )

    return roc_list
