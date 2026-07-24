import sys
from pathlib import Path
from typing import Dict
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from cv_base import run_cv_for_specs
from config import RunConfig, RCFG
from models import get_model_specs


def run_cv_for_task(
    df: pd.DataFrame,
    task: str,
    run_cfg: RunConfig = RCFG,
) -> Dict:
    """Run full CV pipeline for a conventional-CT task."""
    specs = get_model_specs(task=task, run_cfg=run_cfg)
    n_rep = run_cfg.debug.n_repeats_outer_override if run_cfg.debug.enabled else run_cfg.cv.n_repeats_outer

    return run_cv_for_specs(
        df=df,
        specs=specs,
        train_split_value="train",
        test_split_value="test",
        n_splits_outer=run_cfg.cv.n_splits_outer,
        n_repeats_outer=n_rep,
        n_splits_inner=run_cfg.cv.n_splits_inner,
        random_state=run_cfg.cv.random_state,
        l1_ratios=run_cfg.enet.l1_ratios,
        logistic_cs=run_cfg.enet.logistic_cs,
        linear_alphas=run_cfg.enet.linear_alphas,
        max_iter=run_cfg.enet.max_iter,
        convergence_tol=run_cfg.enet.convergence_tol,
        default_cls_threshold=run_cfg.cv.classification_threshold,
    )
