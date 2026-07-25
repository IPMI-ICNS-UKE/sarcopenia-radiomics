import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from models_base import ModelSpec, _GROUP_SENTINEL
from config import RunConfig, RCFG, TaskName


def _task_kind(task: TaskName) -> str:
    if task == "sarcopenia_composite_cls":
        return "classification"
    elif task == "hand_grip_reg":
        return "regression"
    elif task == "chair_rise_cls":
        return "classification"
    raise KeyError(f"Unknown task: {task}")


def build_group_lookup(df: pd.DataFrame, run_cfg: RunConfig) -> dict:
    """Build {map_name: (col1, col2, ...)} from dataset columns."""
    lookup = {}
    for map_name in run_cfg.blocks.map_names:
        cols = run_cfg.blocks.feature_cols_for_map(df.columns, map_name)
        if cols:
            lookup[map_name] = cols
    return lookup


def get_model_specs(
    task: TaskName,
    run_cfg: RunConfig = RCFG,
    group_lookup: Optional[dict] = None,
) -> List[ModelSpec]:
    """
    Return all ModelSpec objects for the deep radiomics pipeline.
    If group_lookup is None, feature columns are resolved via __GROUP__ sentinels
    and must be resolved later via resolve_model_specs(specs, group_lookup).
    """
    tk = _task_kind(task)
    c = run_cfg.blocks.clinical_features
    specs: List[ModelSpec] = []

    if run_cfg.paths.use_3d:
        # 3D models: use 3D mean fraction features
        postfix = "_3d"
    else:
        # 2D models: use 2D mean fraction features
        postfix = ""

    # Per-map: direct and score+clinical
    for map_name in run_cfg.blocks.map_names:
        if group_lookup is not None:
            feats = group_lookup.get(map_name, tuple())
        else:
            feats = (_GROUP_SENTINEL + map_name,)

        specs.append(
            ModelSpec(
                name=f"{map_name}_deep_score_clinical{postfix}",
                kind="score_plus_clinical",
                task_kind=tk,
                base_features=feats,
                clinical_features=c,
            )
        )

    # CT + MuscleFat two-score model
    ct_feats = (
        group_lookup.get(run_cfg.blocks.ct_map_name, tuple())
        if group_lookup
        else (_GROUP_SENTINEL + run_cfg.blocks.ct_map_name,)
    )
    mf_feats = (
        group_lookup.get(run_cfg.blocks.musclefat_map_name, tuple()) if group_lookup else tuple()
    )
    # mf_features stored in score_groups[0][1] for two_scores_plus_clinical
    specs.append(
        ModelSpec(
            name=f"ct_{run_cfg.blocks.musclefat_map_name}_deep_scores_clinical{postfix}",
            kind="two_scores_plus_clinical",
            task_kind=tk,
            base_features=ct_feats,
            clinical_features=c,
            score_groups=((run_cfg.blocks.musclefat_map_name, mf_feats),),
        )
    )

    return specs
