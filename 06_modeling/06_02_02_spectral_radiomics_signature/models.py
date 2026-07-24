import sys
from pathlib import Path
from typing import List
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from models_base import ModelSpec, _GROUP_SENTINEL
from dataset import feature_columns_for_map
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
    """Build {group_key: (col1, col2, ...)} from the dataset columns."""
    lookup = {}
    for map_name in run_cfg.blocks.map_names:
        cols = feature_columns_for_map(df, map_name)
        if cols:
            lookup[f"{map_name}_signature"] = tuple(cols)
    return lookup


def get_pass1_specs(task: TaskName, run_cfg: RunConfig = RCFG) -> List[ModelSpec]:
    """Per-map and all-maps-combined specs (resolved by resolve_model_specs)."""
    tk = _task_kind(task)
    c = run_cfg.blocks.clinical_features
    specs: List[ModelSpec] = []

    if run_cfg.paths.use_3d:
        # 3D models: use 3D mean fraction features
        postfix = "_3d"
    else:
        # 2D models: use 2D mean fraction features
        postfix = ""

    for map_name in run_cfg.blocks.map_names:
        key = f"{map_name}_signature"
        sentinel = (_GROUP_SENTINEL + key,)
        specs.append(ModelSpec(key,                    "direct",             tk, sentinel, c))
        specs.append(ModelSpec(f"{key}_score_clinical{postfix}","score_plus_clinical",tk, sentinel, c))

    return specs


def get_pass2_specs(
    task: TaskName,
    run_cfg: RunConfig = RCFG,
) -> List[ModelSpec]:
    """Score-aggregation specs across maps.

    Fixed map-pair models:
      - mf_vnc_scores / mf_vnc_scores_clinical (MuscleFat + VNC)
      - mf_ct_scores / mf_ct_scores_clinical (MuscleFat + conventional CT)

    Group names must match keys produced by build_group_lookup(),
    i.e. "<map_name>_signature".
    """
    tk = _task_kind(task)
    c = run_cfg.blocks.clinical_features

    if run_cfg.paths.use_3d:
        # 3D models: use 3D mean fraction features
        postfix = "_3d"
    else:
        # 2D models: use 2D mean fraction features
        postfix = ""

    # MuscleFat and VNC groups (if all four are present in the dataset)
    mf_vnc_groups = tuple((f"{m}_signature{postfix}", tuple()) for m in ("musclefat", "vnc", ))

    # MuscleFat and conventional CT maps
    mf_ct_groups = tuple((f"{m}_signature{postfix}", tuple()) for m in ("musclefat", "ct"))

    specs: List[ModelSpec] = [
        # Fixed map-pair scores
        ModelSpec(f"mf_vnc_scores{postfix}",
                  "multi_score", tk, tuple(), c, True, mf_vnc_groups),
        ModelSpec(f"mf_vnc_scores_clinical{postfix}",
                  "multi_score_plus_clinical", tk, tuple(), c, True, mf_vnc_groups),
        ModelSpec(f"mf_ct_scores{postfix}",
                  "multi_score", tk, tuple(), c, True, mf_ct_groups),
        ModelSpec(f"mf_ct_scores_clinical{postfix}",
                  "multi_score_plus_clinical", tk, tuple(), c, True, mf_ct_groups),
    ]

    return specs
