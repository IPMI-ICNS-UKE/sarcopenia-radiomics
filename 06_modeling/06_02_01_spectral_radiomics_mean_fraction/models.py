import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from models_base import ModelSpec
from config import RunConfig, RCFG, TaskName


def get_model_specs(task: TaskName, run_cfg: RunConfig = RCFG) -> List[ModelSpec]:
    """Return all ModelSpec objects for the mean-fraction pipeline."""
    if task == "sarcopenia_composite_cls":
        task_kind = "classification"
    elif task == "hand_grip_reg":
        task_kind = "regression"
    elif task == "chair_rise_cls":
        task_kind = "classification"
    else:
        raise KeyError(f"Unknown task: {task}")

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
        col = f"mean_fraction_{map_name}"
        specs.append(ModelSpec(
            name=f"{map_name}_mean_fraction{postfix}",
            kind="direct", task_kind=task_kind,
            base_features=(col,), clinical_features=c,
        ))
        specs.append(ModelSpec(
            name=f"{map_name}_mean_fraction_score_clinical{postfix}",
            kind="score_plus_clinical", task_kind=task_kind,
            base_features=(col,), clinical_features=c,
        ))

    # All maps combined: direct and score+clinical
    all_cols = tuple(f"mean_fraction_{m}" for m in run_cfg.blocks.map_names)
    specs.append(ModelSpec(
        name=f"all_maps_mean_fraction{postfix}",
        kind="direct", task_kind=task_kind,
        base_features=all_cols, clinical_features=c,
    ))
    specs.append(ModelSpec(
        name=f"all_maps_mean_fraction_score_clinical{postfix}",
        kind="score_plus_clinical", task_kind=task_kind,
        base_features=all_cols, clinical_features=c,
    ))

    return specs
