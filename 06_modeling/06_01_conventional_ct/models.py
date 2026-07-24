import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from models_base import ModelSpec  # re-exported for convenience
from config import RunConfig, RCFG, TaskName


def get_model_specs(task: TaskName, run_cfg: RunConfig = RCFG) -> List[ModelSpec]:
    """Return ModelSpec list for the conventional-CT pipeline."""
    if task == "sarcopenia_composite_cls":
        task_kind = "classification"
    elif task == "hand_grip_reg":
        task_kind = "regression"
    elif task == "chair_rise_cls":
        task_kind = "classification"
    else:
        raise KeyError(f"Unknown task: {task}")

    c = run_cfg.blocks.clinical_features
    auto_smi = run_cfg.blocks.auto_smi_block
    auto_mra = run_cfg.blocks.auto_mra_block
    auto_smi_mra = run_cfg.blocks.auto_smi_mra_block

    if run_cfg.paths.use_3d:
        # 3D models: use 3D mean fraction features
        postfix = "_3d"
    else:
        # 2D models: use 2D mean fraction features
        postfix = ""

    return [
        ModelSpec(f"auto_smi{postfix}",                    "direct",             task_kind, auto_smi,     c),
        ModelSpec(f"auto_mra{postfix}",                    "direct",             task_kind, auto_mra,     c),
        ModelSpec(f"auto_smi_mra{postfix}",                "direct",             task_kind, auto_smi_mra, c),
        ModelSpec(f"auto_smi_score_clinical{postfix}",     "score_plus_clinical",task_kind, auto_smi,     c),
        ModelSpec(f"auto_mra_score_clinical{postfix}",     "score_plus_clinical",task_kind, auto_mra,     c),
        ModelSpec(f"auto_smi_mra_score_clinical{postfix}", "score_plus_clinical",task_kind, auto_smi_mra, c),
    ]
