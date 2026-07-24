import os
import pandas as pd
import config
from data_loader import get_patient_ct_and_masks_by_level
from overlay_utils import plot_segmentation_overlay


def select_representative_patients() -> dict[str, str]:
    """Return {"worst": patient_id, "median": patient_id, "best": patient_id}."""
    per_patient_path = os.path.join(config.OUTPUT_DIR, config.PER_PATIENT_CSV)
    if not os.path.exists(per_patient_path):
        raise FileNotFoundError(
            f"{per_patient_path} not found — run run.py first to compute per-patient Dice."
        )

    df = pd.read_csv(per_patient_path)
    overall = df.groupby("patient_id")["dice"].mean()
    median_dice = overall.median()

    return {
        "worst": str(overall.idxmin()),
        "median": str(overall.sub(median_dice).abs().idxmin()),
        "best": str(overall.idxmax()),
    }


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    selected = select_representative_patients()
    for tag, patient_id in selected.items():
        print(f"{tag:8s} → {patient_id}")

    for tag, patient_id in selected.items():
        by_level = get_patient_ct_and_masks_by_level(patient_id)
        for level in config.LEVELS:
            ct_2d, masks_by_rater = by_level[level]
            output_path = os.path.join(
                config.OUTPUT_DIR, f"segmentation_overlay_{tag}_{patient_id}_{level}.png"
            )
            plot_segmentation_overlay(patient_id, level, ct_2d, masks_by_rater, output_path=output_path)


if __name__ == "__main__":
    main()
