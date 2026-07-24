import os
import config
from data_loader import get_manual_annotation_patients, load_all_patient_masks
from dice_utils import aggregate_per_patient, compute_per_level_dice, summarize_across_patients
from plot_utils import plot_agreement_boxplot


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("07_02 – Segmentation Agreement (Dice)")
    print("=" * 60)

    print("\n[1/4] Selecting manual-annotation subset (cohort 1) …")
    patient_ids = get_manual_annotation_patients()
    print(f"  {len(patient_ids)} patients: {', '.join(patient_ids)}")

    print("\n[2/4] Extracting L2/L3/L4 masks (Reader 1, Reader 2, Automated) …")
    patient_masks = load_all_patient_masks()

    print("\n[3/4] Computing pairwise Dice similarity coefficients …")
    per_level_df = compute_per_level_dice(patient_masks)
    per_patient_df = aggregate_per_patient(per_level_df)
    summary_df = summarize_across_patients(per_patient_df)

    per_level_df.to_csv(os.path.join(config.OUTPUT_DIR, config.PER_LEVEL_CSV), index=False)
    per_patient_df.to_csv(os.path.join(config.OUTPUT_DIR, config.PER_PATIENT_CSV), index=False)
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, config.SUMMARY_CSV), index=False)

    print(summary_df.to_string(index=False))

    print("\n[4/4] Generating figure …")
    plot_agreement_boxplot(
        per_patient_df,
        output_path=os.path.join(config.OUTPUT_DIR, config.FIGURE_FILE),
    )

    print(f"\nDone. All outputs saved to:\n  {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
