import os
from tqdm import tqdm
import config
from data_loader import get_manual_annotation_cohort, get_patient_l3_data
from feature_utils import build_raw_values_df, compute_raw_values
from icc_utils import build_icc_summary


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("07_03 – Feature Agreement (2D SMI / MRA / Muscle fat fraction)")
    print("=" * 60)

    print("\n[1/3] Selecting manual-annotation subset (cohort 1) …")
    cohort_df = get_manual_annotation_cohort()
    print(f"  {len(cohort_df)} patients: {', '.join(cohort_df['Pat ID'])}")

    print("\n[2/3] Computing L3 biomarkers per rater (Reader 1, Reader 2, Automated) …")
    all_rows = []
    for _, row in tqdm(cohort_df.iterrows(), total=len(cohort_df)):
        patient_id = row["Pat ID"]
        patient_data = get_patient_l3_data(patient_id, height_m=float(row["height"]))
        all_rows.extend(compute_raw_values(patient_id, patient_data))

    raw_values_df = build_raw_values_df(all_rows)
    raw_values_df.to_csv(os.path.join(config.OUTPUT_DIR, config.RAW_VALUES_CSV), index=False)

    print("\n[3/3] Computing ICC(2,1) with 95% CI …")
    icc_summary_df = build_icc_summary(raw_values_df)
    icc_summary_df.to_csv(os.path.join(config.OUTPUT_DIR, config.ICC_SUMMARY_CSV), index=False)

    print(icc_summary_df.to_string(index=False))
    print(f"\nDone. All outputs saved to:\n  {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
