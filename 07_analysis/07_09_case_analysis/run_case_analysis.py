import argparse
import traceback
from typing import Any, Dict, Optional

import pandas as pd

from config import CASE_ORDER, CASES, RCFG, RunConfig
from imaging import PatientImageContext, load_patient_images
from scoring import extract_patient_scores
from figure_export import build_single_case_figure, save_case_figure
from table_export import build_case_table, save_excel_table


# Ground-truth loading helpers
_GT_COLUMNS = [
    "Pat ID",
    "sex",
    "age",
    "bmi",
    "height",
    "hand_grip_cont",
    "chair_rise_cont",
    "gait_speed_cont",
    "test_temporal",
]


def _load_gt(cohort_label: str, run_cfg: RunConfig) -> pd.DataFrame:
    if cohort_label == "cohort1":
        path = run_cfg.paths.gt_table_cohort1
    else:
        path = run_cfg.paths.gt_table_cohort2
    df = pd.read_excel(path)
    df["Pat ID"] = df["Pat ID"].astype(str)
    return df


def _get_gt_row(
    patient_id: str,
    cohort_label: str,
    gt_cache: Dict[str, pd.DataFrame],
    run_cfg: RunConfig,
) -> pd.Series:
    if cohort_label not in gt_cache:
        gt_cache[cohort_label] = _load_gt(cohort_label, run_cfg)
    df = gt_cache[cohort_label]
    rows = df[df["Pat ID"] == patient_id]
    if rows.empty:
        raise ValueError(f"Patient {patient_id} not found in GT table for {cohort_label}.")
    return rows.iloc[0]


# Per-patient processing
def _process_patient(
    patient_id: str,
    cohort_label: str,
    case_type: str,
    gt_row: pd.Series,
    run_cfg: RunConfig,
) -> Dict[str, Any]:
    print(f"    Loading images for {patient_id} ({cohort_label}) …")
    ctx = load_patient_images(
        patient_id=patient_id,
        cohort_label=cohort_label,
        paths_cfg=run_cfg.paths,
        img_cfg=run_cfg.imaging,
    )

    print(f"    Running model for {patient_id} …")
    scores = extract_patient_scores(
        patient_id=patient_id,
        cohort_label=cohort_label,
        gt_row=gt_row,
        run_cfg=run_cfg,
    )

    record: Dict[str, Any] = {
        "patient_id": patient_id,
        "case_type": case_type,
        # demographics
        "sex": gt_row.get("sex"),
        "age": gt_row.get("age"),
        "bmi": gt_row.get("bmi"),
        "height": gt_row.get("height"),
        # functional tests
        "hand_grip_cont": gt_row.get("hand_grip_cont"),
        "chair_rise_cont": gt_row.get("chair_rise_cont"),
        "gait_speed_cont": gt_row.get("gait_speed_cont"),
        # model output
        "musclefat_score": scores["musclefat_score"],
        "ct_score": scores["ct_score"],
        "combined_score": scores["combined_score"],
        "pred_proba": scores["pred_proba"],
        "model_threshold": scores["model_threshold"],
        # image context (not written to Excel, used for figure)
        "_ctx": ctx,
    }
    return record


# Set-level runner
def run_set(
    set_name: str,
    run_cfg: RunConfig = RCFG,
) -> None:
    """
    Process all four cases (TP/TN/FP/FN) for one set and save outputs.
    """
    print(f"\n{'='*70}")
    print(f"  Processing set: {set_name}")
    print(f"{'='*70}")

    set_cases = CASES[set_name]
    gt_cache: Dict[str, pd.DataFrame] = {}
    records: Dict[str, Dict[str, Any]] = {}  # case_type → record
    contexts: Dict[str, Optional[PatientImageContext]] = {}

    for case_type in CASE_ORDER:
        entry = set_cases.get(case_type)
        if entry is None:
            print(f"  [SKIP] No {case_type} case defined for set '{set_name}'.")
            contexts[case_type] = None
            continue

        patient_id, cohort_label = entry
        print(f"\n  [{case_type}] {patient_id}  ({cohort_label})")

        try:
            gt_row = _get_gt_row(patient_id, cohort_label, gt_cache, run_cfg)
            rec = _process_patient(
                patient_id=patient_id,
                cohort_label=cohort_label,
                case_type=case_type,
                gt_row=gt_row,
                run_cfg=run_cfg,
            )
            records[case_type] = rec
            contexts[case_type] = rec["_ctx"]

        except Exception:
            print(f"  [ERROR] Failed to process {patient_id} ({case_type}):")
            traceback.print_exc()
            contexts[case_type] = None
            records[case_type] = {}

    # ---- Figures (one per case type: TP / TN / FP / FN) --------------------
    for case_type in CASE_ORDER:
        print(f"\n  Building figure for {set_name} / {case_type} …")
        try:
            fig = build_single_case_figure(
                ctx=contexts.get(case_type),
                case_type=case_type,
                run_cfg=run_cfg,
            )
            fig_path = run_cfg.paths.figures_dir / f"case_{set_name}_{case_type}.png"
            save_case_figure(fig, fig_path, dpi=run_cfg.figure.dpi)
        except Exception:
            print(f"  [ERROR] Figure generation failed for {set_name} / {case_type}:")
            traceback.print_exc()

    # ---- Excel table -------------------------------------------------------
    print(f"  Building table for set '{set_name}' …")
    try:
        df_table = build_case_table(records)
        tbl_path = run_cfg.paths.tables_dir / f"cases_{set_name}.xlsx"
        save_excel_table(df_table, tbl_path, table_cfg=run_cfg.table)
    except Exception:
        print(f"  [ERROR] Table generation failed for set '{set_name}':")
        traceback.print_exc()


# Entry point
_ALL_SETS = list(CASES.keys())  # ["train", "test_1", "test_2"]


def main(sets=None, run_cfg: RunConfig = RCFG) -> None:
    run_cfg.paths.make_all()
    target_sets = sets or _ALL_SETS
    for set_name in target_sets:
        if set_name not in CASES:
            print(f"  [WARNING] Unknown set '{set_name}', skipping.")
            continue
        run_set(set_name, run_cfg=run_cfg)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate case-analysis figures and tables for 07_09."
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=_ALL_SETS,
        default=None,
        help="Which sets to process (default: all).",
    )
    args = parser.parse_args()
    main(sets=args.sets)
