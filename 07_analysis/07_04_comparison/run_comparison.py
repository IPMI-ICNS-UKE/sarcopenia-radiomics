import os
import sys
import traceback

import config_extended as config
from data_loader import load_all_data
from statistical_tests import compute_pvalues_block
from table_builder import build_cls_block, build_reg_block, save_comparison_tables
from heatmap_plotter import plot_all_heatmaps
from interpretation_report import generate_interpretation_report


def main() -> None:
    print("\n" + "=" * 70)
    print("SarcoSpect — Stage 07_02: Statistical Comparison")
    print("=" * 70)
    print(f"  Output directory : {config.OUTPUT_DIR}")
    print(f"  Correction method: {config.CORRECTION_METHOD}")
    print(f"  Alpha            : {config.ALPHA}")
    print()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    print("[1/4] Loading metrics and predictions ...")
    data = load_all_data()
    print()

    # Step 2: Compute p-values
    print("[2/4] Computing p-values ...")
    all_pvalues = {}  # all_pvalues[task][subset_key] -> pvalue dict

    for task in ["cls", "reg", "cls_2"]:
        all_pvalues[task] = {}
        for subset_key in ["train", "test_1", "test_2"]:
            print(f"  [{task}][{subset_key}] ...")
            predictions = data["predictions"][task][subset_key]
            pvalues = compute_pvalues_block(task, subset_key, predictions)
            all_pvalues[task][subset_key] = pvalues
    print()

    # Step 3: Build and save comparison tables
    print("[3/4] Building comparison tables ...")
    cls_blocks = {}
    reg_blocks = {}
    cls_2_blocks = {}

    for subset_key in ["train", "test_1", "test_2"]:
        cls_blocks[subset_key] = build_cls_block(
            subset_key=subset_key,
            metrics=data["metrics"]["cls"][subset_key],
            predictions=data["predictions"]["cls"][subset_key],
            pvalues=all_pvalues["cls"][subset_key],
        )
        reg_blocks[subset_key] = build_reg_block(
            subset_key=subset_key,
            metrics=data["metrics"]["reg"][subset_key],
            predictions=data["predictions"]["reg"][subset_key],
            pvalues=all_pvalues["reg"][subset_key],
        )
        cls_2_blocks[subset_key] = build_cls_block(
            subset_key=subset_key,
            metrics=data["metrics"]["cls_2"][subset_key],
            predictions=data["predictions"]["cls_2"][subset_key],
            pvalues=all_pvalues["cls_2"][subset_key],
            data_subset="cls_2",
        )

    save_comparison_tables(cls_blocks, reg_blocks, cls_2_blocks, config.OUTPUT_DIR)
    print()

    # Step 4: Generate heatmaps and interpretation report
    print("[4/4] Generating heatmaps and interpretation report ...")
    plot_all_heatmaps(all_pvalues, config.OUTPUT_DIR, use_corrected=True)
    generate_interpretation_report(all_pvalues, config.OUTPUT_DIR, use_corrected=True)
    print()

    print("=" * 70)
    print("Done. All outputs saved to:")
    print(f"  {config.OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] An unexpected error occurred:")
        traceback.print_exc()
        sys.exit(1)
