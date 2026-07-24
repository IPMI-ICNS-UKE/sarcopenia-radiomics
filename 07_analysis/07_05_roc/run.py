import os

from config import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    OUTPUT_FILENAMES,
    SPLITS,
    STAGE_OUTPUT_DIR,
)
from data_loader import load_all_splits
from roc_utils import compute_roc_curves
from plot_roc_static import plot_roc_static
from plot_roc_interactive import plot_roc_interactive


def main() -> None:
    os.makedirs(STAGE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {STAGE_OUTPUT_DIR}\n")

    print("Loading data for all splits …")
    all_data = load_all_splits()

    for split_key, (metrics_df, predictions_df) in all_data.items():
        _, _, split_title = SPLITS[split_key]
        base_name = OUTPUT_FILENAMES[split_key]

        print(f"\n── {split_title} ──")

        if predictions_df.empty:
            print(f"  No prediction data found for split '{split_key}'. Skipping.")
            continue

        # Compute ROC curves
        roc_list = compute_roc_curves(
            predictions_df=predictions_df,
            metrics_df=metrics_df,
            model_order=MODEL_ORDER,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

        if not roc_list:
            print(f"  No ROC data could be computed for split '{split_key}'. Skipping.")
            continue

        print(f"  Models with ROC data: {[r.model_name for r in roc_list]}")

        # ── Static PNG ───────────────────────────────────────────────────────
        png_path = os.path.join(STAGE_OUTPUT_DIR, f"{base_name}.png")
        plot_roc_static(
            roc_data_list=roc_list,
            title=split_title,
            output_path=png_path,
        )

        # ── Interactive HTML ─────────────────────────────────────────────────
        html_path = os.path.join(STAGE_OUTPUT_DIR, f"{base_name}.html")
        plot_roc_interactive(
            roc_data_list=roc_list,
            title=split_title,
            output_path=html_path,
        )

    print("\nDone. All ROC figures saved.")


if __name__ == "__main__":
    main()
