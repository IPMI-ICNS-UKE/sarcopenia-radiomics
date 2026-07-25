import sys
from pathlib import Path

# Allow running from repo root  →  python 07_08_probability_distribution/run.py
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from data_loader import load_ground_truth_sex, load_predictions
from plot_static import plot_probability_distribution_static
from plot_interactive import plot_probability_distribution_interactive


# ── Split definitions ─────────────────────────────────────────────────────────

SPLITS = [
    {
        "name": "train",
        "csv_path": cfg.TRAIN_CSV,
        "gt_path": cfg.GT_COHORT1,
        "label": "Training set (OOF)",  # filled with n after loading
        "png_stem": "probability_distribution_train",
        "html_stem": "probability_distribution_train",
    },
    {
        "name": "test_1",
        "csv_path": cfg.TEST1_CSV,
        "gt_path": cfg.GT_COHORT1,
        "label": "Test set 1 (temporal hold-out)",
        "png_stem": "probability_distribution_test_1",
        "html_stem": "probability_distribution_test_1",
    },
    {
        "name": "test_2",
        "csv_path": cfg.TEST2_CSV,
        "gt_path": cfg.GT_COHORT2,
        "label": "Test set 2 (cohort 2)",
        "png_stem": "probability_distribution_test_2",
        "html_stem": "probability_distribution_test_2",
    },
]


def main() -> None:
    out_dir = cfg.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: load every split ──────────────────────────────────────────────
    loaded = {}
    for split in SPLITS:
        gt_path = split["gt_path"]
        if not gt_path.exists():
            print(f"  Ground-truth table not found: {gt_path}. Skipping {split['name']}.")
            continue

        gt_sex = load_ground_truth_sex(gt_path)
        df = load_predictions(
            csv_path=split["csv_path"],
            model_name=cfg.MODEL_NAME,
            gt_sex=gt_sex,
        )
        if df is None:
            # CSV file does not exist yet (e.g. cohort-2 not ready)
            continue
        if df.empty:
            print(f"  No data for model '{cfg.MODEL_NAME}' in {split['csv_path']}")
            continue
        loaded[split["name"]] = df

    # ── One Youden threshold, shared across train / test 1 / test 2 ──────────
    threshold = None
    for name in ("test_1", "test_2", "train"):
        df = loaded.get(name)
        if (
            df is not None
            and "selected_threshold" in df.columns
            and df["selected_threshold"].notna().any()
        ):
            threshold = float(df["selected_threshold"].dropna().iloc[0])
            print(f"\nUsing Youden threshold {threshold:.3f} (from '{name}') for all splits.")
            break

    # Recompute pred_label from the shared threshold so the misclassification
    # (FP/FN) highlighting is consistent with the single cut-point drawn on
    # every panel, instead of each split's own (per-fold, for training) label.
    if threshold is not None:
        for df in loaded.values():
            df["pred_label"] = (df["pred_proba"] >= threshold).astype(int)

    # ── Pass 2: plot each split with the shared threshold ─────────────────────
    for split in SPLITS:
        df = loaded.get(split["name"])
        if df is None:
            continue

        n = len(df)
        split_label = f"{split['label']}  (n={n})"
        print(f"\n── Processing split: {split['name']} ── ({n} cases)")

        # Static PNG
        png_path = out_dir / f"{split['png_stem']}.png"
        plot_probability_distribution_static(
            df=df,
            out_path=png_path,
            split_label=split_label,
            display_name=cfg.DISPLAY_NAME,
            cfg=cfg,
            threshold=threshold,
        )

        # Interactive HTML
        html_path = out_dir / f"{split['html_stem']}.html"
        plot_probability_distribution_interactive(
            df=df,
            out_path=html_path,
            split_label=split_label,
            display_name=cfg.DISPLAY_NAME,
            cfg=cfg,
            threshold=threshold,
        )

    print("\n✓ Done.  Outputs in:", cfg.OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
