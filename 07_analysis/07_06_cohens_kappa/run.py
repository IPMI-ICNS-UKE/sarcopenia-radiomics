import os
import numpy as np
import pandas as pd

import config
from data_loader import build_split_dataframes
from kappa_utils import compute_kappa_matrix
from plot_utils import plot_kappa_heatmap


# ── Titles for each split ──────────────────────────────────────────────

SPLIT_TITLES = {
    "train": "Training set (Cohort 1)",
    "test_1": "Temporal test set (Cohort 1)",
    "test_2": "External test set (Cohort 2)",
}


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("07_04 – Cohen's Kappa Agreement Heatmaps")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading predictions and ground-truth labels …")
    splits = build_split_dataframes()

    for split_name, df in splits.items():
        n_patients = df.shape[0]
        n_missing = df.isna().any(axis=1).sum()
        print(
            f"  {split_name:8s}: {n_patients} patients "
            f"({n_missing} with at least one missing classifier)"
        )

    # ── 2. Compute kappa matrices ─────────────────────────────────────────────
    print("\n[2/3] Computing pairwise Cohen's Kappa …")
    kappa_matrices: dict[str, pd.DataFrame] = {}

    for split_name, df in splits.items():
        kappa_df = compute_kappa_matrix(df)
        kappa_matrices[split_name] = kappa_df

        # Print a compact summary
        lower_tri = kappa_df.values[~np.triu(np.ones(kappa_df.shape, dtype=bool))]
        valid = lower_tri[~np.isnan(lower_tri)]
        if len(valid):
            print(
                f"  {split_name:8s}: κ range [{valid.min():.2f}, {valid.max():.2f}], "
                f"mean κ = {valid.mean():.2f}"
            )
        else:
            print(f"  {split_name:8s}: no valid kappa pairs found")

    # ── 3. Plot and save heatmaps ─────────────────────────────────────────────
    print("\n[3/3] Generating heatmaps …")

    for split_name, kappa_df in kappa_matrices.items():
        filename = config.OUTPUT_FILES[split_name]
        output_path = os.path.join(config.OUTPUT_DIR, filename)
        title = SPLIT_TITLES.get(split_name, split_name)
        plot_kappa_heatmap(kappa_df, title=title, output_path=output_path)

    print(f"\nDone. All figures saved to:\n  {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
