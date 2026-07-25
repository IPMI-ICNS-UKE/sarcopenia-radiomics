from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_cv_tuning_plot(cv_results: pd.DataFrame, out_path: Path) -> None:
    if cv_results.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x = cv_results["C"].astype(float).to_numpy()
    y = cv_results["mean_test_score"].astype(float).to_numpy()
    yerr = cv_results["std_test_score"].astype(float).to_numpy()
    ax.semilogx(x, y, marker="o")
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Mean CV score")
    ax.set_title("L1 logistic regression tuning")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_selected_coefficients_plot(coef_df: pd.DataFrame, out_path: Path, top_n: int = 25) -> None:
    if coef_df.empty:
        return
    plot_df = coef_df.copy()
    plot_df["abs_coef"] = plot_df["coefficient"].abs()
    plot_df = plot_df.sort_values("abs_coef", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["coefficient"])
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title("Selected non-zero coefficients")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_selection_frequency_plot(freq_df: pd.DataFrame, out_path: Path, top_n: int = 25) -> None:
    if freq_df.empty:
        return
    plot_df = freq_df.sort_values(
        ["selection_frequency", "abs_mean_coefficient"], ascending=[False, False]
    ).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["selection_frequency"])
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Feature")
    ax.set_title("Selection frequency across resampling refits")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_coefficient_path_plot(
    path_df: pd.DataFrame,
    out_path: Path,
    best_c: Optional[float] = None,
    n_selected_features: Optional[int] = None,
    max_lines: int = 30,
) -> None:
    """L1 coefficient path plot on a log10(C) axis.

    Parameters
    ----------
    path_df : pd.DataFrame
        Long-format frame with columns ['C', 'feature', 'coefficient'] produced
        by `compute_coefficient_paths`.
    out_path : Path
        Destination PNG path. Saved at 300 DPI.
    best_c : float, optional
        Selected C value (e.g. from GridSearchCV). If provided, a vertical
        reference line is drawn at log10(best_c) and the legend reports the
        selected C and the number of features with non-zero coefficients.
    n_selected_features : int, optional
        Number of features with non-zero coefficients at `best_c`. Shown in
        the legend together with `best_c`.
    max_lines : int
        Maximum number of feature trajectories to draw (top by max |coef|).
    """
    if path_df.empty:
        return

    pivot = path_df.pivot(index="C", columns="feature", values="coefficient").sort_index()
    keep_cols = pivot.abs().max(axis=0).sort_values(ascending=False).head(max_lines).index.tolist()

    c_values = pivot.index.to_numpy(dtype=float)
    if np.any(c_values <= 0):
        raise ValueError(
            "Coefficient-path plot requires strictly positive C values for log scaling."
        )
    log_c = np.log10(c_values)

    with sns.axes_style("whitegrid"), sns.plotting_context("paper", font_scale=1.15):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        palette = sns.color_palette("husl", n_colors=max(len(keep_cols), 1))
        for col, color in zip(keep_cols, palette):
            ax.plot(
                log_c,
                pivot[col].to_numpy(dtype=float),
                linewidth=1.5,
                color=color,
                alpha=0.85,
            )

        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="-", alpha=0.6)

        if best_c is not None:
            best_c = float(best_c)
            if best_c <= 0:
                raise ValueError(f"best_c must be > 0, got {best_c!r}.")
            log_best_c = float(np.log10(best_c))
            label_lines = [f"selected C = {best_c:g}"]
            if n_selected_features is not None:
                label_lines.append(f"non-zero features = {int(n_selected_features)}")
            ax.axvline(
                log_best_c,
                color="#d62728",
                linewidth=1.8,
                linestyle="--",
                label="\n".join(label_lines),
            )
            ax.legend(
                loc="best",
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor="0.5",
                fontsize=10,
            )

        ax.set_xlabel(r"$\log_{10}(C)$")
        ax.set_ylabel("Coefficient")
        ax.set_title("L1 logistic regression — coefficient paths")
        sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="png")
        plt.close(fig)
