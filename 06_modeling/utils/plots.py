from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# Internal helpers
def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_mpl(fig: plt.Figure, out_path: Path, dpi: int = 300) -> None:
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _save_plotly(fig, out_path: Path) -> None:
    if fig is None or go is None:
        return
    _ensure_dir(out_path)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved interactive plot: {out_path}")


def _filter_models(df: pd.DataFrame, model_names: Optional[Sequence[str]]) -> pd.DataFrame:
    if model_names is None:
        return df.copy()
    return df[df["model_name"].isin(list(model_names))].copy()


# ROC curves
def plot_roc_curves(
    df_pred: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    df = _filter_models(df_pred, model_names)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name, g in df.groupby("model_name"):
        y_true = g["target"].to_numpy()
        y_proba = g["pred_proba"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _save_mpl(fig, out_path)


def plot_roc_curves_interactive(
    df_pred: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    if go is None:
        return
    df = _filter_models(df_pred, model_names)
    if df.empty:
        return
    fig = go.Figure()
    for model_name, g in df.groupby("model_name"):
        y_true = g["target"].to_numpy()
        y_proba = g["pred_proba"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc(fpr, tpr):.3f})",
                customdata=np.asarray(thresholds).reshape(-1, 1),
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<br>Thr=%{customdata[0]:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance",
            line=dict(dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="False positive rate", yaxis_title="True positive rate"
    )
    _save_plotly(fig, out_path)


# Calibration curves
def plot_calibration_curves(
    df_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    df = _filter_models(df_curve, model_names)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("pred_mean")
        ax.plot(
            g["pred_mean"].to_numpy(),
            g["obs_rate"].to_numpy(),
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    _save_mpl(fig, out_path)


def plot_calibration_curves_interactive(
    df_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    if go is None:
        return
    df = _filter_models(df_curve, model_names)
    if df.empty:
        return
    fig = go.Figure()
    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("pred_mean").copy()
        custom = np.stack(
            [
                g.get("n_bin", pd.Series([np.nan] * len(g))).to_numpy(),
                g.get("obs_count", pd.Series([np.nan] * len(g))).to_numpy(),
                g.get("bin_label", pd.Series([""] * len(g))).astype(str).to_numpy(),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=g["pred_mean"],
                y=g["obs_rate"],
                mode="lines+markers",
                name=model_name,
                customdata=custom,
                hovertemplate=(
                    "Predicted=%{x:.3f}<br>Observed=%{y:.3f}"
                    "<br>n=%{customdata[0]}<br>events=%{customdata[1]}"
                    "<br>bin=%{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="ideal", line=dict(dash="dash"), hoverinfo="skip"
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Predicted probability", yaxis_title="Observed event rate"
    )
    _save_plotly(fig, out_path)


# Decision curves
def plot_decision_curves(
    df_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
    use_standardized: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    metric_col = "standardized_net_benefit" if use_standardized else "net_benefit"
    ref_names = ["treat_none", "treat_all"]
    keep = list(model_names or []) + ref_names
    df = (
        df_curve[df_curve["model_name"].isin(keep)].copy()
        if model_names is not None
        else df_curve.copy()
    )
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("threshold")
        ax.plot(g["threshold"].to_numpy(), g[metric_col].to_numpy(), linewidth=2, label=model_name)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Standardized net benefit" if use_standardized else "Net benefit")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False)
    _save_mpl(fig, out_path)


def plot_decision_curves_interactive(
    df_curve: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
    use_standardized: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    if go is None:
        return
    metric_col = "standardized_net_benefit" if use_standardized else "net_benefit"
    ref_names = ["treat_none", "treat_all"]
    keep = list(model_names or []) + ref_names
    df = (
        df_curve[df_curve["model_name"].isin(keep)].copy()
        if model_names is not None
        else df_curve.copy()
    )
    if df.empty:
        return
    fig = go.Figure()
    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("threshold")
        fig.add_trace(
            go.Scatter(
                x=g["threshold"],
                y=g[metric_col],
                mode="lines",
                name=model_name,
                hovertemplate="Threshold=%{x:.3f}<br>NB=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Threshold probability",
        yaxis_title="Standardized net benefit" if use_standardized else "Net benefit",
    )
    if xlim is not None:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    _save_plotly(fig, out_path)


# Probability distribution (sorted, coloured by outcome, Youden line)
def plot_probability_distribution(
    df_pred: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    """
    For each model: cases sorted ascending by predicted probability on x-axis,
    predicted probability on y-axis. Positive cases in red, negative in blue.
    Youden threshold shown as horizontal dashed line.
    """
    df = _filter_models(df_pred, model_names)
    if df.empty:
        return
    models = sorted(df["model_name"].unique().tolist())
    n_models = len(models)
    fig, axes = plt.subplots(
        nrows=n_models, ncols=1, figsize=(10, max(3, 3 * n_models)), squeeze=False
    )
    for ax, model_name in zip(axes[:, 0], models):
        g = df[df["model_name"] == model_name].copy()
        g = g.sort_values("pred_proba").reset_index(drop=True)
        x = np.arange(len(g))
        colors = ["red" if t == 1 else "steelblue" for t in g["target"]]
        ax.bar(x, g["pred_proba"], color=colors, width=1.0, alpha=0.8)
        if "selected_threshold" in g.columns and g["selected_threshold"].notna().any():
            thr = float(g["selected_threshold"].dropna().iloc[0])
            ax.axhline(
                thr, linestyle="--", linewidth=1.5, color="black", label=f"Youden thr={thr:.3f}"
            )
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("Case (sorted by probability)")
        ax.set_ylabel("Predicted probability")
        ax.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_mpl(fig, out_path)


def plot_probability_distribution_interactive(
    df_pred: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    if go is None:
        return
    df = _filter_models(df_pred, model_names)
    if df.empty:
        return
    models = sorted(df["model_name"].unique().tolist())
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=len(models), cols=1, subplot_titles=models)

    for row_idx, model_name in enumerate(models, start=1):
        g = df[df["model_name"] == model_name].copy()
        g = g.sort_values("pred_proba").reset_index(drop=True)
        x = np.arange(len(g))
        colors = ["red" if t == 1 else "steelblue" for t in g["target"]]
        patient_ids = (
            g["patient_id"].astype(str).tolist() if "patient_id" in g.columns else x.tolist()
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=g["pred_proba"],
                marker_color=colors,
                name=model_name,
                showlegend=False,
                customdata=np.stack([patient_ids, g["target"]], axis=1),
                hovertemplate="Patient=%{customdata[0]}<br>Target=%{customdata[1]}<br>Prob=%{y:.3f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        if "selected_threshold" in g.columns and g["selected_threshold"].notna().any():
            thr = float(g["selected_threshold"].dropna().iloc[0])
            fig.add_hline(
                y=thr,
                line_dash="dash",
                line_color="black",
                row=row_idx,
                col=1,
                annotation_text=f"Thr={thr:.3f}",
                annotation_position="right",
            )
    fig.update_layout(title=title, height=max(400, 300 * len(models)))
    _save_plotly(fig, out_path)


# Prediction scatter (regression)
def plot_prediction_scatter(
    df_pred: pd.DataFrame,
    out_path: Path,
    title: str,
    model_names: Optional[Sequence[str]] = None,
) -> None:
    df = _filter_models(df_pred, model_names)
    if df.empty:
        return
    models = sorted(df["model_name"].unique().tolist())
    n_models = len(models)
    fig, axes = plt.subplots(
        nrows=n_models, ncols=1, figsize=(6, max(3, 3 * n_models)), squeeze=False
    )
    for ax, model_name in zip(axes[:, 0], models):
        g = df[df["model_name"] == model_name]
        obs = g["target"].to_numpy()
        pred = g["pred_value"].to_numpy()
        ax.scatter(obs, pred, alpha=0.8)
        lo = np.nanmin([np.nanmin(obs), np.nanmin(pred)])
        hi = np.nanmax([np.nanmax(obs), np.nanmax(pred)])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")
        ax.set_title(model_name)
        ax.set_xlabel("Observed value")
        ax.set_ylabel("Predicted value")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_mpl(fig, out_path)


# Coefficient bar charts
def plot_coefficients(
    df_coef: pd.DataFrame,
    out_path: Path,
    title: str,
    model_name: str,
    stage: Optional[str] = None,
) -> None:
    df = df_coef[df_coef["model_name"] == model_name].copy()
    if stage is not None and "stage" in df.columns:
        df = df[df["stage"] == stage].copy()
    if df.empty:
        return
    df = df.sort_values("coefficient")
    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(df) + 1)))
    ax.barh(y_pos, df["coefficient"].to_numpy())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].tolist())
    ax.axvline(0.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    _save_mpl(fig, out_path)


# Convenience dispatcher called from run_task scripts
def generate_classification_plots(
    test_predictions: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    decision_curve: pd.DataFrame,
    coefficients: pd.DataFrame,
    plots_dir: Path,
    task_name: str,
    key_models: Optional[Sequence[str]] = None,
    interactive_plots_dir: Optional[Path] = None,
    dca_xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Generate all classification-related plots for a given task."""
    roc_dir = plots_dir / "roc"
    cal_dir = plots_dir / "calibration"
    dca_dir = plots_dir / "decision_curve"
    prob_dir = plots_dir / "probability_distribution"
    coef_dir = plots_dir / "coefficients"

    plot_roc_curves(
        test_predictions, roc_dir / f"roc_{task_name}.png", f"ROC — {task_name}", key_models
    )
    plot_calibration_curves(
        calibration_curve,
        cal_dir / f"calibration_{task_name}.png",
        f"Calibration — {task_name}",
        key_models,
    )
    plot_decision_curves(
        decision_curve,
        dca_dir / f"decision_curve_{task_name}.png",
        f"Decision curve — {task_name}",
        key_models,
        False,
        dca_xlim,
    )
    plot_probability_distribution(
        test_predictions,
        prob_dir / f"prob_dist_{task_name}.png",
        f"Probability distribution — {task_name}",
        key_models,
    )

    if interactive_plots_dir is not None:
        plot_roc_curves_interactive(
            test_predictions,
            interactive_plots_dir / f"roc_{task_name}.html",
            f"ROC — {task_name}",
            key_models,
        )
        plot_calibration_curves_interactive(
            calibration_curve,
            interactive_plots_dir / f"calibration_{task_name}.html",
            f"Calibration — {task_name}",
            key_models,
        )
        plot_decision_curves_interactive(
            decision_curve,
            interactive_plots_dir / f"decision_curve_{task_name}.html",
            f"Decision curve — {task_name}",
            key_models,
            False,
            dca_xlim,
        )
        plot_probability_distribution_interactive(
            test_predictions,
            interactive_plots_dir / f"prob_dist_{task_name}.html",
            f"Probability distribution — {task_name}",
            key_models,
        )

    if coefficients is not None and not coefficients.empty:
        models_to_plot = (
            key_models if key_models is not None else coefficients["model_name"].unique().tolist()
        )
        for mn in models_to_plot:
            sub = coefficients[coefficients["model_name"] == mn]
            if sub.empty:
                continue
            if "stage" in sub.columns and sub["stage"].nunique() > 1:
                for stage in sorted(sub["stage"].dropna().unique()):
                    plot_coefficients(
                        coefficients,
                        coef_dir / f"coef_{task_name}_{mn}_{stage}.png",
                        f"Coefficients — {task_name} — {mn} — {stage}",
                        mn,
                        stage,
                    )
            else:
                plot_coefficients(
                    coefficients,
                    coef_dir / f"coef_{task_name}_{mn}.png",
                    f"Coefficients — {task_name} — {mn}",
                    mn,
                    None,
                )


def generate_regression_plots(
    test_predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    plots_dir: Path,
    task_name: str,
    key_models: Optional[Sequence[str]] = None,
) -> None:
    """Generate all regression-related plots for a given task."""
    scatter_dir = plots_dir / "prediction_scatter"
    coef_dir = plots_dir / "coefficients"

    plot_prediction_scatter(
        test_predictions,
        scatter_dir / f"scatter_{task_name}.png",
        f"Observed vs predicted — {task_name}",
        key_models,
    )

    if coefficients is not None and not coefficients.empty:
        models_to_plot = (
            key_models if key_models is not None else coefficients["model_name"].unique().tolist()
        )
        for mn in models_to_plot:
            sub = coefficients[coefficients["model_name"] == mn]
            if sub.empty:
                continue
            if "stage" in sub.columns and sub["stage"].nunique() > 1:
                for stage in sorted(sub["stage"].dropna().unique()):
                    plot_coefficients(
                        coefficients,
                        coef_dir / f"coef_{task_name}_{mn}_{stage}.png",
                        f"Coefficients — {task_name} — {mn} — {stage}",
                        mn,
                        stage,
                    )
            else:
                plot_coefficients(
                    coefficients,
                    coef_dir / f"coef_{task_name}_{mn}.png",
                    f"Coefficients — {task_name} — {mn}",
                    mn,
                    None,
                )
