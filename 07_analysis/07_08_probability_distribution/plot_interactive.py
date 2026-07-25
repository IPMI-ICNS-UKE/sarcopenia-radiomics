from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# ── Colour helpers ────────────────────────────────────────────────────────────
def _bar_color(target: int, sex: float, cfg) -> str:
    is_male = sex == 1
    if int(target) == 1:
        return cfg.COLOR_POSITIVE_MALE if is_male else cfg.COLOR_POSITIVE_FEMALE
    return cfg.COLOR_NEGATIVE_MALE if is_male else cfg.COLOR_NEGATIVE_FEMALE


def _is_error(row: pd.Series) -> bool:
    return int(row["pred_label"]) != int(row["target"])


def _result_label(row: pd.Series) -> str:
    t, p = int(row["target"]), int(row["pred_label"])
    if t == 1 and p == 1:
        return "TP"
    if t == 0 and p == 0:
        return "TN"
    if t == 0 and p == 1:
        return "FP ✗"
    return "FN ✗"


# ── Core plotting routine ─────────────────────────────────────────────────────
def plot_probability_distribution_interactive(
    df: pd.DataFrame,
    out_path: Path,
    split_label: str,
    display_name: str,
    cfg,
    threshold: float | None = None,
) -> None:
    if go is None:
        print("[plot_interactive] plotly not available, skipping.")
        return
    if df is None or df.empty:
        print(f"[plot_interactive] Empty data for {out_path}, skipping.")
        return

    g = df.sort_values("pred_proba").reset_index(drop=True)
    n = len(g)
    x = np.arange(n, dtype=float)

    colors = [_bar_color(row["target"], row["sex"], cfg) for _, row in g.iterrows()]
    is_error = [_is_error(row) for _, row in g.iterrows()]
    results = [_result_label(row) for _, row in g.iterrows()]
    sex_str = ["Male" if s == 1 else "Female" if s == 0 else "Unknown" for s in g["sex"]]

    bar_widths = [
        0.7 * (1 - cfg.BAR_GAP_FRACTION) if err else (1 - cfg.BAR_GAP_FRACTION) for err in is_error
    ]

    # ── Legend groups ─────────────────────────────────────────────────────────
    # Use invisible scatter traces for the legend entries, then real bars.
    fig = go.Figure()

    legend_specs = [
        ("Sarcopenia (male)", cfg.COLOR_POSITIVE_MALE, False),
        ("No sarcopenia (male)", cfg.COLOR_NEGATIVE_MALE, False),
        ("Sarcopenia (female)", cfg.COLOR_POSITIVE_FEMALE, False),
        ("No sarcopenia (female)", cfg.COLOR_NEGATIVE_FEMALE, False),
        ("Misclassified (FP / FN)", "#555555", True),
    ]
    for leg_name, leg_color, is_hatched in legend_specs:
        marker_kwargs = dict(color=leg_color, size=12, symbol="square")
        if is_hatched:
            marker_kwargs.update(
                line=dict(color="#555555", width=2),
                symbol="x-open-dot",
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=marker_kwargs,
                name=leg_name,
                showlegend=True,
            )
        )

    # ── Bars ──────────────────────────────────────────────────────────────────
    hover = [
        f"<b>{pid}</b><br>Probability: {p:.3f}<br>True label: {int(t)}<br>"
        f"Prediction: {pred}<br>Sex: {sx}<br>Result: {res}"
        for pid, p, t, pred, sx, res in zip(
            g["patient_id"],
            g["pred_proba"],
            g["target"],
            g["pred_label"],
            sex_str,
            results,
        )
    ]

    fig.add_trace(
        go.Bar(
            x=x,
            y=g["pred_proba"],
            marker=dict(
                color=colors,
                line=dict(
                    color=["#555555" if err else "rgba(0,0,0,0)" for err in is_error],
                    width=[1.5 if err else 0.0 for err in is_error],
                ),
            ),
            width=bar_widths,
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
            name="",
        )
    )

    # Overlay cross markers for misclassified bars
    err_x = [xi for xi, err in zip(x, is_error) if err]
    err_y = [float(p) for p, err in zip(g["pred_proba"], is_error) if err]
    if err_x:
        fig.add_trace(
            go.Scatter(
                x=err_x,
                y=err_y,
                mode="markers",
                marker=dict(symbol="x", size=8, color="#2C3E50", line=dict(width=2)),
                showlegend=False,
                hoverinfo="skip",
                name="",
            )
        )

    # ── Threshold line ────────────────────────────────────────────────────────
    thr = threshold
    if thr is None and "selected_threshold" in g.columns and g["selected_threshold"].notna().any():
        thr = float(g["selected_threshold"].dropna().iloc[0])
    if thr is not None:
        fig.add_shape(
            type="line",
            x0=-0.8,
            x1=n - 0.2,
            y0=thr,
            y1=thr,
            line=dict(color=cfg.THRESHOLD_LINE_COLOR, width=cfg.THRESHOLD_LINE_WIDTH, dash="dash"),
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(
                    color=cfg.THRESHOLD_LINE_COLOR, width=cfg.THRESHOLD_LINE_WIDTH, dash="dash"
                ),
                name=f"Youden thr = {thr:.3f}",
            )
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{display_name}</b>  ·  {split_label}",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Cases (sorted by predicted probability)",
            tickmode="array",
            tickvals=list(x),
            ticktext=g["patient_id"].tolist(),
            tickfont=dict(size=8),
            tickangle=90,
        ),
        yaxis=dict(
            title="Predicted probability",
            range=[0, 1.05],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1100,
        height=500,
        bargap=0,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[plot_interactive] Saved: {out_path}")
