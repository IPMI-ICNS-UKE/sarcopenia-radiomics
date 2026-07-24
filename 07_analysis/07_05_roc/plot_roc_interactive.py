import os
from typing import List

import plotly.graph_objects as go

from config import (
    LINE_STYLES,
    LINE_WIDTH,
    MODEL_ORDER,
    PALETTE,
    FONT_FAMILY,
    FONT_SIZE_AXIS,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TITLE,
)
from roc_utils import RocData


def _dash_style(dash_pattern) -> str:
    """Map our dash-gap tuple convention to Plotly dash names."""
    if dash_pattern is None:
        return "solid"
    # dash-dot patterns → longdash; pure dash → dash
    if len(dash_pattern) == 4:
        return "dashdot"
    return "dash"


def plot_roc_interactive(
    roc_data_list: List[RocData],
    title: str,
    output_path: str,
) -> None:
    """
    Draw and save an interactive Plotly ROC figure as HTML.

    Parameters
    ----------
    roc_data_list : list of RocData
        Pre-computed ROC data, one entry per model.
    title : str
        Plot title string.
    output_path : str
        Full path for the output HTML file (must end in .html).
    """
    style_map = {
        m: (PALETTE[i], LINE_STYLES[i])
        for i, m in enumerate(MODEL_ORDER)
        if i < len(PALETTE)
    }

    fig = go.Figure()

    for roc in roc_data_list:
        colour, dash = style_map.get(roc.model_name, ("#333333", None))
        dash_str = _dash_style(dash)
        label = f"{roc.display_name}<br>{roc.auc_label}"

        fig.add_trace(
            go.Scatter(
                x=roc.fpr,
                y=roc.tpr,
                mode="lines",
                name=label,
                line=dict(color=colour, dash=dash_str, width=LINE_WIDTH),
                hovertemplate=(
                    f"<b>{roc.display_name}</b><br>"
                    "FPR = %{x:.3f}<br>"
                    "TPR = %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Reference diagonal
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="#bbbbbb", dash="dash", width=1.0),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=FONT_SIZE_TITLE)),
        xaxis=dict(
            title="1 − Specificity (FPR)",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
            dtick=0.2,
        ),
        yaxis=dict(
            title="Sensitivity (TPR)",
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
            tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_AXIS),
            range=[-0.02, 1.02],
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
            dtick=0.2,
        ),
        legend=dict(
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_LEGEND),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            x=0.55,
            y=0.05,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=520,
        height=520,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  [interactive] Saved → {output_path}")
