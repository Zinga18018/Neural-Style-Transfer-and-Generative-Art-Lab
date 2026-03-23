"""
Visualisation Utilities
========================
Plotly-based chart helpers for analysing generated images.
All charts use a dark theme (bg #0a0a0a, white text).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ---------------------------------------------------------------------------
# Shared dark layout
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    font=dict(color="white", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ---------------------------------------------------------------------------
# RGB Histogram
# ---------------------------------------------------------------------------

def color_histogram(image: Image.Image) -> go.Figure:
    """
    Return a Plotly figure showing the RGB histogram of *image*.
    """
    arr = np.array(image.convert("RGB"))
    fig = go.Figure()

    for ch, colour, name in [(0, "#ff4444", "Red"),
                              (1, "#44ff44", "Green"),
                              (2, "#4488ff", "Blue")]:
        vals = arr[:, :, ch].ravel()
        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=256,
            marker_color=colour,
            opacity=0.55,
            name=name,
        ))

    fig.update_layout(
        title="RGB Colour Histogram",
        xaxis_title="Pixel Intensity",
        yaxis_title="Count",
        barmode="overlay",
        height=340,
        **_DARK_LAYOUT,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Feature-Map Grid
# ---------------------------------------------------------------------------

def feature_map_grid(feature_maps: dict[str, "torch.Tensor"]) -> go.Figure:
    """
    Display the first 8 channels of each layer's feature maps in a grid.

    Parameters
    ----------
    feature_maps : dict[str, torch.Tensor]
        Mapping of layer name → (1, C, H, W) tensor.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import torch  # deferred import — only needed when visualising features

    layers = list(feature_maps.keys())
    n_cols = min(8, min(fm.shape[1] for fm in feature_maps.values()))
    n_rows = len(layers)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{l} ch{c}" for l in layers for c in range(n_cols)],
        vertical_spacing=0.03,
        horizontal_spacing=0.01,
    )

    for r, layer in enumerate(layers, start=1):
        fm = feature_maps[layer].detach().squeeze(0)  # (C, H, W)
        for c in range(n_cols):
            channel = fm[c].cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=channel,
                    colorscale="Viridis",
                    showscale=False,
                ),
                row=r,
                col=c + 1,
            )

    fig.update_layout(
        height=220 * n_rows,
        title="VGG19 Feature Maps",
        showlegend=False,
        **_DARK_LAYOUT,
    )
    # Hide axes for cleanliness
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
