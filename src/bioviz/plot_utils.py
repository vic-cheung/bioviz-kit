"""
Helper functions for processing and annotating plotting data.
"""

# %%
import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes

# Expose all public functions
__all__ = [
    "resolve_font_family",
    "adjust_legend",
    "get_oncoplot_fig_top_margin",
    "get_scaled_oncoplot_dimensions",
    "get_oncoplot_dimensions_fixed_cell",
]


# %%
def resolve_font_family() -> str | None:
    """
    Return the first configured font family from rcParams when available.
    """
    rc_family = mpl.rcParams.get("font.family")
    if isinstance(rc_family, (list, tuple)) and rc_family:
        return rc_family[0]
    if isinstance(rc_family, str) and rc_family:
        return rc_family
    return None


# %%
def adjust_legend(
    ax: Axes, bbox: tuple[float, float], loc: str = "center left", redraw: bool = False
) -> None:
    """
    Adjust an axes legend position and optionally redraw the figure.

    Args:
       ax: Matplotlib `Axes` instance containing the legend.
       bbox: Tuple of (x, y) coordinates to anchor the legend bbox.
       loc: Legend location string (default: "center left").
       redraw: If True, trigger a canvas redraw after moving the legend.
    """
    leg = ax.get_legend()
    if leg:
        leg.set_bbox_to_anchor(bbox)
        leg.set_loc(loc)
        if redraw:
            ax.figure.canvas.draw_idle()


def get_oncoplot_fig_top_margin(
    fig_height: float | int,
    min_height: float | int = 5.0,
    mid_height: float | int = 33.33,
    max_height: float | int = 131.56,
    min_margin: float | int = 0.78,
    mid_margin: float | int = 0.82,
    max_margin: float | int = 0.85,
) -> float | int:
    """
    Compute a top margin fraction for an oncoplot figure based on figure height.

    This uses a fitted quadratic to interpolate sensible top margins across a
    range of figure heights and clamps values to provided min/max margins.

    Args:
        fig_height: Figure height (in inches) used to determine top margin.
        min_height: Minimum height for interpolation.
        mid_height: Midpoint height for interpolation.
        max_height: Maximum height for interpolation.
        min_margin: Minimum allowed margin fraction.
        mid_margin: Midpoint margin fraction.
        max_margin: Maximum allowed margin fraction.

    Returns:
        A float margin fraction to use for layout spacing.
    """

    x = np.array([min_height, mid_height, max_height])
    y = np.array([min_margin, mid_margin, max_margin])
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    margin = a * fig_height**2 + b * fig_height + c
    margin = max(min_margin, min(max_margin, margin))
    if fig_height < min_height:
        margin = min_margin + (0.90 - min_margin) * (1 - fig_height / min_height)
    if 18.0 <= fig_height <= 22.0:
        margin = 0.85
    elif fig_height > 100:
        margin = 0.82
    elif fig_height > 60:
        margin = 0.84
    return margin


def get_scaled_oncoplot_dimensions(
    ncols: int,
    nrows: int,
    num_top_annotations: int,
    top_annotation_height: float = 1.0,
    col_scale: float = 16 / 9,
    row_scale: float = 10 / 30,
    aspect: float = 1.0,
    max_width: None | float = None,
    max_height: None | float = None,
) -> tuple[float, float]:
    """
    Calculate a scaled figure width and height for an oncoplot given layout parameters.

    Args:
        ncols: Number of columns (patients) in the oncoplot.
        nrows: Number of rows (genes) in the oncoplot.
        num_top_annotations: Number of top annotation rows to reserve.
        top_annotation_height: Height (in inches) for each top annotation row.
        col_scale: Base column width scale factor.
        row_scale: Base row height scale factor.
        aspect: Desired aspect ratio multiplier.
        max_width: Optional cap for figure width.
        max_height: Optional cap for figure height.

    Returns:
        Tuple of (fig_width, fig_height) in inches.
    """

    base_width = ncols * col_scale
    base_height = nrows * row_scale + num_top_annotations * top_annotation_height
    min_figure_height = 6.0
    if nrows <= 10:
        if nrows <= 3:
            adjusted_cell_size = min(0.8, 7.0 / max(nrows, 1))
        else:
            adjusted_cell_size = min(0.6, 6.0 / max(nrows, 1))
        desired_height = max(nrows * adjusted_cell_size, min_figure_height)
        base_height = max(base_height, desired_height)
    if ncols <= 10:
        adjusted_cell_width = min(0.8, 8.0 / max(ncols, 1))
        desired_width = ncols * adjusted_cell_width
        base_width = min(base_width, desired_width)
    if nrows == 1:
        base_height = max(min_figure_height, base_height)
    if ncols == 1:
        min_width = 2.0
        base_width = max(min_width, base_width)
    if aspect < 1:
        fig_width = base_width / aspect
        fig_height = base_height
    elif aspect > 1:
        fig_width = base_width
        fig_height = base_height * aspect
    else:
        fig_width = base_width
        fig_height = base_height
    if max_width is not None:
        fig_width = min(fig_width, max_width)
    if max_height is not None:
        fig_height = min(fig_height, max_height)
    return (fig_width, fig_height)


def get_oncoplot_dimensions_fixed_cell(
    ncols: int,
    nrows: int,
    target_cell_width: float = 0.8,
    target_cell_height: float = 0.8,
    min_width: float = 8.0,
    max_width: float = 200.0,
    max_height: float = 60.0,
    num_top_annotations: int = 0,
    top_annotation_height: float = 0.25,
    aspect: float = 1.0,
    auto_adjust_cell_size: bool = True,
) -> tuple[float, float]:
    """
    Compute figure width/height when cells should target fixed sizes, with optional auto-adjustment.

    Args:
        ncols: Number of columns (patients).
        nrows: Number of rows (genes).
        target_cell_width: Desired width per cell (inches).
        target_cell_height: Desired height per cell (inches).
        min_width: Minimum figure width.
        max_width: Maximum figure width.
        max_height: Maximum figure height.
        num_top_annotations: Number of top annotation rows to include.
        top_annotation_height: Height per top annotation row.
        aspect: Width/height aspect ratio to enforce.
        auto_adjust_cell_size: If True, scales cell size down to fit large grids.

    Returns:
        Tuple of (width, height, font_scale_factor) where width/height are inches and
        font_scale_factor is a multiplier to apply to font sizes to maintain readability.
    """

    adjusted_cell_width = target_cell_width
    adjusted_cell_height = target_cell_height
    if auto_adjust_cell_size:
        patient_threshold = int(0.6 * max_width / target_cell_width)
        gene_threshold = int(0.6 * max_height / target_cell_height)
        min_cell_ratio = 1 / 3
        min_cell_width = target_cell_width * min_cell_ratio
        min_cell_height = target_cell_height * min_cell_ratio
        if ncols > patient_threshold:
            scale_factor = min(1.0, 0.9 * (patient_threshold / ncols) ** 0.4)
            adjusted_cell_width = max(min_cell_width, target_cell_width * scale_factor)
        if nrows > gene_threshold:
            scale_factor = min(1.0, 0.95 * (gene_threshold / nrows) ** 0.3)
            adjusted_cell_height = max(min_cell_height, target_cell_height * scale_factor)
    width = ncols * adjusted_cell_width
    height = nrows * adjusted_cell_height
    if num_top_annotations > 0:
        height += num_top_annotations * top_annotation_height
    padding_width = 1.0
    padding_height = 1.0
    width += padding_width
    height += padding_height
    if height > max_height:
        scale_factor = max_height / height
        height = max_height
        width = width * scale_factor
    if aspect != 1.0:
        target_width = height / aspect
        if target_width > width:
            width = target_width
        else:
            height = width * aspect
    width = min(max(width, min_width), max_width)
    height = min(height, max_height)
    width_scale_factor = 1.0
    height_scale_factor = 1.0
    if auto_adjust_cell_size:
        if ncols > patient_threshold:
            width_scale_factor = min(1.0, 0.9 * (patient_threshold / ncols) ** 0.4)
        if nrows > gene_threshold:
            height_scale_factor = min(1.0, 0.95 * (gene_threshold / nrows) ** 0.3)
    font_scale_factor = min(width_scale_factor, height_scale_factor)
    if font_scale_factor < 1.0:
        font_scale_factor = min(1.0, font_scale_factor**0.6)
    return (width, height, font_scale_factor)
