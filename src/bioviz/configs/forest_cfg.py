"""Configuration for Forest plots.

Forest plots display hazard ratios with confidence intervals from survival
analysis. Font sizes default to None to inherit from matplotlib rcParams.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

MarkerStyle = Literal["s", "o", "D", "^", "v", "<", ">", "p", "*", "h"]


class ForestPlotConfig(BaseModel):
    """Configuration for forest plot generation.

    Notes
    -----
    - Column names (hr_col, ci_lower_col, etc.) describe your data schema.
    - Font sizes default to None to inherit from rcParams (e.g., RVMDStyle.apply_theme()).
    - Use log_scale=True for standard HR visualization (centered around 1.0).
    """

    # ==========================================================================
    # Column mapping (data schema)
    # ==========================================================================
    hr_col: Annotated[
        str,
        Field(default="hr", description="Column name for hazard ratio values"),
    ]
    ci_lower_col: Annotated[
        str,
        Field(default="ci_lower", description="Column name for lower CI bound"),
    ]
    ci_upper_col: Annotated[
        str,
        Field(default="ci_upper", description="Column name for upper CI bound"),
    ]
    label_col: Annotated[
        str,
        Field(default="comparator", description="Column name for row labels"),
    ]
    pvalue_col: Annotated[
        str,
        Field(default="p_value", description="Column name for p-values"),
    ]
    reference_col: Annotated[
        Optional[str],
        Field(default="reference", description="Column name for reference group"),
    ]
    variable_col: Annotated[
        Optional[str],
        Field(default="variable", description="Column for variable grouping (multi-section plots)"),
    ]

    # ==========================================================================
    # Figure layout
    # ==========================================================================
    figsize: Annotated[
        Tuple[float, float],
        Field(default=(10.0, 8.0), description="Figure size (width, height) in inches"),
    ]
    title: Annotated[
        Optional[str],
        Field(default=None, description="Plot title"),
    ]
    xlabel: Annotated[
        str,
        Field(default="Hazard Ratio (95% CI)", description="X-axis label"),
    ]

    # ==========================================================================
    # Reference line
    # ==========================================================================
    show_reference_line: Annotated[
        bool,
        Field(default=True, description="Show vertical line at HR=1"),
    ]
    reference_line_color: Annotated[
        str,
        Field(default="#D32F2F", description="Reference line color"),
    ]
    reference_line_style: Annotated[
        str,
        Field(default="--", description="Reference line style"),
    ]
    reference_line_width: Annotated[
        float,
        Field(default=1.5, ge=0.0, description="Reference line width"),
    ]

    # ==========================================================================
    # Statistics table
    # ==========================================================================
    show_stats_table: Annotated[
        bool,
        Field(default=True, description="Show HR/CI/p-value table on right side"),
    ]
    stats_table_x_position: Annotated[
        float,
        Field(default=1.05, description="X-position for stats table (>1.0 = right of plot)"),
    ]
    stats_table_col_spacing: Annotated[
        float,
        Field(default=0.15, description="Spacing between stats table columns"),
    ]
    stats_fontsize: Annotated[
        Optional[int],
        Field(default=None, ge=1, description="Stats table font size. None uses rcParams."),
    ]

    # ==========================================================================
    # Scale and axis
    # ==========================================================================
    log_scale: Annotated[
        bool,
        Field(default=False, description="Use log scale for x-axis"),
    ]
    xlim: Annotated[
        Optional[Tuple[float, float]],
        Field(default=None, description="X-axis limits (min, max)"),
    ]
    xticks: Annotated[
        Optional[List[float]],
        Field(default=None, description="Custom x-tick positions"),
    ]
    center_around_null: Annotated[
        bool,
        Field(default=False, description="Center x-axis symmetrically around HR=1"),
    ]

    # ==========================================================================
    # Colors and significance
    # ==========================================================================
    color_significant: Annotated[
        str,
        Field(default="#2E7D32", description="Color for significant results (p < alpha)"),
    ]
    color_nonsignificant: Annotated[
        str,
        Field(default="#757575", description="Color for non-significant results"),
    ]
    marker_color_significant: Annotated[
        Optional[str],
        Field(
            default=None, description="Marker color for significant. None uses color_significant."
        ),
    ]
    marker_color_nonsignificant: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Marker color for non-significant. None uses color_nonsignificant.",
        ),
    ]
    alpha_threshold: Annotated[
        float,
        Field(default=0.05, ge=0.0, le=1.0, description="Significance threshold"),
    ]

    # ==========================================================================
    # Markers and error bars
    # ==========================================================================
    marker_size: Annotated[
        float,
        Field(default=8.0, ge=0.0, description="Size of point markers"),
    ]
    marker_style: Annotated[
        MarkerStyle,
        Field(default="s", description="Marker style (s=square, o=circle, D=diamond, etc.)"),
    ]
    linewidth: Annotated[
        float,
        Field(default=2.0, ge=0.0, description="Error bar line width"),
    ]
    show_caps: Annotated[
        bool,
        Field(default=False, description="Show caps on error bars"),
    ]
    capsize: Annotated[
        float,
        Field(default=2.0, ge=0.0, description="Size of error bar caps"),
    ]

    # ==========================================================================
    # Section styling (multi-variable plots)
    # ==========================================================================
    section_labels: Annotated[
        Optional[Dict[str, str]],
        Field(default=None, description="Custom section labels: {variable: 'Display Name'}"),
    ]
    show_section_separators: Annotated[
        bool,
        Field(default=True, description="Show separator lines between sections"),
    ]
    section_separator_color: Annotated[
        str,
        Field(default="blue", description="Color of section separator lines"),
    ]
    section_separator_alpha: Annotated[
        float,
        Field(default=0.25, ge=0.0, le=1.0, description="Transparency of separator lines"),
    ]
    section_gap: Annotated[
        float,
        Field(default=0.0, description="Extra vertical spacing between sections"),
    ]
    section_label_x_position: Annotated[
        float,
        Field(default=-0.35, description="X-position for section labels (negative = left)"),
    ]

    # ==========================================================================
    # Grid and spines
    # ==========================================================================
    show_grid: Annotated[
        bool,
        Field(default=False, description="Show vertical grid lines"),
    ]
    show_y_spine: Annotated[
        bool,
        Field(default=False, description="Show left y-axis spine"),
    ]
    show_yticks: Annotated[
        bool,
        Field(default=False, description="Show y-axis tick marks"),
    ]
    y_margin: Annotated[
        float,
        Field(default=0.5, ge=0.0, description="Padding above/below plot in row units"),
    ]

    # ==========================================================================
    # Font sizes (None = use rcParams)
    # ==========================================================================
    ytick_fontsize: Annotated[
        Optional[int],
        Field(default=None, ge=1, description="Y-axis label font size. None uses rcParams."),
    ]
    xtick_fontsize: Annotated[
        Optional[int],
        Field(default=None, ge=1, description="X-tick label font size. None uses rcParams."),
    ]
    xlabel_fontsize: Annotated[
        Optional[int],
        Field(default=None, ge=1, description="X-axis label font size. None uses rcParams."),
    ]
    title_fontsize: Annotated[
        Optional[int],
        Field(default=None, ge=1, description="Title font size. None uses rcParams."),
    ]

    # ==========================================================================
    # Category ordering
    # ==========================================================================
    category_order: Annotated[
        Optional[Dict[str, List[Any]]],
        Field(
            default=None,
            description=(
                "Custom ordering for variable sections and categories. "
                "Dict maps variable names to lists of comparator values in display order."
            ),
        ),
    ]

    class Config:
        """Pydantic config."""

        extra = "forbid"


__all__ = ["ForestPlotConfig"]
