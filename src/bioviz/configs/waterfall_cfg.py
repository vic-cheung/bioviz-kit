"""Configuration for waterfall plots.

This provides a pydantic-based configuration model for waterfall plots,
following the same pattern as other bioviz-kit plotters.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ThresholdLine(BaseModel):
    """Configuration for a horizontal threshold line on the waterfall plot."""

    value: Annotated[float, Field(description="Y-value for the threshold line.")]
    color: Annotated[str, Field(default="red", description="Color of the line.")]
    linestyle: Annotated[str, Field(default="--", description="Line style (e.g., '--', '-', ':').")]
    linewidth: Annotated[float, Field(default=1.0, description="Line width.")]
    label: Annotated[str | None, Field(default=None, description="Label for legend.")]


class WaterfallConfig(BaseModel):
    """Configuration for waterfall plot generation.

    A waterfall plot displays sorted bar values, commonly used for:
    - Best percent change from baseline (tumor response)
    - VAF change from baseline (ctDNA molecular response)
    - Any sorted value comparison across samples

    Notes
    -----
    - Field names match common waterfall plot parameters.
    - value_col is required; other columns are optional.
    - Font sizes default to None to inherit from rcParams.
    """

    # ==========================================================================
    # Required data columns
    # ==========================================================================
    value_col: Annotated[
        str,
        Field(..., description="Column containing values to plot (e.g., percent change)."),
    ]

    # ==========================================================================
    # Optional data columns
    # ==========================================================================
    id_col: Annotated[
        str | None,
        Field(default=None, description="Column for sample/patient IDs (x-axis labels)."),
    ]
    color_col: Annotated[
        str | None,
        Field(default=None, description="Column for bar colors (categorical, e.g., BOR)."),
    ]
    group_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column for aggregation grouping (used with aggregate parameter).",
        ),
    ]
    facet_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column for creating faceted (small multiple) plots.",
        ),
    ]

    # ==========================================================================
    # Sorting and aggregation
    # ==========================================================================
    sort_ascending: Annotated[
        bool,
        Field(
            default=False,
            description="Sort bars ascending (True) or descending (False).",
        ),
    ]
    aggregate: Annotated[
        str | None,
        Field(
            default=None,
            description="Aggregation function when using group_col ('mean', 'sum', 'median').",
        ),
    ]

    # ==========================================================================
    # Color configuration
    # ==========================================================================
    palette: Annotated[
        dict[str, str] | list[str] | None,
        Field(
            default=None,
            description="Color palette. Dict maps color_col values to colors, or list of colors.",
        ),
    ]
    default_color: Annotated[
        str,
        Field(
            default="steelblue",
            description="Default bar color when color_col is not specified.",
        ),
    ]

    # ==========================================================================
    # Threshold lines
    # ==========================================================================
    threshold_lines: Annotated[
        list[ThresholdLine] | None,
        Field(
            default=None,
            description="List of horizontal threshold lines to draw (e.g., -30%, -100%).",
        ),
    ]
    show_zero_line: Annotated[
        bool,
        Field(default=True, description="Whether to show a horizontal line at y=0."),
    ]
    zero_line_color: Annotated[
        str,
        Field(default="black", description="Color of the zero line."),
    ]
    zero_line_width: Annotated[
        float,
        Field(default=0.5, description="Width of the zero line."),
    ]

    # ==========================================================================
    # Figure settings
    # ==========================================================================
    figsize: Annotated[
        tuple[float, float],
        Field(default=(10, 6), description="Figure size (width, height) in inches."),
    ]

    # ==========================================================================
    # Labels and title
    # ==========================================================================
    title: Annotated[
        str | None,
        Field(default=None, description="Plot title."),
    ]
    title_fontweight: Annotated[
        str,
        Field(
            default="normal",
            description="Font weight for title ('normal', 'bold', etc.).",
        ),
    ]
    xlabel: Annotated[
        str | None,
        Field(default=None, description="X-axis label."),
    ]
    ylabel: Annotated[
        str | None,
        Field(default=None, description="Y-axis label."),
    ]
    title_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for title. None uses rcParams."),
    ]
    xlabel_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for x-axis label. None uses rcParams."),
    ]
    ylabel_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for y-axis label. None uses rcParams."),
    ]
    tick_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for tick labels. None uses rcParams."),
    ]

    # ==========================================================================
    # Axis limits
    # ==========================================================================
    xlim: Annotated[
        tuple[float | None, float | None] | None,
        Field(default=None, description="X-axis limits (min, max)."),
    ]
    ylim: Annotated[
        tuple[float | None, float | None] | None,
        Field(default=None, description="Y-axis limits (min, max)."),
    ]
    xticks: Annotated[
        list[float] | None,
        Field(default=None, description="Custom x-axis tick positions."),
    ]
    yticks: Annotated[
        list[float] | None,
        Field(default=None, description="Custom y-axis tick positions."),
    ]

    # ==========================================================================
    # X-axis tick display
    # ==========================================================================
    show_xticks: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to show x-axis tick labels (sample IDs).",
        ),
    ]
    xtick_rotation: Annotated[
        float,
        Field(default=90, description="Rotation angle for x-axis tick labels."),
    ]

    # ==========================================================================
    # Grouped waterfall mode
    # ==========================================================================
    sort_within_group_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column to group bars by, with sorting within each group. "
            "Enables grouped waterfall mode with gaps between groups.",
        ),
    ]
    group_order: Annotated[
        list[str] | None,
        Field(default=None, description="Order of groups (for sort_within_group_col)."),
    ]
    group_gap: Annotated[
        int,
        Field(default=5, description="Gap size (in bar positions) between groups."),
    ]
    show_group_counts: Annotated[
        bool,
        Field(default=True, description="Whether to show (n=X) counts in group labels."),
    ]
    show_group_separators: Annotated[
        bool,
        Field(default=False, description="Whether to show vertical lines between groups."),
    ]
    group_separator_color: Annotated[
        str,
        Field(default="black", description="Color for group separator lines."),
    ]
    group_separator_style: Annotated[
        str,
        Field(default="--", description="Line style for group separators."),
    ]
    group_separator_width: Annotated[
        float,
        Field(default=2.0, description="Line width for group separators."),
    ]
    group_separator_alpha: Annotated[
        float,
        Field(default=0.5, description="Alpha for group separators."),
    ]

    # ==========================================================================
    # Edge color by column (separate from fill color)
    # ==========================================================================
    edgecolor_col: Annotated[
        str | None,
        Field(
            default=None,
            description="Column for bar edge colors (different from fill color_col).",
        ),
    ]
    edgecolor_palette: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Palette for edge colors when edgecolor_col is used.",
        ),
    ]

    # ==========================================================================
    # Per-bar text annotations
    # ==========================================================================
    bar_annotation_col: Annotated[
        str | None,
        Field(default=None, description="Column for per-bar text annotations."),
    ]
    bar_annotation_rotation: Annotated[
        float,
        Field(default=90, description="Rotation angle for bar annotations."),
    ]
    bar_annotation_fontsize: Annotated[
        float,
        Field(default=6, description="Font size for bar annotations."),
    ]
    bar_annotation_alpha: Annotated[
        float,
        Field(default=0.7, description="Alpha for bar annotation text."),
    ]
    bar_annotation_offset: Annotated[
        float,
        Field(default=3, description="Offset from y=0 for bar annotations."),
    ]

    # ==========================================================================
    # Legend
    # ==========================================================================
    show_legend: Annotated[
        bool,
        Field(default=True, description="Whether to show legend (when color_col is used)."),
    ]
    legend_loc: Annotated[
        str,
        Field(default="upper right", description="Legend location."),
    ]
    legend_bbox_to_anchor: Annotated[
        tuple[float, float] | None,
        Field(
            default=None,
            description="Position legend outside plot. E.g., (1.02, 0.5) for right-center.",
        ),
    ]
    legend_title: Annotated[
        str | None,
        Field(
            default=None,
            description="Legend title. Defaults to color_col if not specified.",
        ),
    ]
    legend_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for legend. None uses rcParams."),
    ]
    legend_title_fontsize: Annotated[
        float | None,
        Field(default=None, description="Font size for legend title."),
    ]
    legend_frameon: Annotated[
        bool,
        Field(default=False, description="Whether to show legend frame."),
    ]

    # ==========================================================================
    # Bar styling
    # ==========================================================================
    bar_width: Annotated[
        float,
        Field(default=0.9, description="Width of bars (0-1 scale)."),
    ]
    edgecolor: Annotated[
        str | None,
        Field(default="black", description="Default edge color for bars."),
    ]
    linewidth: Annotated[
        float,
        Field(default=1.5, description="Edge line width for bars."),
    ]

    model_config = {"extra": "forbid"}
