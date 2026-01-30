"""Configuration for grouped bar plots with optional confidence intervals."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class GroupedBarConfig(BaseModel):
    """
    Configuration for bar plots with optional grouping and confidence intervals.

    Supports:
    - Grouped bar plots (multiple bars per category) or simple bar plots
    - Horizontal or vertical orientation
    - Optional confidence intervals (Clopper-Pearson, bootstrap, or pre-computed)
    - Flexible styling and annotations

    Common use cases:
    - Gene prevalence comparisons across cohorts
    - Response rates by treatment arm
    - Mutation frequency at baseline vs progression
    - Any categorical comparison with or without error bars

    Attributes:
        category_col: Column name for categories (x-axis for vertical, y-axis for horizontal).
        group_col: Column for grouping within categories. None for simple (ungrouped) bars.
        value_col: Column name for bar values.
        orientation: 'horizontal' (barh) or 'vertical' (bar).
        ci_low_col: Column for lower CI bound. None to skip error bars.
        ci_high_col: Column for upper CI bound. None to skip error bars.
        k_col: Column for count (numerator) to compute CI from proportions.
        n_col: Column for total (denominator) to compute CI from proportions.
        ci_method: Method for CI computation: 'clopper', 'bootstrap', or 'none'.
    """

    # ==========================================================================
    # Data column mappings
    # ==========================================================================
    category_col: Annotated[
        str,
        Field(
            default="Category",
            description="Column for categories (e.g., Gene, Pathway, Treatment).",
        ),
    ]
    group_col: Annotated[
        str | None,
        Field(
            default="Group",
            description="Column for grouping within categories. None for ungrouped bars.",
        ),
    ]
    value_col: Annotated[
        str,
        Field(default="value", description="Column for bar values (heights/lengths)."),
    ]

    # ==========================================================================
    # Orientation
    # ==========================================================================
    orientation: Annotated[
        Literal["horizontal", "vertical"],
        Field(
            default="horizontal",
            description="Bar orientation: 'horizontal' (barh) or 'vertical' (bar).",
        ),
    ]

    # ==========================================================================
    # CI columns (optional - None means no error bars)
    # ==========================================================================
    ci_low_col: Annotated[
        str | None,
        Field(default=None, description="Column for lower CI bound. None to skip error bars."),
    ]
    ci_high_col: Annotated[
        str | None,
        Field(default=None, description="Column for upper CI bound. None to skip error bars."),
    ]

    # ==========================================================================
    # CI computation from counts (for proportion CIs)
    # ==========================================================================
    k_col: Annotated[
        str | None,
        Field(default=None, description="Column for count (numerator) to compute proportion CI."),
    ]
    n_col: Annotated[
        str | None,
        Field(default=None, description="Column for total (denominator) to compute proportion CI."),
    ]
    ci_method: Annotated[
        Literal["clopper-pearson", "bootstrap", "none"],
        Field(default="none", description="CI method: 'clopper-pearson', 'bootstrap', or 'none'."),
    ]
    alpha: Annotated[
        float,
        Field(default=0.05, description="Significance level for CI (0.05 = 95% CI)."),
    ]
    n_boot: Annotated[
        int,
        Field(default=10000, description="Number of bootstrap samples (if method='bootstrap')."),
    ]
    random_state: Annotated[
        int | None,
        Field(default=12345, description="Random seed for bootstrap reproducibility."),
    ]

    # ==========================================================================
    # Group configuration (for grouped bars)
    # ==========================================================================
    group_order: Annotated[
        list[str] | None,
        Field(default=None, description="Order of groups within each category."),
    ]
    group_colors: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Mapping of group name to color."),
    ]
    group_labels: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Mapping of group name to display label (e.g., with n=X)."),
    ]

    # ==========================================================================
    # Figure settings
    # ==========================================================================
    figsize: Annotated[
        tuple[float, float] | None,
        Field(default=None, description="Figure size (width, height). Auto-computed if None."),
    ]
    bar_width: Annotated[
        float,
        Field(default=0.28, description="Width/height of each bar."),
    ]
    group_spacing: Annotated[
        float,
        Field(default=1.5, description="Spacing between category groups."),
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
            default="normal", description="Font weight for title ('normal', 'bold', 'light', etc.)."
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
        Field(
            default=None, description="Font size for title. None uses rcParams['axes.titlesize']."
        ),
    ]
    xlabel_fontsize: Annotated[
        float | None,
        Field(
            default=None,
            description="Font size for x-axis label. None uses rcParams['axes.labelsize'].",
        ),
    ]
    ylabel_fontsize: Annotated[
        float | None,
        Field(
            default=None,
            description="Font size for y-axis label. None uses rcParams['axes.labelsize'].",
        ),
    ]
    tick_fontsize: Annotated[
        float | None,
        Field(
            default=None,
            description="Font size for tick labels. None uses rcParams['xtick.labelsize'].",
        ),
    ]

    # ==========================================================================
    # Annotations
    # ==========================================================================
    show_annotations: Annotated[
        bool,
        Field(default=True, description="Whether to show value annotations on bars."),
    ]
    annot_fontsize: Annotated[
        float | None,
        Field(
            default=None,
            description="Font size for value annotations. None uses rcParams['font.size'] * 0.9.",
        ),
    ]
    annot_format: Annotated[
        str,
        Field(default="{:.1f}%", description="Format string for annotations."),
    ]
    annot_offset: Annotated[
        float,
        Field(default=0.8, description="Offset for annotations from bar end (data units)."),
    ]
    annot_padding: Annotated[
        float,
        Field(
            default=5.0,
            description="Extra padding (data units) added to axis limit to ensure annotations fit.",
        ),
    ]

    # ==========================================================================
    # Legend
    # ==========================================================================
    show_legend: Annotated[
        bool,
        Field(default=True, description="Whether to show legend."),
    ]
    legend_loc: Annotated[
        str,
        Field(default="upper left", description="Legend location."),
    ]
    legend_bbox_to_anchor: Annotated[
        tuple[float, float] | None,
        Field(default=(1.02, 1), description="Legend bbox_to_anchor. None for auto."),
    ]
    legend_fontsize: Annotated[
        float | None,
        Field(
            default=None, description="Font size for legend. None uses rcParams['legend.fontsize']."
        ),
    ]
    legend_title: Annotated[
        str | None,
        Field(default=None, description="Legend title."),
    ]

    # ==========================================================================
    # Bar styling
    # ==========================================================================
    default_color: Annotated[
        str,
        Field(default="#1f77b4", description="Default bar color when no group_colors specified."),
    ]
    bar_edgecolor: Annotated[
        str,
        Field(default="white", description="Edge color for bars."),
    ]
    bar_linewidth: Annotated[
        float,
        Field(default=0.9, description="Line width for bar edges."),
    ]

    # ==========================================================================
    # Error bar styling
    # ==========================================================================
    capsize: Annotated[
        float,
        Field(default=4, description="Size of error bar caps."),
    ]
    error_color: Annotated[
        str,
        Field(default="black", description="Color for error bars."),
    ]

    # ==========================================================================
    # Axis limits
    # ==========================================================================
    value_min: Annotated[
        float | None,
        Field(default=0, description="Minimum value axis limit. None for auto."),
    ]
    value_max: Annotated[
        float | None,
        Field(default=None, description="Maximum value axis limit. None for auto."),
    ]
    value_padding_pct: Annotated[
        float,
        Field(default=0.15, description="Padding as fraction of max value for auto limit."),
    ]
    xlim: Annotated[
        tuple[float, float] | None,
        Field(
            default=None,
            description="Explicit (min, max) for x-axis. Overrides value_min/max for horizontal.",
        ),
    ]
    ylim: Annotated[
        tuple[float, float] | None,
        Field(
            default=None,
            description="Explicit (min, max) for y-axis. Overrides value_min/max for vertical.",
        ),
    ]
    xticks: Annotated[
        list[float] | None,
        Field(default=None, description="Custom x-tick positions. None uses auto ticks."),
    ]
    yticks: Annotated[
        list[float] | None,
        Field(default=None, description="Custom y-tick positions. None uses auto ticks."),
    ]
    xtick_labels: Annotated[
        list[str] | None,
        Field(default=None, description="Custom x-tick labels. Must match xticks length."),
    ]
    ytick_labels: Annotated[
        list[str] | None,
        Field(default=None, description="Custom y-tick labels. Must match yticks length."),
    ]

    # ==========================================================================
    # Category axis settings
    # ==========================================================================
    invert_categories: Annotated[
        bool,
        Field(
            default=True, description="Invert category axis (first category at top for horizontal)."
        ),
    ]

    model_config = {"extra": "forbid"}
