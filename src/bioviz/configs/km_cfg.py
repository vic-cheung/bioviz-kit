"""Configuration for Kaplan-Meier survival plots.

This provides a pydantic-based configuration model for KM plots, mirroring
the keyword arguments of the KM plotting API. Font sizes default to None
to inherit from matplotlib rcParams.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

LegendLoc = Literal["bottom", "right", "inside"]
PvalLoc = Literal[
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center_right",
]
CIStyle = Literal["fill", "lines"]
ConfType = Literal["log_log", "linear"]


class KMPlotConfig(BaseModel):
    """Configuration for Kaplan-Meier plot generation.

    Notes
    -----
    - Field names match the KM plotting kwargs where possible.
    - time_col, event_col, and group_col describe data schema; they are required.
    - color_dict maps group labels to colors; optional.
    - You can override xticks directly or provide xtick_interval_months to auto-build ticks.
    - Font sizes default to None to inherit from rcParams (set via RVMDStyle.apply_theme()).
    """

    # ==========================================================================
    # Required dataset columns
    # ==========================================================================
    time_col: Annotated[
        str,
        Field(..., description="Column containing times (e.g., months)"),
    ]
    event_col: Annotated[
        str,
        Field(..., description="Column containing event indicator (1/0 or True/False)"),
    ]
    group_col: Annotated[
        str,
        Field(..., description="Column for grouping/stratification"),
    ]

    # ==========================================================================
    # Labels and axis limits
    # ==========================================================================
    title: Annotated[
        str | None,
        Field(default=None, description="Plot title"),
    ]
    xlim: Annotated[
        Tuple[float | None, float | None] | None,
        Field(default=None, description="X-axis limits (min, max)"),
    ]
    ylim: Annotated[
        Tuple[float, float],
        Field(default=(0.0, 1.05), description="Y-axis limits (min, max)"),
    ]
    xlabel: Annotated[
        str,
        Field(default="Time (Months)", description="X-axis label"),
    ]
    ylabel: Annotated[
        str,
        Field(default="Survival Probability", description="Y-axis label"),
    ]

    # ==========================================================================
    # Figure/layout
    # ==========================================================================
    figsize: Annotated[
        Tuple[float, float] | None,
        Field(
            default=None,
            description="Figure size as (width, height) tuple. Takes precedence over fig_width/fig_height.",
        ),
    ]
    fig_width: Annotated[
        float,
        Field(
            default=10.0,
            ge=1.0,
            description="Figure width in inches (used if figsize is None)",
        ),
    ]
    fig_height: Annotated[
        float,
        Field(
            default=6.0,
            ge=1.0,
            description="Figure height in inches for KM panel (used if figsize is None)",
        ),
    ]

    def get_figsize(self) -> Tuple[float, float]:
        """Return effective figsize, preferring figsize over fig_width/fig_height."""
        if self.figsize is not None:
            return self.figsize
        return (self.fig_width, self.fig_height)

    # ==========================================================================
    # Legend options
    # ==========================================================================
    legend_loc: Annotated[
        LegendLoc,
        Field(default="bottom", description="Legend location: 'bottom', 'right', or 'inside'"),
    ]
    legend_title: Annotated[
        str | None,
        Field(default=None, description="Legend title"),
    ]
    legend_title_fontweight: Annotated[
        str | None,
        Field(default="bold", description="Legend title font weight"),
    ]
    legend_fontsize: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Legend font size. None uses rcParams['legend.fontsize'].",
        ),
    ]
    legend_frameon: Annotated[
        bool,
        Field(default=False, description="Whether to draw a frame around the legend"),
    ]
    legend_markerscale: Annotated[
        float,
        Field(default=1.0, gt=0.0, description="Scale factor for legend marker sizes"),
    ]
    legend_linewidth_scale: Annotated[
        float | None,
        Field(default=None, description="Multiply legend line widths by this factor"),
    ]
    legend_show_n: Annotated[
        bool,
        Field(default=False, description="Whether to show (n=...) in legend labels"),
    ]
    legend_label_wrap_chars: Annotated[
        int | None,
        Field(default=None, description="Wrap legend labels at this many characters"),
    ]
    legend_label_max_lines: Annotated[
        int,
        Field(default=2, ge=1, description="Maximum lines for wrapped legend labels"),
    ]
    legend_label_overrides: Annotated[
        Dict[Any, str] | None,
        Field(default=None, description="Override labels: {group_value: 'Display Label'}"),
    ]
    auto_expand_for_legend: Annotated[
        bool,
        Field(default=False, description="Auto-expand figure to fit legend"),
    ]

    # ==========================================================================
    # P-value options
    # ==========================================================================
    show_pvalue: Annotated[
        bool,
        Field(default=True, description="Whether to show p-value annotation"),
    ]
    pval_loc: Annotated[
        PvalLoc,
        Field(default="top_right", description="P-value annotation location"),
    ]
    pvalue_fontsize: Annotated[
        int | None,
        Field(
            default=None, ge=1, description="P-value font size. None uses rcParams['font.size']."
        ),
    ]
    pvalue_box: Annotated[
        bool,
        Field(default=False, description="Whether to draw a box around p-value"),
    ]

    # ==========================================================================
    # Curve styling
    # ==========================================================================
    show_ci: Annotated[
        bool,
        Field(default=True, description="Whether to show confidence intervals"),
    ]
    ci_style: Annotated[
        CIStyle,
        Field(default="fill", description="CI style: 'fill' or 'lines'"),
    ]
    ci_alpha: Annotated[
        float,
        Field(default=0.25, ge=0.0, le=1.0, description="CI fill transparency"),
    ]
    linewidth: Annotated[
        float,
        Field(default=3.0, ge=0.0, description="Survival curve line width"),
    ]
    linestyle: Annotated[
        str,
        Field(default="-", description="Survival curve line style"),
    ]
    conf_type: Annotated[
        ConfType,
        Field(default="log_log", description="Confidence interval type: 'log_log' or 'linear'"),
    ]

    # ==========================================================================
    # Censor marker options
    # ==========================================================================
    censor_marker: Annotated[
        str,
        Field(default="+", description="Marker style for censored points"),
    ]
    censor_markersize: Annotated[
        float,
        Field(default=12.0, ge=0.0, description="Size of censor markers"),
    ]
    censor_markeredgewidth: Annotated[
        float,
        Field(default=2.5, ge=0.0, description="Edge width of censor markers"),
    ]
    force_show_censors: Annotated[
        bool,
        Field(default=True, description="Force show censors even if none detected"),
    ]
    per_patient_censor_markers: Annotated[
        bool,
        Field(default=True, description="Show individual censor markers per patient"),
    ]

    # ==========================================================================
    # Risk table options
    # ==========================================================================
    show_risktable: Annotated[
        bool,
        Field(default=True, description="Whether to show risk table below plot"),
    ]
    risktable_fontsize: Annotated[
        int | None,
        Field(
            default=None, ge=1, description="Risk table font size. None uses rcParams['font.size']."
        ),
    ]
    risktable_title_fontsize: Annotated[
        int | None,
        Field(
            default=None,
            description="Risk table title font size. None uses risktable_fontsize + 2.",
        ),
    ]
    risktable_row_spacing: Annotated[
        float,
        Field(
            default=1.8, ge=0.5, description="Vertical spacing multiplier between risk table rows"
        ),
    ]
    risktable_title_gap_factor: Annotated[
        float,
        Field(default=0.6, ge=0.0, description="Extra top padding between title and first row"),
    ]
    risktable_hspace: Annotated[
        float,
        Field(default=0.5, ge=0.0, description="Space between KM plot and risk table"),
    ]
    risktable_min_rows: Annotated[
        int,
        Field(default=4, ge=1, description="Minimum risk table rows to reserve for layout"),
    ]
    color_risktable_counts: Annotated[
        bool,
        Field(default=False, description="Whether to color risk table counts by group"),
    ]
    risktable_label_wrap_chars: Annotated[
        int | None,
        Field(default=None, description="Wrap risk table labels at this many characters"),
    ]
    risktable_label_max_lines: Annotated[
        int,
        Field(default=2, ge=1, description="Maximum lines for wrapped risk table labels"),
    ]
    risktable_label_overrides: Annotated[
        Dict[Any, str] | None,
        Field(default=None, description="Override risk table labels: {group_value: 'Label'}"),
    ]

    # ==========================================================================
    # Ticks/timeline
    # ==========================================================================
    xticks: Annotated[
        List[float] | None,
        Field(default=None, description="Explicit x-tick positions"),
    ]
    timeline: Annotated[
        Iterable[float] | None,
        Field(default=None, description="Timeline values for risk table"),
    ]
    xtick_interval_months: Annotated[
        float | None,
        Field(default=3.0, gt=0, description="Interval for auto-generated x-ticks"),
    ]

    # ==========================================================================
    # Group ordering and colors
    # ==========================================================================
    group_order: Annotated[
        List[Any] | None,
        Field(
            default=None,
            description="Explicit order of groups for plotting and legend. "
            "If None, uses pd.Categorical order if set, else data order.",
        ),
    ]
    color_dict: Annotated[
        Dict[Any, str] | None,
        Field(default=None, description="Mapping of group values to colors"),
    ]

    # ==========================================================================
    # Font sizes (None = use rcParams)
    # ==========================================================================
    label_fontsize: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Axis label font size. None uses rcParams['axes.labelsize'].",
        ),
    ]
    title_fontsize: Annotated[
        int | None,
        Field(
            default=None, ge=1, description="Title font size. None uses rcParams['axes.titlesize']."
        ),
    ]
    title_fontweight: Annotated[
        str,
        Field(
            default="bold", description="Font weight for title ('normal', 'bold', 'light', etc.)."
        ),
    ]

    # ==========================================================================
    # Save options
    # ==========================================================================
    save_bbox_inches: Annotated[
        str | None,
        Field(default="tight", description="bbox_inches argument for savefig"),
    ]

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("ylim")
    @classmethod
    def _check_ylim(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        y0, y1 = v
        if y0 >= y1:
            raise ValueError("ylim must be (min, max) with min < max")
        return v

    @field_validator("color_dict", mode="before")
    @classmethod
    def _coerce_color_dict(cls, v):
        """Allow any mapping for color_dict keys; enforce mapping type."""
        if v is None:
            return v
        if isinstance(v, Mapping):
            return dict(v)
        raise TypeError("color_dict must be a mapping of group -> color string")

    @field_validator("color_dict")
    @classmethod
    def _check_color_values(cls, v: Optional[Dict[Any, str]]) -> Optional[Dict[Any, str]]:
        if v is None:
            return v
        for _, color in v.items():
            if not isinstance(color, str):
                raise ValueError("color_dict values must be strings (e.g., hex colors)")
        return v

    class Config:
        """Pydantic config."""

        extra = "forbid"


__all__ = ["KMPlotConfig"]
