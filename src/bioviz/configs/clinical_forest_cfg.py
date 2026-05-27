"""Configuration for Clinical Forest plots.

Clinical forest plots are enhanced forest plots with additional table columns
for events/patients, reference/comparator labels, median survival times,
and formatted p-values. They are commonly used in oncology trial publications.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

MarkerStyle = Literal["s", "o", "D", "^", "v", "<", ">", "p", "*", "h"]


class ClinicalForestPlotConfig(BaseModel):
    """Configuration for clinical forest plot generation.

    This config provides column mappings for a table-style forest plot with
    events/patients counts, HR with CI, median survival times, and p-values.

    Notes
    -----
    - All column names have sensible defaults matching common survival analysis output.
    - Set a column to None to hide that table section.
    - Font sizes default to None to inherit from matplotlib rcParams.
    """

    # ==========================================================================
    # Required column mapping (data schema)
    # ==========================================================================
    label_col: Annotated[
        str,
        Field(default="display_label", description="Column name for row labels"),
    ]
    hr_col: Annotated[
        str,
        Field(default="hr", description="Column name for hazard ratio values"),
    ]
    ci_lower_col: Annotated[
        str,
        Field(default="hr_ci_lower", description="Column name for lower CI bound"),
    ]
    ci_upper_col: Annotated[
        str,
        Field(default="hr_ci_upper", description="Column name for upper CI bound"),
    ]

    # ==========================================================================
    # Reference arm columns (Events/Patients left table)
    # ==========================================================================
    reference_label_col: Annotated[
        str | None,
        Field(default="reference", description="Column for reference arm label"),
    ]
    reference_events_col: Annotated[
        str | None,
        Field(default="events_per_reference_group", description="Column for reference events"),
    ]
    reference_n_col: Annotated[
        str | None,
        Field(default="n_per_reference_group", description="Column for reference N"),
    ]

    # ==========================================================================
    # Comparator arm columns (Events/Patients left table)
    # ==========================================================================
    comparator_label_col: Annotated[
        str | None,
        Field(default="comparator", description="Column for comparator arm label"),
    ]
    comparator_events_col: Annotated[
        str | None,
        Field(default="events_per_comparator_group", description="Column for comparator events"),
    ]
    comparator_n_col: Annotated[
        str | None,
        Field(default="n_per_comparator_group", description="Column for comparator N"),
    ]

    # ==========================================================================
    # Median survival columns (right table)
    # ==========================================================================
    median_ref_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref", description="Column for reference median"),
    ]
    median_ref_ci_lower_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref_ci_lower", description="Reference median CI lower"),
    ]
    median_ref_ci_upper_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref_ci_upper", description="Reference median CI upper"),
    ]
    median_ref_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref_not_reached", description="Reference median NR flag"),
    ]
    median_ref_ci_lower_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref_ci_lower_not_reached", description="Ref CI lower NR"),
    ]
    median_ref_ci_upper_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_ref_ci_upper_not_reached", description="Ref CI upper NR"),
    ]

    median_cmp_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp", description="Column for comparator median"),
    ]
    median_cmp_ci_lower_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp_ci_lower", description="Comparator median CI lower"),
    ]
    median_cmp_ci_upper_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp_ci_upper", description="Comparator median CI upper"),
    ]
    median_cmp_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp_not_reached", description="Comparator median NR flag"),
    ]
    median_cmp_ci_lower_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp_ci_lower_not_reached", description="Cmp CI lower NR"),
    ]
    median_cmp_ci_upper_not_reached_col: Annotated[
        str | None,
        Field(default="median_surv_time_cmp_ci_upper_not_reached", description="Cmp CI upper NR"),
    ]

    # ==========================================================================
    # P-value column
    # ==========================================================================
    pvalue_col: Annotated[
        str | None,
        Field(default="p_value_wald", description="Column name for p-values"),
    ]

    # ==========================================================================
    # Figure layout
    # ==========================================================================
    title: Annotated[
        str | None,
        Field(default=None, description="Plot title (supports newlines)"),
    ]
    title_wrap_width: Annotated[
        int | None,
        Field(default=None, description="Max chars before title wraps. None = auto."),
    ]
    figsize: Annotated[
        tuple[float, float] | None,
        Field(default=None, description="Figure size (width, height). None = auto."),
    ]
    figure_width: Annotated[
        float,
        Field(default=13.4, description="Figure width in inches"),
    ]
    min_figure_height: Annotated[
        float,
        Field(default=4.0, description="Minimum figure height in inches"),
    ]
    max_figure_height: Annotated[
        float,
        Field(default=24.0, description="Maximum figure height in inches"),
    ]
    row_height: Annotated[
        float,
        Field(default=0.42, description="Base height per row in inches"),
    ]
    base_height: Annotated[
        float,
        Field(default=3.6, description="Base figure height before row scaling"),
    ]
    scale_vertical_positions_on_tall_figures: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "When True, scale title/header/footer vertical positions toward their anchors "
                "on very tall figures to avoid oversized whitespace. Set False to use exact "
                "manual vertical positions."
            ),
        ),
    ]
    scale_title_position_on_tall_figures: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Optional override for title vertical scaling on tall figures. "
                "None inherits scale_vertical_positions_on_tall_figures."
            ),
        ),
    ]
    scale_header_footer_positions_on_tall_figures: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Optional override for header and footer vertical scaling on tall figures. "
                "None inherits scale_vertical_positions_on_tall_figures."
            ),
        ),
    ]

    # ==========================================================================
    # Layout positioning (axes fraction coordinates)
    # ==========================================================================
    # Row label position (left side)
    label_x_position: Annotated[
        float,
        Field(default=-0.90, description="X position for row labels (negative = left of plot)"),
    ]
    # Events/Patients columns (left table)
    reference_x_position: Annotated[
        float,
        Field(default=-0.34, description="X position for reference Events/Patients column"),
    ]
    comparator_x_position: Annotated[
        float,
        Field(default=-0.12, description="X position for comparator Events/Patients column"),
    ]
    # Right-side table columns (HR, Medians, p-value)
    hr_x_position: Annotated[
        float,
        Field(default=1.02, description="X position for HR column (>1.0 = right of plot)"),
    ]
    median_ref_x_position: Annotated[
        float,
        Field(default=1.26, description="X position for Median Ref column"),
    ]
    median_cmp_x_position: Annotated[
        float,
        Field(default=1.53, description="X position for Median Cmp column"),
    ]
    pvalue_x_position: Annotated[
        float,
        Field(default=1.80, description="X position for p-value column"),
    ]
    # Header rule (horizontal line below headers)
    header_rule_end_x: Annotated[
        float,
        Field(default=1.92, description="X position where header rule line ends"),
    ]
    # Header vertical positions
    header_top_y: Annotated[
        float,
        Field(default=1.02, description="Y position for top header row"),
    ]
    header_sub_y: Annotated[
        float,
        Field(default=0.97, description="Y position for sub-header row"),
    ]
    # Title position (None = auto)
    title_y_position: Annotated[
        float | None,
        Field(default=None, description="Y position for title (None = auto-calculated)"),
    ]

    # ==========================================================================
    # Table section visibility
    # ==========================================================================
    show_events_patients: Annotated[
        bool,
        Field(default=True, description="Show Events/Patients table section"),
    ]
    show_hr_column: Annotated[
        bool,
        Field(default=True, description="Show HR (95% CI) column"),
    ]
    show_median_columns: Annotated[
        bool,
        Field(default=True, description="Show Median Ref/Cmp columns"),
    ]
    show_pvalue_column: Annotated[
        bool,
        Field(default=True, description="Show p-value column"),
    ]

    # ==========================================================================
    # Footer labels
    # ==========================================================================
    left_footer_label: Annotated[
        str,
        Field(default="Comparator Better", description="Label below left arrow"),
    ]
    right_footer_label: Annotated[
        str,
        Field(default="Reference Better", description="Label below right arrow"),
    ]
    xlabel: Annotated[
        str,
        Field(default="Hazard Ratio (95% CI)", description="X-axis label"),
    ]

    # ==========================================================================
    # X-axis behavior
    # ==========================================================================
    xlim: Annotated[
        tuple[float, float] | None,
        Field(default=None, description="X-axis limits. None = auto bounded."),
    ]
    xticks: Annotated[
        list[float] | None,
        Field(default=None, description="Custom x-tick positions. None = auto."),
    ]
    x_max_cap: Annotated[
        float,
        Field(default=6.0, description="Upper cap for auto x-axis limit"),
    ]
    show_truncation_markers: Annotated[
        bool,
        Field(default=True, description="Show truncation markers for clipped CIs"),
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
        Field(default="gray", description="Reference line color"),
    ]
    reference_line_style: Annotated[
        str,
        Field(default="-", description="Reference line style"),
    ]
    reference_line_width: Annotated[
        float,
        Field(default=1.0, ge=0.0, description="Reference line width"),
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
        Field(default="o", description="Marker style"),
    ]
    marker_color: Annotated[
        str,
        Field(default="black", description="Marker and CI line color"),
    ]
    linewidth: Annotated[
        float,
        Field(default=2.0, ge=0.0, description="Error bar line width"),
    ]
    show_caps: Annotated[
        bool,
        Field(default=True, description="Show caps on error bars"),
    ]
    capsize: Annotated[
        float | None,
        Field(default=None, description="Cap size. None = auto by row count."),
    ]

    # ==========================================================================
    # Font sizes (None = auto)
    # ==========================================================================
    title_fontsize: Annotated[
        float | None,
        Field(default=None, description="Title font size. None = auto."),
    ]
    label_fontsize: Annotated[
        float,
        Field(default=9.0, description="Row label font size"),
    ]
    header_fontsize: Annotated[
        float,
        Field(default=10.0, description="Table header font size"),
    ]
    cell_fontsize: Annotated[
        float,
        Field(default=8.5, description="Table cell font size"),
    ]
    axis_fontsize: Annotated[
        float,
        Field(default=10.0, description="X-axis tick font size"),
    ]
    xlabel_fontsize: Annotated[
        float,
        Field(default=11.0, description="X-axis label font size"),
    ]
    footer_fontsize: Annotated[
        float,
        Field(default=9.0, description="Footer label font size"),
    ]

    # Footer vertical positioning (negative values move down)
    footer_xlabel_offset: Annotated[
        float,
        Field(default=-0.06, description="Vertical offset for x-axis label (in axes fraction)"),
    ]
    footer_arrow_offset: Annotated[
        float,
        Field(
            default=-0.12, description="Vertical offset for directional arrows (in axes fraction)"
        ),
    ]
    footer_text_offset: Annotated[
        float,
        Field(
            default=-0.165, description="Vertical offset for footer text labels (in axes fraction)"
        ),
    ]

    model_config = {"extra": "forbid"}


__all__ = ["ClinicalForestPlotConfig"]
