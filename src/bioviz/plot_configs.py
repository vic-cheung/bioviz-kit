"""
Utility functions and Pydantic configuration models for plotting.

These are copied from the `tm_toolbox` implementation to provide a
feature-complete configuration layer in `bioviz`.
"""

from typing import Annotated, Any, Tuple, Union

import pandas as pd  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore

# Expose all public classes and functions
__all__ = [
    "BasePlotConfig",
    "StyledLinePlotConfig",
    "StyledSpiderPlotConfig",
    "ScanOverlayPlotConfig",
    "StyledTableConfig",
    "TopAnnotationConfig",
    "HeatmapAnnotationConfig",
    "make_annotation_config",
    "OncoplotConfig",
]


class BasePlotConfig(BaseModel):
    palette: Annotated[
        list[Any] | dict[Any, Any] | None,
        Field(
            default=None,
            description=(
                "Color palette to use for the plot. Can be a list or a dictionary. "
                "If None, a default palette is used."
            ),
        ),
    ]
    lw: Annotated[float, Field(default=4.0, description="Line width for plotted lines.")]
    markersize: Annotated[
        float | None,
        Field(
            default=None,
            description="Size of markers on the plot. Defaults to twice the line width if None.",
        ),
    ]
    figsize: Annotated[
        tuple[int, int],
        Field(
            default=(9, 6),
            description="Figure size as (width, height) in inches.",
        ),
    ]
    legend_loc: Annotated[
        str,
        Field(
            default="upper left",
            description="Location string for the legend placement.",
        ),
    ]

    col_vals_to_include_in_title: Annotated[
        list[str],
        Field(
            default_factory=lambda: [
                "Patient_ID",
                "Plot_Indication",
                "BOR_con",
                "RAS_Mutation",
            ],
            description=(
                "Columns whose values are included in the plot title if no custom "
                "title is provided."
            ),
        ),
    ]
    title: Annotated[str | None, Field(default=None, description="Custom title for the plot.")]

    rhs_pdf_padding: Annotated[
        float,
        Field(
            default=0.8,
            description="Padding on the right side to avoid clipping in PDF exports.",
        ),
    ]

    @field_validator("palette")
    @classmethod
    def validate_palette(
        cls, v: list[Any] | dict[Any, Any] | None
    ) -> list[Any] | dict[Any, Any] | None:
        if v is not None and not isinstance(v, (list, dict)):
            raise ValueError("Palette must be a list, a dict, or None.")
        return v

    @model_validator(mode="after")
    def set_default_markersize(cls, values: "BasePlotConfig") -> "BasePlotConfig":
        if values.markersize is None:
            values.markersize = values.lw * 2
        return values


class StyledLinePlotConfig(BasePlotConfig):
    patient_id: Annotated[str | int, Field()]
    label_col: Annotated[
        str, Field(default="label", description="Column name for mutation/gene labels.")
    ]
    subset_cols: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["Patient_ID", "label", "Timepoint"],
            description="Columns to use for filtering duplicates.",
        ),
    ]
    filter_dict: Annotated[
        dict[str, list[str]] | None,
        Field(
            default=None,
            description="Filters to apply to the DataFrame before plotting.",
        ),
    ]
    x: Annotated[
        str,
        Field(default="Timepoint", description="Column name to use for the x-axis."),
    ]
    xlabel: Annotated[
        str | None,
        Field(default=None, description="X axis label"),
    ]
    y: Annotated[str, Field(default="Value", description="Column name to use for the y-axis.")]
    ylabel: Annotated[
        str | None,
        Field(default=None, description="Y axis label"),
    ]
    match_legend_text_color: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to color legend text to match line colors.",
        ),
    ]
    label_points: Annotated[
        bool, Field(default=True, description="Whether to label the first point in each line.")
    ]
    xlim_padding: Annotated[
        float, Field(default=0.8, description="Padding to apply to x-axis limits.")
    ]
    ylim: Annotated[
        tuple[float | None, float | None],
        Field(default=(0, None), description="Limits for the y-axis."),
    ]
    add_extra_tick_to_ylim: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to add extra space above the y-axis max tick.",
        ),
    ]
    ymin_padding_fraction: Annotated[
        float,
        Field(
            default=0.05,
            ge=0.0,
            le=1.0,
            description="Fraction of y-range to use as padding on both ends of the y-axis.",
        ),
    ]
    threshold: Annotated[
        float | int | None,
        Field(
            default=None,
            description=(
                "Y-axis threshold below which markers are hollow with colored edges. "
                "Also adds a dotted threshold line."
            ),
        ),
    ]
    filled_marker_scale: Annotated[
        float,
        Field(
            default=1.2,
            ge=1.0,
            description="Scale factor for filled markers to visually match unfilled ones.",
        ),
    ]


class StyledSpiderPlotConfig(BasePlotConfig):
    group_col: Annotated[str, Field(description="Column name to group lines by.")]
    x: Annotated[str, Field(description="Usually the timepoint column name for the x-axis.")]
    y: Annotated[str, Field(description="Column containing values to plot.")]
    xlabel: Annotated[
        str | None,
        Field(default=None, description="Label for the x-axis. Defaults to `x` if None."),
    ]
    ylabel: Annotated[
        str | None,
        Field(default=None, description="Label for the y-axis. Defaults to `y` if None."),
    ]
    subgroup_name: Annotated[
        str | None,
        Field(default=None, description="Optional name of the subgroup being plotted."),
    ]
    marker_style: Annotated[
        str,
        Field(
            default="o",
            description="Marker style string (e.g., 'o', 's', '^') for plot points.",
        ),
    ]
    color_dict_subgroup: Annotated[
        dict[Any, Any] | None,
        Field(
            default=None,
            description=(
                "Optional dictionary mapping subgroup labels to colors. Overrides patient coloring."
            ),
        ),
    ]
    linestyle_dict: Annotated[
        dict[Any, str] | None,
        Field(
            default=None,
            description=(
                "Optional dictionary mapping categorical values to matplotlib line styles "
                "(e.g., '-', '--')."
            ),
        ),
    ]
    markerstyle_dict: Annotated[
        dict[Any, str] | None,
        Field(
            default=None,
            description=(
                "Optional dictionary mapping categorical values to matplotlib marker "
                "styles (e.g., 'o', '^')."
            ),
        ),
    ]
    linestyle_col: Annotated[
        str | None, Field(default=None, description="Column name to determine line styles.")
    ]
    markerstyle_col: Annotated[
        str | None, Field(default=None, description="Column name to determine marker styles.")
    ]
    use_absolute_scale: Annotated[
        bool,
        Field(
            default=False,
            description="If True, use 0-max scale instead of negative-positive relative scale.",
        ),
    ]
    absolute_ylim: Annotated[
        tuple[float, float] | None,
        Field(
            default=None,
            description=(
                "Y-axis limits when use_absolute_scale=True. If None, uses (-5, 105) for clonality."
            ),
        ),
    ]
    absolute_yticks: Annotated[
        list[float] | None,
        Field(
            default=None,
            description=(
                "Y-axis tick positions when use_absolute_scale=True. "
                "If None, uses [0, 25, 50, 75, 100] for clonality."
            ),
        ),
    ]


class ScanOverlayPlotConfig(BasePlotConfig):
    x: Annotated[str, Field(description="X-axis column for scan data.")]
    y: Annotated[str, Field(description="Y-axis column for scan data.")]
    hue_col: Annotated[str, Field(description="Column for color grouping (e.g., Location).")]
    recist_col: Annotated[str, Field(description="Column for where RECISTS are stored. e.g., BOR")]
    palette: Annotated[
        dict | list | None, Field(default=None, description="Palette for scan overlay.")
    ]
    linestyle: Annotated[str, Field(default=":", description="Line style for scan overlay.")]
    alpha: Annotated[float, Field(default=0.5, description="Alpha for scan overlay lines.")]


class StyledTableConfig(BaseModel):
    title: Annotated[str, Field(default="", description="Title for the table.")]
    title_font_size: Annotated[
        float | int, Field(default=20, description="Font size for title text.")
    ]
    header_bg_color: Annotated[
        str,
        Field(default="#1E6B5C", description="Background color for the header row."),
    ]
    header_text_color: Annotated[
        str,
        Field(default="white", description="Text color for the header row."),
    ]
    row_colors: Annotated[
        tuple[str, str],
        Field(
            default=("#f2f2f2", "gainsboro"),
            description="Colors for alternating table rows.",
        ),
    ]
    edge_color: Annotated[str, Field(default="white", description="Color of cell edges.")]
    header_font_size: Annotated[
        float | int, Field(default=16, description="Font size for header text.")
    ]
    header_font_weight: Annotated[
        str, Field(default="bold", description="Font weight for header text.")
    ]
    cell_font_size: Annotated[
        float | int, Field(default=16, description="Font size for all text in the table.")
    ]
    max_chars: Annotated[
        int,
        Field(
            default=18,
            description="Max characters per cell before shrinking font size.",
        ),
    ]
    shrink_by: Annotated[
        float | int,
        Field(
            default=2,
            description="Points to reduce font size by when text exceeds `max_chars`.",
        ),
    ]
    row_height: Annotated[
        float, Field(default=0.06, description="Relative height of each row (0 to 1 scale).")
    ]
    header_row_height: Annotated[
        float | None,
        Field(
            default=None,
            description="Relative height of header row. If None, uses row_height value.",
        ),
    ]
    row_height_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            description="Height multiplier per line for dynamic row scaling. ",
        ),
    ]
    table_width: Annotated[
        float, Field(default=2.5, description="Relative width of the table (0 to 1 scale).")
    ]
    table_scale: Annotated[
        tuple[float, float],
        Field(default=(12, 12), description="Scaling factors for table (width, height)."),
    ]
    absolute_font_size: Annotated[
        bool,
        Field(
            default=False,
            description="If True, font sizes are applied after table scaling and remain absolute.",
        ),
    ]
    header_font_family: Annotated[
        str | None, Field(default=None, description="Font family for header text.")
    ]
    body_font_family: Annotated[
        str | None, Field(default=None, description="Font family for body text.")
    ]
    body_font_weight: Annotated[
        str, Field(default="normal", description="Font weight for body text.")
    ]


class TopAnnotationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: Annotated[
        pd.Series | dict[Any, Any],
        Field(description="Annotation values for each patient (indexed by patient ID)."),
    ]
    colors: Annotated[
        dict[
            str, str | tuple[float, float, float] | tuple[float, float, float, float] | float | int
        ],
        Field(
            description=(
                "Mapping from annotation values to matplotlib colors "
                "(named colors, hex strings, RGB/RGBA tuples, or grayscale values)."
            )
        ),
    ]
    height: Annotated[float, Field(default=1, description="Height of the annotation bar.")]
    fontsize: Annotated[
        float | int, Field(default=16, description="Font size for annotation labels.")
    ]
    display_name: Annotated[
        str | None,
        Field(default=None, description="Display name for the annotation (used as label)."),
    ]
    legend_title: Annotated[
        str | None,
        Field(default=None, description="Title for the legend entry for this annotation."),
    ]
    legend_value_order: Annotated[
        list[str] | None,
        Field(default=None, description="Order of values in the legend."),
    ]
    show_category_labels: Annotated[
        bool,
        Field(default=False, description="Whether to show one label per category block."),
    ]
    merge_labels: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to merge contiguous blocks with the same value into one label.",
        ),
    ]
    label_fontsize: Annotated[
        float | int, Field(default=14, description="Font size for merged/centered labels.")
    ]
    label_text_colors: Annotated[
        dict[
            str, str | tuple[float, float, float] | tuple[float, float, float, float] | float | int
        ]
        | None,
        Field(
            default=None,
            description=(
                "Mapping from annotation values to matplotlib text colors "
                "(named colors, hex strings, RGB/RGBA tuples, or grayscale values)."
            ),
        ),
    ]
    na_color: Annotated[
        str,
        Field(
            default="gainsboro",
            description="Color to use for missing (NA) values in the annotation bar and legend.",
        ),
    ]
    draw_border: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether to draw a border around all annotation blocks. "
                "If False and border_categories is None, borders are only drawn around white/light colors. "
            ),
        ),
    ]
    border_categories: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "List of specific category values that should have borders. "
                "If specified, only these categories will have borders (overrides draw_border). "
            ),
        ),
    ]
    border_color: Annotated[
        str,
        Field(default="black", description="Color for the border around annotation blocks."),
    ]
    border_width: Annotated[
        float,
        Field(default=0.5, description="Line width for the border around annotation blocks."),
    ]


class HeatmapAnnotationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: Annotated[
        str | pd.Series,
        Field(
            description=(
                "Either the column name in the DataFrame or a Series of values for the heatmap."
            )
        ),
    ]
    colors: Annotated[
        dict[
            str, str | tuple[float, float, float] | tuple[float, float, float, float] | float | int
        ],
        Field(
            description=(
                "Mapping from mutation types or values to matplotlib colors "
                "(named colors, hex strings, RGB/RGBA tuples, or grayscale values)."
            )
        ),
    ]
    legend_title: Annotated[
        str | None,
        Field(default=None, description="Title for the legend entry for this annotation."),
    ]
    legend_value_order: Annotated[
        list[str] | None, Field(default=None, description="Order of values in the legend.")
    ]
    draw_border: Annotated[
        bool,
        Field(
            default=False,
            description=("Whether to draw a border around all legend patches. "),
        ),
    ]
    border_categories: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=("List of specific category values that should have borders. "),
        ),
    ]
    border_color: Annotated[
        str,
        Field(default="black", description="Color for the border around legend patches."),
    ]
    border_width: Annotated[
        float,
        Field(default=0.5, description="Line width for the border around legend patches."),
    ]
    bottom_left_triangle_values: Annotated[
        list[str],
        Field(
            default=["SV"],
            description=(
                "List of mutation type values that should be drawn as bottom-left triangles "
                "(diagonal fill). Default is ['SV'] for structural variants. "
            ),
        ),
    ]
    upper_right_triangle_values: Annotated[
        list[str],
        Field(
            default=["CNV"],
            description=(
                "List of mutation type values that should be drawn as upper-right triangles (diagonal fill). "
            ),
        ),
    ]


def make_annotation_config(
    values: pd.Series | dict[Any, Any],
    colors: dict[
        str, str | tuple[float, float, float] | tuple[float, float, float, float] | float | int
    ],
    display_name: str,
    legend_title: str,
    **kwargs: Any,
) -> TopAnnotationConfig:
    return TopAnnotationConfig(
        values=values,
        colors=colors,
        height=0.8,
        fontsize=16,
        display_name=display_name,
        legend_title=legend_title,
        **kwargs,
    )


class OncoplotConfig(BaseModel):
    fig_title: Annotated[
        str | None, Field(default=None, description="Figure title for the oncoplot.")
    ]
    fig_title_fontsize: Annotated[
        float | int, Field(default=22, description="Font size for the figure title.")
    ]
    fig_top_margin: Annotated[
        float,
        Field(
            default=0.9,
            description=(
                "Fraction of the figure height to reserve for the top margin (for the title). "
            ),
        ),
    ]
    fig_bottom_margin: Annotated[
        float,
        Field(
            default=0.18,
            description=("Fraction of the figure height to reserve for the bottom margin "),
        ),
    ]
    col_split_by: Annotated[
        list[str],
        Field(default_factory=list, description="Columns to split the columns by."),
    ]
    col_split_order: Annotated[
        dict[str, list[str]],
        Field(default_factory=dict, description="Order of categories for each column split."),
    ]
    col_sort_by: Annotated[
        list[str], Field(default_factory=list, description="Columns to sort by.")
    ]

    figsize: Annotated[
        tuple[float, float],
        Field(default=(12, 8), description="Figure size as (width, height)."),
    ]
    bar_width: Annotated[float, Field(default=0.12, description="Width of the pathway bars.")]
    row_group_label_fontsize: Annotated[
        float | int, Field(default=18, description="Font size for pathway group labels.")
    ]
    row_label_fontsize: Annotated[
        float | int, Field(default=16, description="Font size for gene labels.")
    ]
    column_label_fontsize: Annotated[
        float | int, Field(default=16, description="Font size for patient labels.")
    ]
    legend_fontsize: Annotated[
        float | int, Field(default=16, description="Font size for legend text.")
    ]
    legend_title_fontsize: Annotated[
        float | int, Field(default=16, description="Font size for legend title.")
    ]
    rotate_left_annotation_label: Annotated[
        bool, Field(default=False, description="Whether to rotate the left annotation labels.")
    ]

    top_annotations: Annotated[
        dict[str, TopAnnotationConfig] | None,
        Field(default=None, description="Dictionary of top annotations."),
    ]
    top_annotation_order: Annotated[
        list[str] | None,
        Field(default=None, description="Order to display top annotations (from top to bottom)."),
    ]
    top_annotation_inter_spacer: Annotated[
        float, Field(default=1.3, description="Spacing between top annotation groups.")
    ]
    top_annotation_intra_spacer: Annotated[
        float, Field(default=0.2, description="Spacing within top annotation groups.")
    ]

    x_col: Annotated[
        str, Field(default="Patient_ID", description="Column name for x-axis (patient IDs).")
    ]
    y_col: Annotated[
        str,
        Field(
            default="Gene_Mutation",
            description="Column name for y-axis (gene mutations).",
        ),
    ]
    row_group_col: Annotated[
        str, Field(default="Pathway", description="Column name for grouping rows (pathways).")
    ]

    heatmap_annotation: Annotated[
        HeatmapAnnotationConfig | None,
        Field(default=None, description="Configuration for heatmap annotations."),
    ]
    row_values_color_dict: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description="Legacy parameter - Dictionary mapping mutation types to colors.",
        ),
    ]
    value_col: Annotated[
        str,
        Field(
            default="Variant_type",
            description="Legacy parameter - Column name for mutation values.",
        ),
    ]
    value_legend_title: Annotated[
        str | None,
        Field(
            default=None, description="Legacy parameter - Title for the mutation legend section."
        ),
    ]

    aspect: Annotated[int | float, Field(default=1.0, description="Aspect ratio for the plot.")]

    col_split_gap: Annotated[
        float | int, Field(default=0.5, description="Gap between column split groups.")
    ]
    row_split_gap: Annotated[float, Field(default=0.5, description="Gap between row split groups.")]
    legend_category_order: Annotated[
        list[str] | None,
        Field(default=None, description="Order of categories in the legend."),
    ]
    bar_offset: Annotated[
        float | int,
        Field(default=-0.65, description="Offset distance from gene labels to pathway bars."),
    ]
    bar_buffer: Annotated[
        float | int,
        Field(default=0.0, description="Extra buffer to shift row group labels/bars left."),
    ]

    cell_aspect: Annotated[
        float | int, Field(default=1.0, description="Aspect ratio for cells (width/height).")
    ]
    xticklabel_xoffset: Annotated[
        float | int,
        Field(default=0.1, description="Offset to shift the x tick label left or right."),
    ]
    xticklabel_yoffset: Annotated[
        float | int, Field(default=0.05, description="Offset to shift the y tick label up or down.")
    ]

    fig_y_margin: Annotated[
        float | int, Field(default=0.01, description="Margin to add to the y-axis.")
    ] = None

    legend_bbox_to_anchor: Annotated[
        Union[
            Tuple[float, float], Tuple[float, float, float], Tuple[float, float, float, float], None
        ],
        Field(default=None, description="Bounding box for the legend placement."),
    ] = None

    legend_offset: Annotated[
        float | int,
        Field(default=0.01, description="Amount to offset the legend."),
    ]
    remove_unused_keys_in_legend: Annotated[
        bool,
        Field(
            default=True,
            description="If True, only show legend entries for values present in the data.",
        ),
    ]

    @model_validator(mode="after")
    def validate_heatmap_annotation(self) -> "OncoplotConfig":
        if self.heatmap_annotation is None and self.row_values_color_dict is None:
            raise ValueError("Either heatmap_annotation or row_values_color_dict must be provided")
        return self

    @model_validator(mode="after")
    def set_split_gaps_based_on_aspect(self) -> "OncoplotConfig":
        aspect = getattr(self, "aspect", 1.0)
        if aspect > 1.0:
            object.__setattr__(self, "row_split_gap", self.row_split_gap / aspect)
        elif aspect < 1.0:
            object.__setattr__(self, "col_split_gap", self.col_split_gap * aspect)
        return self


class OncoplotAutoConfig(BaseModel):
    plot_type_settings: Annotated[
        dict[str, Any],
        Field(
            default={
                "BL": {
                    "data_generator": "generate_bl_df_for_oncoplot",
                    "default_basename": "BL_Oncoplot",
                    "title_suffix": "Baseline",
                },
                "EOT": {
                    "data_generator": "generate_eot_df_for_oncoplot",
                    "default_basename": "EOT_Oncoplot",
                    "title_suffix": "EOT",
                },
            },
            description="Set defaults for plot type and data generator",
        ),
    ]
    heatmap_annotation_config: Annotated[
        dict[str, Any],
        Field(
            default={
                "values": "Variant_type",
                "legend_title": "Mutation Types",
                "color_key": "MUTATION_TYPE_COLOR_DICT",
            },
            description="Heatmap annotation configuration settings",
        ),
    ]
    data_generator: Annotated[
        str,
        Field(default="generate_bl_df_for_oncoplot", description="R data generator function name"),
    ]
    default_basename: Annotated[
        str, Field(default="BL_Oncoplot", description="Default output basename")
    ]
    title_suffix: Annotated[str, Field(default="Baseline", description="Plot title suffix")]

    compound_study: Annotated[str | None, Field(default=None, description="Compound study name")]
    split_by_indication: Annotated[
        dict[str, bool],
        Field(
            default={
                "6236-001": True,
                "6291-001": True,
                "6291-101": True,
                "9805-001": True,
                "LU-101A": True,
                "LU-101B": True,
            },
            description="Split by indication per study",
        ),
    ]
    add_missing_genes: Annotated[
        dict[str, bool],
        Field(
            default={
                "6291-001": True,
                "6291-101": True,
                "9805-001": True,
                "6236-001": True,
                "LU-101A": True,
                "LU-101B": True,
            },
            description="Add missing genes per study",
        ),
    ]
    bor_type: Annotated[
        dict[str, dict[str, str]],
        Field(
            default={
                "6236-001": {"BL": "con", "EOT": "con"},
                "6291-001": {"BL": "con", "EOT": "con"},
                "6291-101": {"BL": "unc", "EOT": "con"},
                "9805-001": {"BL": "unc", "EOT": "con"},
                "LU-101A": {"BL": "unc", "EOT": "con"},
                "LU-101B": {"BL": "unc", "EOT": "con"},
            },
            description="BOR type override per study and plot type",
        ),
    ]
    dose_filters: Annotated[
        dict[str, dict[str, list[int]]],
        Field(
            default={
                "6236-001": {
                    "PDAC": [160, 200, 220, 300],
                    "NSCLC": [120, 160, 200, 220],
                },
            },
            description="Dose filters per study/indication",
        ),
    ]
    use_therapy_type_split: Annotated[
        dict[str, bool],
        Field(
            default={
                "6236-001": True,
                "6291-001": False,
                "6291-101": False,
                "9805-001": True,
                "LU-101A": False,
                "LU-101B": False,
            },
            description="Use therapy type split per study",
        ),
    ]
    use_cohort_type_split: Annotated[
        dict[str, bool],
        Field(
            default={
                "6236-001": False,
                "6291-001": False,
                "6291-101": False,
                "9805-001": False,
                "LU-101A": False,
                "LU-101B": False,
            },
            description="Use cohort type split per study",
        ),
    ]

    study_specific_annotations: Annotated[
        dict[str, dict[str, list[str]]],
        Field(
            default={
                "6236-001": {
                    "BL": [
                        "Indication",
                        "Dose",
                        "TMB_category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Dose",
                        "TMB_category",
                        "PFS_Category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                },
                "6291-001": {
                    "BL": [
                        "Indication",
                        "Prior_G12Ci",
                        "Dose",
                        "TMB_category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Prior_G12Ci",
                        "Dose",
                        "TMB_category",
                        "PFS_Category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                },
                "6291-101": {
                    "BL": [
                        "Indication",
                        "Prior_G12Ci",
                        "Dose",
                        "TMB_category",
                        "BOR_unc",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Prior_G12Ci",
                        "Dose",
                        "TMB_category",
                        "PFS_Category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                },
                "9805-001": {
                    "BL": [
                        "Indication",
                        "Dose",
                        "TMB_category",
                        "BOR_unc",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Dose",
                        "TMB_category",
                        "PFS_Category",
                        "BOR_con",
                        "TP53",
                        "Patient_ID",
                    ],
                },
                "LU-101A": {
                    "BL": [
                        "Indication",
                        "Part",
                        "Dose",
                        "Dose_Cohort",
                        "TMB_category",
                        "BOR_unc",
                        "Enrollment_Codon",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Part",
                        "Dose",
                        "Dose_Cohort",
                        "TMB_category",
                        "PFS_Category",
                        "Responder_Type",
                        "BOR_con",
                        "Enrollment_Codon",
                        "TP53",
                        "Patient_ID",
                    ],
                },
                "LU-101B": {
                    "BL": [
                        "Indication",
                        "Part",
                        "Dose",
                        "Dose_Cohort",
                        "TMB_category",
                        "BOR_unc",
                        "Enrollment_Codon",
                        "TP53",
                        "Patient_ID",
                    ],
                    "EOT": [
                        "Indication",
                        "Part",
                        "Dose",
                        "Dose_Cohort",
                        "TMB_category",
                        "PFS_Category",
                        "Responder_Type",
                        "BOR_con",
                        "Enrollment_Codon",
                        "TP53",
                        "Patient_ID",
                    ],
                },
            },
            description=(
                "Study-specific annotation column order configuration per study and plot type. "
            ),
        ),
    ]

    indication_order: Annotated[
        list[str],
        Field(
            default=["PDAC", "NSCLC", "CRC", "GYNECOLOGIC", "MELANOMA", "OTHER"],
            description="Order of indications",
        ),
    ]
    first_pathways: Annotated[
        list[str],
        Field(default=["RAS", "MAPK", "RTK", "PI3K"], description="Pathways to show first"),
    ]

    ppt_width: Annotated[
        float, Field(default=13.33, description="Width of the PowerPoint slide in inches")
    ]
    ppt_height: Annotated[
        float, Field(default=7.5, description="Height of the PowerPoint slide in inches")
    ]

    base_fontsize: Annotated[int, Field(default=16, description="Base font size")]
    target_cell_width: Annotated[float, Field(default=0.5, description="Cell width in inches")]
    target_cell_height: Annotated[float, Field(default=0.5, description="Cell height in inches")]
    auto_adjust_cell_size: Annotated[bool, Field(default=True, description="Auto cell size")]

    class Config:
        extra = "allow"
