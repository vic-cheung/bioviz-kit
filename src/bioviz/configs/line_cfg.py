from typing import Annotated, Any

from pydantic import Field
from .base_cfg import BasePlotConfig


class LinePlotConfig(BasePlotConfig):
    """
    Unified config for single- and multi-series longitudinal line plots (and optional overlay).
    """

    # Identity / grouping
    entity_id: Annotated[str | int | None, Field(default=None, alias="patient_id")]
    group_col: Annotated[str | None, Field(default=None)]
    label_col: Annotated[str | None, Field(default=None)]  # preferred hue column for single-series
    secondary_group_col: Annotated[str | None, Field(default=None)]
    subgroup_name: Annotated[str | None, Field(default=None)]

    # Axes
    x: Annotated[str | None, Field(default=None)]
    y: Annotated[str | None, Field(default=None)]
    xlabel: Annotated[str | None, Field(default=None)]
    ylabel: Annotated[str | None, Field(default=None)]

    # Labeling / selection
    subset_cols: Annotated[list[str] | None, Field(default=None)]
    filter_dict: Annotated[dict[str, list[str]] | None, Field(default=None)]
    label_points: Annotated[bool, Field(default=False)]
    match_legend_text_color: Annotated[bool, Field(default=True)]

    # Styling
    marker_style: Annotated[str, Field(default="o")]
    color_dict_subgroup: Annotated[dict[Any, Any] | None, Field(default=None)]
    linestyle_dict: Annotated[dict[Any, str] | None, Field(default=None)]
    markerstyle_dict: Annotated[dict[Any, str] | None, Field(default=None)]
    linestyle_col: Annotated[str | None, Field(default=None)]
    markerstyle_col: Annotated[str | None, Field(default=None)]

    # Scaling
    use_absolute_scale: Annotated[bool, Field(default=False)]
    absolute_ylim: Annotated[tuple[float, float] | None, Field(default=None)]
    absolute_yticks: Annotated[list[float] | None, Field(default=None)]
    reference: Annotated[float | None, Field(default=None)]
    reference_color: Annotated[str, Field(default="#C0C0C0")]
    reference_style: Annotated[str, Field(default="--")]
    reference_width: Annotated[float, Field(default=1.0)]
    reference_dashes: Annotated[tuple[float, float] | None, Field(default=(5, 5))]
    reference_alpha: Annotated[float, Field(default=1.0)]

    # Limits / ticks
    xlim_padding: Annotated[float, Field(default=0.8)]
    ylim: Annotated[tuple[float | None, float | None] | None, Field(default=None)]
    xlim: Annotated[tuple[float | None, float | None] | None, Field(default=None)]
    add_extra_tick_to_ylim: Annotated[bool, Field(default=True)]
    ymin_padding_fraction: Annotated[float, Field(default=0.05)]
    align_first_tick_to_origin: Annotated[bool, Field(default=False)]
    symmetric_ylim: Annotated[bool, Field(default=True)]

    # Threshold support
    threshold: Annotated[float | int | None, Field(default=None)]
    threshold_color: Annotated[str, Field(default="#C0C0C0")]
    threshold_style: Annotated[str, Field(default="--")]
    threshold_width: Annotated[float, Field(default=1.0)]
    threshold_dashes: Annotated[tuple[float, float] | None, Field(default=(5, 5))]
    threshold_alpha: Annotated[float, Field(default=1.0)]
    threshold_legend_title: Annotated[str, Field(default="Threshold")]
    threshold_below_label: Annotated[str, Field(default="Below Threshold")]
    threshold_above_label: Annotated[str, Field(default="Above Threshold")]
    filled_marker_scale: Annotated[float, Field(default=1.2)]
    threshold_label: Annotated[str | None, Field(default=None)]
    threshold_label_color: Annotated[str | None, Field(default=None)]
    threshold_label_alpha: Annotated[float, Field(default=1.0)]
    threshold_label_fontsize: Annotated[float, Field(default=14)]

    # Twin/annotations
    twin_alpha: Annotated[float, Field(default=0.5)]  # opacity for twin series
    overlay_col: Annotated[str | None, Field(default=None)]  # annotation text column on twin axis
    overlay_palette: Annotated[dict | str | None, Field(default=None)]  # colors for overlay labels
    overlay_fontweight: Annotated[str, Field(default="bold")]  # weight for overlay labels
    overlay_fontsize: Annotated[float | None, Field(default=None)]  # size for overlay labels
    overlay_in_axes_coords: Annotated[
        bool, Field(default=True)
    ]  # place overlay text in axes coords (fixed visual offset)
    overlay_ypos_axes: Annotated[
        float, Field(default=0.97)
    ]  # y-position in axes fraction when overlay_in_axes_coords is True
    overlay_vline_color: Annotated[str, Field(default="gainsboro")]
    overlay_vline_style: Annotated[str, Field(default="--")]
    overlay_vline_width: Annotated[float, Field(default=1.0)]
    overlay_vline_dashes: Annotated[tuple[float, float] | None, Field(default=(5, 5))]
    overlay_vline_alpha: Annotated[float, Field(default=1.0)]
