from typing import Annotated, Any

from pydantic import BaseModel, Field
from .base_cfg import BasePlotConfig


class StyledLinePlotConfig(BasePlotConfig):
    patient_id: Annotated[str | int, Field()]
    # Let callers define column semantics; keep None defaults here.
    label_col: Annotated[str | None, Field(default=None)]
    subset_cols: Annotated[list[str] | None, Field(default=None)]
    filter_dict: Annotated[dict[str, list[str]] | None, Field(default=None)]
    x: Annotated[str | None, Field(default=None)]
    xlabel: Annotated[str | None, Field(default=None)]
    y: Annotated[str | None, Field(default=None)]
    ylabel: Annotated[str | None, Field(default=None)]
    secondary_group_col: Annotated[str | None, Field(default=None)]
    match_legend_text_color: Annotated[bool, Field(default=True)]
    # Default to no point labels to keep visuals generic; callers can enable as needed.
    label_points: Annotated[bool, Field(default=False)]
    xlim_padding: Annotated[float, Field(default=0.8)]
    ylim: Annotated[tuple[float | None, float | None] | None, Field(default=None)]
    xlim: Annotated[tuple[float | None, float | None] | None, Field(default=None)]
    add_extra_tick_to_ylim: Annotated[bool, Field(default=True)]
    ymin_padding_fraction: Annotated[float, Field(default=0.05)]
    threshold: Annotated[float | int | None, Field(default=None)]
    threshold_color: Annotated[str, Field(default="#C0C0C0")]
    threshold_style: Annotated[str, Field(default="--")]
    threshold_width: Annotated[float, Field(default=1.0)]
    filled_marker_scale: Annotated[float, Field(default=1.2)]
    # canonical package expects long-format DataFrame inputs; downstream adapters
    # (e.g. tm_toolbox) should perform any forward-fill or wide->long transformations.


class StyledLinePlotOverlayConfig(BasePlotConfig):
    """
    Multi-series longitudinal line plot with optional x-axis annotation overlay.
    """

    group_col: Annotated[str, Field()]
    x: Annotated[str, Field()]
    y: Annotated[str, Field()]
    xlabel: Annotated[str | None, Field(default=None)]
    ylabel: Annotated[str | None, Field(default=None)]
    subgroup_name: Annotated[str | None, Field(default=None)]
    marker_style: Annotated[str, Field(default="o")]
    color_dict_subgroup: Annotated[dict[Any, Any] | None, Field(default=None)]
    linestyle_dict: Annotated[dict[Any, str] | None, Field(default=None)]
    markerstyle_dict: Annotated[dict[Any, str] | None, Field(default=None)]
    # bioviz expects callers to supply properly long-format data; adapters may
    # perform forward-fill or wide->long transformations.
    linestyle_col: Annotated[str | None, Field(default=None)]
    markerstyle_col: Annotated[str | None, Field(default=None)]
    use_absolute_scale: Annotated[bool, Field(default=False)]
    absolute_ylim: Annotated[tuple[float, float] | None, Field(default=None)]
    absolute_yticks: Annotated[list[float] | None, Field(default=None)]

    # Optional horizontal baseline line (e.g., at 0). If provided, a dashed line is drawn.
    baseline: Annotated[float | None, Field(default=None)]
    baseline_color: Annotated[str, Field(default="#C0C0C0")]
    baseline_style: Annotated[str, Field(default="--")]
    baseline_width: Annotated[float, Field(default=1.0)]
    baseline_dashes: Annotated[tuple[float, float] | None, Field(default=(5, 5))]

    # no per-config data preparation here — bioviz operates on long-format data
    # and leaves forward-fill/reshaping responsibilities to callers/adapters.


class XAxisAnnotationOverlayConfig(BasePlotConfig):
    """Config for annotating x-axis positions (e.g., assessments) with optional overlays."""

    x: Annotated[str, Field()]
    y: Annotated[str, Field()]
    hue_col: Annotated[str, Field()]
    annotation_col: Annotated[str | None, Field(default=None)]
    palette: Annotated[dict | list | None, Field(default=None)]
    linestyle: Annotated[str, Field(default=":")]
    alpha: Annotated[float, Field(default=0.5)]


# Backwards compatibility alias
ScanOverlayPlotConfig = XAxisAnnotationOverlayConfig


class LineplotOverlayConfig(BaseModel):
    """Container for overlay layers: line trajectories plus optional x-axis annotations."""

    line: Annotated[StyledLinePlotOverlayConfig | None, Field(default=None)]
    annotations: Annotated[XAxisAnnotationOverlayConfig | None, Field(default=None)]
