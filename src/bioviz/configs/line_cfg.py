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
