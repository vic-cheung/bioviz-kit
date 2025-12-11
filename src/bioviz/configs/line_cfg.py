from typing import Annotated, Any

from pydantic import BaseModel, Field
from .base_cfg import BasePlotConfig


class StyledLinePlotConfig(BasePlotConfig):
    patient_id: Annotated[str | int, Field()]
    label_col: Annotated[str, Field(default="label")]
    subset_cols: Annotated[
        list[str], Field(default_factory=lambda: ["Patient_ID", "label", "Timepoint"])
    ]
    filter_dict: Annotated[dict[str, list[str]] | None, Field(default=None)]
    x: Annotated[str, Field(default="Timepoint")]
    xlabel: Annotated[str | None, Field(default=None)]
    y: Annotated[str, Field(default="Value")]
    ylabel: Annotated[str | None, Field(default=None)]
    secondary_group_col: Annotated[str, Field(default="Variant_type")]
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
