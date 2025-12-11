from typing import Annotated, Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, Field

try:
    from pydantic import ConfigDict
except Exception:
    ConfigDict = None


class TopAnnotationConfig(BaseModel):
    """Configuration for top annotation tracks in oncoplots.

    Top annotations are horizontal tracks displayed above the main heatmap,
    showing categorical data (e.g., cohort, dose, BOR) for each patient/column.

    Label display controls:
        - merge_labels=True, show_category_labels=False (default):
          Shows merged labels over consecutive cells with the same value.
        - merge_labels=False, show_category_labels=True:
          Shows a label in each cell (can be cluttered).
        - merge_labels=False, show_category_labels=False:
          No labels displayed inside annotation bars (colors only).

        Border controls:
                - draw_border=False (default): no border unless enabled or the color is white.
                - border_categories: list of category values that should have borders
                    (None = all categories if draw_border=True).
    """

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        model_config = BaseModel.model_config if hasattr(BaseModel, "model_config") else {}
    values: Annotated[pd.Series | dict[Any, Any] | Sequence[Any] | str, Any]
    colors: Annotated[dict, Any]
    height: Annotated[float, Any] = 1
    fontsize: Annotated[int, Any] = 16
    display_name: Annotated[str | None, Any] = None
    legend_title: Annotated[str | None, Any] = None
    legend_value_order: Annotated[list[str] | None, Any] = None
    show_category_labels: Annotated[bool, Any] = False
    merge_labels: Annotated[bool, Any] = True
    label_fontsize: Annotated[int, Any] = 12
    label_text_colors: Annotated[dict | None, Any] = None
    na_color: Annotated[str, Any] = "gainsboro"
    # Borders off by default; white cells still get a border for visibility
    draw_border: Annotated[bool, Any] = False
    border_categories: Annotated[list[str] | None, Any] = None
    border_color: Annotated[str, Any] = "black"
    border_width: Annotated[float, Any] = 0.8

    @field_validator("values", mode="before")
    def _coerce_values(cls, v):
        # None or string (column reference) pass through
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, pd.Series):
            return v
        if isinstance(v, dict):
            return pd.Series(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return pd.Series(v)
        return v


class HeatmapAnnotationConfig(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        model_config = BaseModel.model_config if hasattr(BaseModel, "model_config") else {}

    values: Annotated[str | pd.Series | Sequence[Any], Any]
    # Allow constructing a HeatmapAnnotationConfig without explicitly
    # providing `colors`. The plotting code will fall back to the
    # `OncoplotConfig.row_values_color_dict` when `colors` is empty.
    colors: Annotated[dict | None, Any] = Field(default_factory=dict)
    legend_title: Annotated[str | None, Any] = None
    legend_value_order: Annotated[list[str] | None, Any] = None
    draw_border: Annotated[bool, Any] = False
    border_categories: Annotated[list[str] | None, Any] = None
    border_color: Annotated[str, Any] = "black"
    border_width: Annotated[float, Any] = 0.5
    bottom_left_triangle_values: Annotated[list[str], Any] = ["SNV"]
    upper_right_triangle_values: Annotated[list[str], Any] = ["CNV"]

    @field_validator("values", mode="before")
    def _coerce_heatmap_values(cls, v):
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, pd.Series):
            return v
        if isinstance(v, dict):
            return pd.Series(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return pd.Series(v)
        return v


def make_annotation_config(
    values: pd.Series | dict[Any, Any],
    colors: dict,
    display_name: str,
    legend_title: str,
    **kwargs: Any,
) -> TopAnnotationConfig:
    return TopAnnotationConfig(
        values=values, colors=colors, display_name=display_name, legend_title=legend_title, **kwargs
    )
