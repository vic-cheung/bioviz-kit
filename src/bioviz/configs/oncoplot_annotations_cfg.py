from typing import Annotated, Any, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - fallback for older pydantic
    ConfigDict = None


class TopAnnotationConfig(BaseModel):
    """
    Configuration for top annotation tracks in oncoplots.
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
    merge_labels: Annotated[bool, Any] = False
    label_fontsize: Annotated[int, Any] = 12
    label_text_colors: Annotated[dict | None, Any] = None
    na_color: Annotated[str, Any] = "gainsboro"
    # Borders off by default; white cells still get a border for visibility
    draw_border: Annotated[bool, Any] = False
    border_categories: Annotated[list[str] | None, Any] = None
    border_color: Annotated[str, Any] = "black"
    border_width: Annotated[float, Any] = 0.8

    @field_validator("values", mode="before")
    def _coerce_values(cls, v):  # noqa: D417
        """
        Coerce various input types into a pandas Series for `values` field.

        Args:
           cls: Validator class reference.
           v: Input value which may be a Series, dict, list, tuple, ndarray, or str/None.

        Returns:
           A `pd.Series` when appropriate, or the original value for strings/None.
        """

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

    @field_validator("colors", mode="before")
    def _coerce_colors_keys(cls, v):
        # Ensure mapping keys are strings to avoid mismatches when comparing
        # against dataframe values that may be stringified in the plotting code.
        if v is None:
            return v
        try:
            return {str(k): val for k, val in dict(v).items()}
        except Exception:
            return v

    @field_validator("legend_value_order", mode="before")
    def _coerce_legend_order(cls, v):
        if v is None:
            return v
        try:
            return [str(x) for x in list(v)]
        except Exception:
            return v


class HeatmapAnnotationConfig(BaseModel):
    """
    Configuration for heatmap-style annotations displayed beneath the oncoplot.

    Fields:
       values: Column name or pd.Series mapping patient/sample -> annotation value.
       colors: Optional mapping of category -> color used when rendering heatmap cells.
       bottom_left_triangle_values / upper_right_triangle_values: Lists of values
          that should be rendered as triangular halves inside the heatmap cell.
    """

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:
        model_config = BaseModel.model_config if hasattr(BaseModel, "model_config") else {}

    values: Annotated[str | pd.Series | Sequence[Any], Any]
    # Allow constructing a HeatmapAnnotationConfig without explicitly providing `colors`.
    # The plotting code will fall back to `OncoplotConfig.row_values_color_dict` when empty.
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
    def _coerce_heatmap_values(cls, v):  # noqa: D417
        """
        Coerce various input types into a pandas Series for Heatmap `values`.

        Args:
           cls: Validator class reference.
           v: Input value (Series, dict, list, tuple, ndarray, or str/None).

        Returns:
           A `pd.Series` when appropriate, or the original value for strings/None.
        """

        if v is None or isinstance(v, str):
            return v
        if isinstance(v, pd.Series):
            return v
        if isinstance(v, dict):
            return pd.Series(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            return pd.Series(v)
        return v

    @field_validator("colors", mode="before")
    def _coerce_heatmap_colors_keys(cls, v):
        if v is None:
            return v
        try:
            return {str(k): val for k, val in dict(v).items()}
        except Exception:
            return v

    @field_validator("legend_value_order", mode="before")
    def _coerce_heatmap_legend_order(cls, v):
        if v is None:
            return v
        try:
            return [str(x) for x in list(v)]
        except Exception:
            return v
