from typing import Annotated

from pydantic import BaseModel, Field

from .base_cfg import BasePlotConfig


class XAxisAnnotationOverlayConfig(BasePlotConfig):
    """
    Config for annotating x-axis positions (e.g., scan/assessment labels) with optional overlays."""

    x: Annotated[str, Field()]
    y: Annotated[str, Field()]
    hue_col: Annotated[str, Field()]
    annotation_col: Annotated[str | None, Field(default=None)]
    palette: Annotated[dict | list | None, Field(default=None)]
    linestyle: Annotated[str, Field(default=":")]
    alpha: Annotated[float, Field(default=0.5)]


# Backwards compatibility with previous name
ScanOverlayPlotConfig = XAxisAnnotationOverlayConfig
