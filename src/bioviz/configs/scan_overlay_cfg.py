from typing import Annotated, Any

from pydantic import BaseModel, Field
from .base_cfg import BasePlotConfig


class ScanOverlayPlotConfig(BasePlotConfig):
    x: Annotated[str, Field()]
    y: Annotated[str, Field()]
    hue_col: Annotated[str, Field()]
    recist_col: Annotated[str, Field()]
    palette: Annotated[dict | list | None, Field(default=None)]
    linestyle: Annotated[str, Field(default=":")]
    alpha: Annotated[float, Field(default=0.5)]
