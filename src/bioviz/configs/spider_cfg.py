from typing import Annotated, Any

from pydantic import BaseModel, Field
from .base_cfg import BasePlotConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class StyledSpiderPlotConfig(BasePlotConfig):
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

    # no per-config data preparation here — bioviz operates on long-format data
    # and leaves forward-fill/reshaping responsibilities to callers/adapters.
