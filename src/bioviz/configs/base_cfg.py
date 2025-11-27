from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BasePlotConfig(BaseModel):
    palette: Annotated[
        list[Any] | dict[Any, Any] | None,
        Field(default=None, description="Color palette to use for the plot."),
    ]
    lw: Annotated[float, Field(default=4.0, description="Line width for plotted lines.")]
    markersize: Annotated[
        float | None,
        Field(default=None, description="Size of markers; defaults to twice the line width."),
    ]
    figsize: Annotated[tuple[int, int], Field(default=(9, 6), description="Figure size.")]
    legend_loc: Annotated[str, Field(default="upper left", description="Legend location.")]

    col_vals_to_include_in_title: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Columns whose values are included in the plot title if no title is provided. "
                "Default is empty for a generic package; downstream packages (e.g. tm-toolbox) "
                "may provide their own defaults when constructing configs."
            ),
        ),
    ]

    title: Annotated[str | None, Field(default=None, description="Custom title for the plot.")]

    rhs_pdf_padding: Annotated[
        float,
        Field(
            default=0.8, description="Padding on the right side to avoid clipping in PDF exports."
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
