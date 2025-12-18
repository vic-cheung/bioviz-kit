from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BasePlotConfig(BaseModel):
    """
    Base configuration shared by multiple plot types.

    This pydantic model centralizes common visual parameters such as palettes,
    line width, marker sizing, figure size, legend placement and font sizes.
    Subclasses (e.g., `LinePlotConfig`) extend this with plot-specific fields.
    """

    palette: Annotated[
        list[Any] | dict[Any, Any] | None,
        Field(default=None, description="Color palette to use for the plot."),
    ]
    lw: Annotated[float, Field(default=4.0, description="Line width for plotted lines.")]
    markersize: Annotated[
        float | None,
        Field(
            default=None,
            description="Size of markers; defaults to twice the line width.",
        ),
    ]
    figsize: Annotated[tuple[int, int], Field(default=(9, 6), description="Figure size.")]
    legend_loc: Annotated[str, Field(default="upper left", description="Legend location.")]

    xlabel_fontsize: Annotated[
        float | int | None,
        Field(default=20, description="Optional x-axis label font size (default 20)."),
    ]
    ylabel_fontsize: Annotated[
        float | int | None,
        Field(default=20, description="Optional y-axis label font size (default 20)."),
    ]
    xtick_fontsize: Annotated[
        float | int | None,
        Field(default=16, description="Optional x-tick label font size (default 16)."),
    ]
    ytick_fontsize: Annotated[
        float | int | None,
        Field(default=16, description="Optional y-tick label font size (default 16)."),
    ]

    title_fontsize: Annotated[
        float | int | None,
        Field(default=20, description="Optional title font size (default 20)."),
    ]

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
            default=0.8,
            description="Padding on the right side to avoid clipping in PDF exports.",
        ),
    ]

    # Figure-level display controls; keep axes facecolor opaque. These
    # configure the Figure patch used for export/transparent backgrounds.
    figure_facecolor: Annotated[str | None, Field(default="white")]
    figure_transparent: Annotated[bool, Field(default=True)]

    @field_validator("palette")
    @classmethod
    def validate_palette(
        cls, v: list[Any] | dict[Any, Any] | None
    ) -> list[Any] | dict[Any, Any] | None:
        """
        Validate that the provided palette is a supported type.

        Args:
           cls: The pydantic model class (unused here but required for validator signature).
           v: Palette value; expected to be a list, dict, or None.

        Returns:
           The validated palette (unchanged when valid).

        Raises:
           ValueError: If `v` is not a list, dict, or None.
        """

        if v is not None and not isinstance(v, (list, dict)):
            raise ValueError("Palette must be a list, a dict, or None.")
        return v

    @model_validator(mode="after")
    def set_default_markersize(cls, values: "BasePlotConfig") -> "BasePlotConfig":
        """
        Set a default markersize after model validation when not provided.

        Args:
            cls: The model class (validator method).
            values: The validated `BasePlotConfig` instance under construction.

        Behavior:
            If `markersize` is not supplied, it will be set to twice the configured
            line width (`lw`) to provide reasonable default marker sizing.

        Returns:
            The `BasePlotConfig` instance with `markersize` possibly populated.
        """

        if values.markersize is None:
            values.markersize = values.lw * 2
        return values
