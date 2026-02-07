from typing import Annotated

from pydantic import BaseModel, Field


class StyledTableConfig(BaseModel):
    """
    Configuration for styled table rendering.

    Controls table title, fonts, header/body colors, row heights, width and
    behavior for automatic shrinking when many rows are present.

    Each field corresponds to a visual parameter used by `generate_styled_table`.
    """

    title: Annotated[str, Field(default="")]
    title_font_size: Annotated[
        float | int | None,
        Field(
            default=None,
            description="Title font size. None uses rcParams['axes.titlesize'].",
        ),
    ]
    header_bg_color: Annotated[str, Field(default="#1E6B5C")]
    header_text_color: Annotated[str, Field(default="white")]
    row_colors: Annotated[tuple[str, str], Field(default=("#f2f2f2", "gainsboro"))]
    edge_color: Annotated[str, Field(default="white")]
    header_font_size: Annotated[
        float | int | None,
        Field(
            default=None,
            description="Header font size. None uses rcParams['axes.labelsize'].",
        ),
    ]
    header_font_weight: Annotated[str, Field(default="bold")]
    cell_font_size: Annotated[
        float | int | None,
        Field(default=None, description="Cell font size. None uses rcParams['font.size']."),
    ]
    max_chars: Annotated[int, Field(default=18)]
    shrink_by: Annotated[float | int, Field(default=2)]
    row_height: Annotated[float, Field(default=0.3)]
    header_row_height: Annotated[float | None, Field(default=None)]
    row_height_multiplier: Annotated[float, Field(default=0.75)]
    table_width: Annotated[float, Field(default=1.0)]
    table_scale: Annotated[tuple[float, float], Field(default=(12, 12))]
    absolute_font_size: Annotated[bool, Field(default=True)]
    header_font_family: Annotated[str | None, Field(default=None)]
    body_font_family: Annotated[str | None, Field(default=None)]
    body_font_weight: Annotated[str, Field(default="normal")]
    respect_newlines: Annotated[bool, Field(default=True)]
    auto_shrink_total_height: Annotated[bool, Field(default=False)]
    shrink_row_threshold: Annotated[int, Field(default=30)]
    max_total_height: Annotated[float, Field(default=10.0)]
    column_widths: Annotated[
        list[float] | None,
        Field(
            default=None,
            description="Explicit column width fractions (must sum to ~1.0 or table_width). "
            "If provided, takes precedence over auto_column_widths. "
            "E.g., [0.1, 0.1, 0.5, 0.15, 0.15] for 5 columns.",
        ),
    ]
    auto_column_widths: Annotated[
        bool,
        Field(
            default=False,
            description="Automatically size column widths based on content length. "
            "Wider columns for longer text (e.g., 'HAZARD RATIO (95% CI)') vs narrower for short text (e.g., 'PFS'). "
            "Considers both header and cell values - uses whichever is longer per column. "
            "Ignored if column_widths is explicitly set.",
        ),
    ]
    min_column_width_fraction: Annotated[
        float,
        Field(
            default=0.05,
            description="Minimum column width as fraction of table_width when using auto_column_widths "
            "(prevents columns from being too narrow).",
        ),
    ]
