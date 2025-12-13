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
    title_font_size: Annotated[float | int, Field(default=16)]
    header_bg_color: Annotated[str, Field(default="#1E6B5C")]
    header_text_color: Annotated[str, Field(default="white")]
    row_colors: Annotated[tuple[str, str], Field(default=("#f2f2f2", "gainsboro"))]
    edge_color: Annotated[str, Field(default="white")]
    header_font_size: Annotated[float | int, Field(default=16)]
    header_font_weight: Annotated[str, Field(default="bold")]
    cell_font_size: Annotated[float | int, Field(default=16)]
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
