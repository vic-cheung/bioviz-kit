"""
Table utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

import matplotlib.pyplot as plt
import pandas as pd

from bioviz.configs import StyledTableConfig
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

__all__ = ["generate_styled_table"]


def generate_styled_table(
    df: pd.DataFrame,
    config: StyledTableConfig,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if df.empty:
        print("DataFrame is empty.")
        return None

    created_fig = None
    if ax is None:
        # Use a consistent base figure size; avoid implicit scaling surprises
        created_fig, ax = plt.subplots()
    fig = ax.figure

    ax.axis("off")

    header_height = (
        config.header_row_height if config.header_row_height is not None else config.row_height
    )

    # Reduce margins so saved output is tight around the table
    table_left = 0.0
    table_bottom = 0.0

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        cellLoc="center",
        edges="closed",
        bbox=[
            table_left,
            table_bottom,
            config.table_width,
            header_height + (config.row_height * df.shape[0]),
        ],
    )

    if config.absolute_font_size:
        table.auto_set_font_size(False)

    row_heights = {}
    for (row, col), cell in table.get_celld().items():
        text = cell.get_text().get_text()
        # Optionally ignore embedded newlines to keep uniform row heights
        num_lines = text.count("\n") + 1 if config.respect_newlines else 1
        is_header = row == 0
        base_height = header_height if is_header else config.row_height
        if is_header:
            needed_height = base_height
        else:
            needed_height = base_height * num_lines * config.row_height_multiplier
        row_heights[row] = max(row_heights.get(row, base_height), needed_height)

    total_height = sum(row_heights.values())

    # When there are many rows, cap the overall table height to avoid oversized single-line cells
    scale_factor = 1.0
    if (
        config.auto_shrink_total_height
        and df.shape[0] >= config.shrink_row_threshold
        and total_height >= config.max_total_height
    ):
        scale_factor = config.max_total_height / total_height

    effective_total_height = total_height * scale_factor
    table._bbox = [table_left, table_bottom, config.table_width, effective_total_height]

    for (row, col), cell in table.get_celld().items():
        text_obj = cell.get_text()
        is_header = row == 0
        cell.set_edgecolor(config.edge_color)
        font_size = config.header_font_size if is_header else config.cell_font_size
        if len(text_obj.get_text()) > config.max_chars:
            font_size = max(8, font_size - config.shrink_by)
        text_obj.set_fontsize(font_size)
        text_obj.set_fontname(config.header_font_family if is_header else config.body_font_family)
        text_obj.set_fontweight(config.header_font_weight if is_header else config.body_font_weight)
        text_obj.set_color(config.header_text_color if is_header else "black")
        text_obj.set_ha("center")
        text_obj.set_va("center")
        # Keep row height fractions normalized; bbox enforces total height
        cell.set_height(row_heights[row] / total_height)
        if is_header:
            cell.set_facecolor(config.header_bg_color)
        else:
            cell.set_facecolor(config.row_colors[row % 2])

    if config.title:
        ax.text(
            0.5,
            table_bottom + effective_total_height + 0.05,
            config.title,
            ha="center",
            va="bottom",
            fontsize=config.title_font_size,
            fontweight="bold",
            transform=ax.transAxes,
        )

    # Tighten layout and adjust figure size to content. Use sensible minimums so very
    # small configs still produce a readable figure.
    padding_w, padding_h = 0.4, 0.4
    content_width = config.table_width
    content_height = effective_total_height
    min_width, min_height = 2.0, 1.5
    fig.set_size_inches(
        max(content_width + padding_w, min_width),
        max(content_height + padding_h, min_height),
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if created_fig is not None:
        created_fig.tight_layout()

    return created_fig or fig
