"""
Table utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

import matplotlib.pyplot as plt
import pandas as pd

from bioviz.plot_configs import StyledTableConfig
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
        if config.absolute_font_size:
            figsize = (config.table_scale[0], config.table_scale[1])
            created_fig, ax = plt.subplots(figsize=figsize)
        else:
            created_fig, ax = plt.subplots()

    ax.axis("off")

    header_height = (
        config.header_row_height if config.header_row_height is not None else config.row_height
    )

    table_left = 0.05
    table_bottom = 0.15

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
        num_lines = text.count("\n") + 1
        is_header = row == 0
        base_height = header_height if is_header else config.row_height
        if is_header:
            needed_height = base_height
        else:
            needed_height = base_height * num_lines * config.row_height_multiplier
        row_heights[row] = max(row_heights.get(row, base_height), needed_height)

    total_height = sum(row_heights.values())
    table._bbox = [table_left, table_bottom, config.table_width, total_height]

    if config.absolute_font_size:
        table.scale(*config.table_scale)
    else:
        table.scale(*config.table_scale)

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
        cell.set_height(row_heights[row] / total_height)
        if is_header:
            cell.set_facecolor(config.header_bg_color)
        else:
            cell.set_facecolor(config.row_colors[row % 2])

    if config.title:
        ax.text(
            0.5,
            table_bottom + total_height + 0.05,
            config.title,
            ha="center",
            va="bottom",
            fontsize=config.title_font_size,
            fontweight="bold",
            transform=ax.transAxes,
        )

    if created_fig is not None:
        created_fig.tight_layout()

    return created_fig
