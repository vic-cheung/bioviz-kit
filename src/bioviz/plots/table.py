"""
Table utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

import matplotlib.pyplot as plt
import pandas as pd

from bioviz.configs import StyledTableConfig
from bioviz.utils.plotting import resolve_font_family

__all__ = ["TablePlotter"]


def generate_styled_table(
    df: pd.DataFrame,
    config: StyledTableConfig,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Generate a styled matplotlib table figure from a DataFrame using `StyledTableConfig`.

    Args:
       df: pandas DataFrame containing the table data (rows x columns).
       config: `StyledTableConfig` (pydantic) controlling visual aspects such as
          title, font sizes, header/body colors, row heights, table width, and
          automatic shrinking behavior when many rows are present.
       ax: Optional matplotlib `Axes` to draw the table into; if omitted a new
          figure and axes will be created and returned.

    Returns:
       A matplotlib `Figure` (or None if input DataFrame is empty).
    """

    if df.empty:
        print("DataFrame is empty.")
        return None

    created_fig = None
    if ax is None:
        # Use a consistent base figure size; avoid implicit scaling surprises
        created_fig, ax = plt.subplots()
        try:
            created_fig.patch.set_facecolor('white')
            created_fig.patch.set_alpha(0.0)
        except Exception:
            pass
    fig = ax.figure

    ax.axis("off")

    header_height = (
        config.header_row_height
        if config.header_row_height is not None
        else config.row_height
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

    header_family = config.header_font_family or resolve_font_family()
    body_family = config.body_font_family or resolve_font_family()
    title_family = header_family or body_family or resolve_font_family()

    for (row, col), cell in table.get_celld().items():
        text_obj = cell.get_text()
        is_header = row == 0
        cell.set_edgecolor(config.edge_color)
        font_size = config.header_font_size if is_header else config.cell_font_size
        if len(text_obj.get_text()) > config.max_chars:
            font_size = max(8, font_size - config.shrink_by)
        text_obj.set_fontsize(font_size)
        family = header_family if is_header else body_family
        if family:
            text_obj.set_fontname(family)
        text_obj.set_fontweight(
            config.header_font_weight if is_header else config.body_font_weight
        )
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
            fontfamily=title_family,
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


class TablePlotter:
    """Stateful wrapper for styled tables.

    Construct with `(df, config)` where `config` is `StyledTableConfig` or
    a dict acceptable to it. Delegates rendering to `generate_styled_table`.
    """

    def __init__(self, df: pd.DataFrame, config: StyledTableConfig | dict):
        if isinstance(config, dict):
            config = StyledTableConfig(**config)
        self.df = df.copy()
        self.config = config
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    def set_data(self, df: pd.DataFrame) -> "TablePlotter":
        self.df = df.copy()
        return self

    def update_config(self, **kwargs) -> "TablePlotter":
        for k, v in kwargs.items():
            try:
                setattr(self.config, k, v)
            except Exception:
                continue
        return self

    def plot(self, ax: plt.Axes | None = None) -> tuple[plt.Figure | None, plt.Axes | None]:
        """Render the styled table and store `fig, ax` on the instance."""
        self.fig = generate_styled_table(self.df, self.config, ax=ax)
        if self.fig is None:
            self.ax = None
        else:
            try:
                self.ax = self.fig.axes[0] if self.fig.axes else None
            except Exception:
                self.ax = None
        return self.fig, self.ax

    def save(self, path: str, **save_kwargs) -> None:
        from pathlib import Path

        if self.fig is None:
            raise RuntimeError("No figure available; call .plot() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, **save_kwargs)

    def close(self) -> None:
        try:
            if self.fig is not None:
                plt.close(self.fig)
        finally:
            self.fig = None
            self.ax = None
