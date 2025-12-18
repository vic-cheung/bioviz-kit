"""
Oncoplot utilities adapted from tm_toolbox, ported into bioviz.

This module implements oncoprint drawing helpers and the
`OncoPlotter` class. It accepts a `style` object implementing
the `StyleBase` protocol and calls `style.apply_theme()` when plotting.
"""

from typing import Any

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.transforms as mtransforms  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from matplotlib import font_manager  # type: ignore
from matplotlib.patches import Patch  # type: ignore

from bioviz.configs import HeatmapAnnotationConfig, OncoplotConfig
from bioviz.utils.style import DefaultStyle, StyleBase
from bioviz.utils.plotting import resolve_font_family

# Do not apply a global style at import time; callers should pass a style.

__all__ = [
    "diagonal_fill",
    "my_shape_func",
    "draw_top_annotation",
    "merge_labels_without_splits",
    "label_block",
    "create_custom_legend_patch",
    "is_white_color",
    "OncoPlotter",
]


def is_white_color(color) -> bool:
    """
    Determine whether a color is effectively white.

    Args:
       color: Any matplotlib-parseable color (name, hex, RGB(A) tuple).

    Returns:
       True if the color is close to white, False otherwise.
    """
    if color is None:
        return False
    try:
        rgb = mcolors.to_rgb(color)
        return all(abs(c - 1.0) < 0.01 for c in rgb)
    except (ValueError, AttributeError):
        if isinstance(color, str):
            return color.lower() in ["white", "#ffffff", "#fff"]
        return False


def _ensure_opaque_color(color, default="white"):
    """
    Ensure a color is opaque; replace transparent/None with a default color.

    Args:
        color: Matplotlib color input (name, hex, rgba tuple) or None.
        default: Color to return when `color` is None or fully transparent.

    Returns:
        A color value acceptable to Matplotlib that is guaranteed not to be fully transparent.
    """
    if color is None:
        return default
    try:
        rgba = mcolors.to_rgba(color)
        # If alpha==0 treat as transparent and replace with default
        if len(rgba) >= 4 and rgba[3] == 0:
            return default
        return color
    except Exception:
        # If color cannot be parsed, return it as-is (matplotlib may raise later)
        return color


def diagonal_fill(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    which_half: str = "bottom_left",
) -> None:
    """
    Fill half of a rectangular cell with a triangular diagonal fill.

    Args:
       ax: Matplotlib Axes to draw on.
       x, y: Lower-left coordinates of the cell.
       width, height: Cell dimensions.
       color: Fill color for the triangle half.
       which_half: Either "bottom_left" or "upper_right" selecting which triangle to fill.
    """
    if which_half == "bottom_left":
        coords = [(x, y + height), (x, y), (x + width, y + height)]
    elif which_half == "upper_right":
        coords = [(x + width, y), (x, y), (x + width, y + height)]
    else:
        raise ValueError("which_half must be 'bottom_left' or 'upper_right'")
    # Always draw an opaque background so the other half of the cell is white
    bg = _ensure_opaque_color(None, default="white")
    rect = mpatches.Rectangle((x, y), width, height, color=bg, linewidth=0, zorder=1)
    ax.add_patch(rect)
    color = _ensure_opaque_color(color, default="white")
    poly = mpatches.Polygon(coords, closed=True, color=color, linewidth=0, zorder=2)
    ax.add_patch(poly)


def my_shape_func(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    value: str,
    bottom_left_values: list[str],
    upper_right_values: list[str],
) -> None:
    """
    Draw a shape for an oncoplot cell: full rectangle or half-triangle variants.

    Args:
       ax: Matplotlib Axes to draw into.
       x, y: Data coordinates for the cell (lower-left).
       width, height: Size of the drawn cell.
       color: Fill color to use (transparency handled).
       value: Value used to decide which half to fill.
       bottom_left_values: Values that should render as bottom-left triangles.
       upper_right_values: Values that should render as upper-right triangles.
    """
    # Avoid drawing transparent/None colors for heatmap cells
    color = _ensure_opaque_color(color, default="white")
    if value in bottom_left_values:
        diagonal_fill(ax, x, y, width, height, color, which_half="bottom_left")
    elif value in upper_right_values:
        diagonal_fill(ax, x, y, width, height, color, which_half="upper_right")
    else:
        rect = mpatches.Rectangle((x, y), width, height, color=color, linewidth=0)
        ax.add_patch(rect)


def draw_top_annotation(
    ax,
    x_values,
    col_positions,
    annotation_y,
    ann_config,
    ann_name,
    col_split_map=None,
    cell_aspect: float = 1.0,
    label_x: float | None = None,
    label_transform=None,
) -> Any:
    """
    Draw a top annotation track above the oncoplot columns.

    Args:
       ax: Matplotlib Axes to draw on.
       x_values: Iterable of x-axis values (column keys) aligning with `col_positions`.
       col_positions: Numeric positions for each column on the axis.
       annotation_y: Y coordinate to place the annotation track.
       ann_config: `TopAnnotationConfig` containing values, colors, and label settings.
       ann_name: Name of the annotation (used as display fallback).
       col_split_map: Optional mapping for split columns (unused in basic rendering).
       cell_aspect: Aspect multiplier for cell width.
       label_x: Optional X coordinate for the left-hand annotation label.
       label_transform: Optional transform to apply to label text.

    Returns:
       The function draws onto `ax` and returns None.
    """

    if not col_positions:
        return
    values = pd.Series(ann_config.values)
    values = values[~values.index.duplicated(keep="first")]
    colors = ann_config.colors
    height = ann_config.height
    fontsize = ann_config.fontsize
    display_name = ann_config.display_name or ann_name
    merge_labels = ann_config.merge_labels
    show_category_labels = ann_config.show_category_labels

    blocks_needing_borders = []
    block_start = None
    last_value = None

    border_categories = getattr(ann_config, "border_categories", None)
    draw_border = getattr(ann_config, "draw_border", False)

    for j, x_value in enumerate(x_values):
        if x_value is None:
            continue
        x = col_positions[j]
        value = values.get(x_value)
        if pd.isna(value) or (isinstance(value, str) and value.strip().lower() == "nan"):
            color = ann_config.na_color
            value_str = "NA"
        else:
            color = colors.get(str(value), "white")
            value_str = str(value)

        ax.add_patch(
            mpatches.Rectangle(
                (x, annotation_y),
                cell_aspect,
                height,
                color=color,
                clip_on=False,
                zorder=10,
            )
        )

        if border_categories is not None:
            needs_border = value_str in border_categories
        elif draw_border:
            needs_border = True
        else:
            needs_border = is_white_color(color)

        if needs_border:
            if block_start is None or value_str != last_value:
                if block_start is not None:
                    blocks_needing_borders.append((block_start, j - 1))
                block_start = j
                last_value = value_str
        else:
            if block_start is not None:
                blocks_needing_borders.append((block_start, j - 1))
                block_start = None
                last_value = None

    if block_start is not None:
        blocks_needing_borders.append((block_start, len(x_values) - 1))

    border_color = getattr(ann_config, "border_color", "black")
    border_width = getattr(ann_config, "border_width", 0.5)

    for start, end in blocks_needing_borders:
        x0 = col_positions[start]
        x1 = col_positions[end] + cell_aspect
        rect = mpatches.Rectangle(
            (x0, annotation_y),
            x1 - x0,
            height,
            fill=False,
            edgecolor=border_color,
            linewidth=border_width,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(rect)

    if col_positions:
        label_x_pos = label_x if label_x is not None else (min(col_positions) - 0.3)
        ax.text(
            label_x_pos,
            annotation_y + height / 2,
            display_name,
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="normal",
            clip_on=False,
            zorder=13,
            transform=label_transform or ax.transData,
        )

    if merge_labels:
        x_value_entries = []
        for i, x_value in enumerate(x_values):
            if x_value is not None:
                value = values.get(x_value)
                if pd.notna(value):
                    x_value_entries.append((i, x_value, value))

        if x_value_entries:
            blocks = []
            current_block = [x_value_entries[0]]

            for i in range(1, len(x_value_entries)):
                prev_idx, prev_x_val, prev_value = x_value_entries[i - 1]
                curr_idx, curr_x_val, curr_value = x_value_entries[i]

                if curr_value == prev_value and curr_idx == prev_idx + 1:
                    current_block.append(x_value_entries[i])
                else:
                    blocks.append(current_block)
                    current_block = [x_value_entries[i]]

            blocks.append(current_block)

            for block in blocks:
                if block:
                    start_idx = block[0][0]
                    end_idx = block[-1][0]
                    value = block[0][2]
                    if start_idx < len(col_positions) and end_idx < len(col_positions):
                        block_cols = col_positions[start_idx : end_idx + 1]
                        if block_cols:
                            x_center = (
                                (min(block_cols) + max(block_cols) + cell_aspect) / 2
                                if block_cols
                                else 0
                            )
                            text_color = (ann_config.label_text_colors or {}).get(
                                str(value), "white"
                            )
                            ax.text(
                                x_center,
                                annotation_y + height / 2,
                                str(value),
                                ha="center",
                                va="center",
                                fontsize=ann_config.label_fontsize or fontsize,
                                color=text_color,
                                fontweight="bold",
                                clip_on=False,
                                zorder=100,
                            )

    elif show_category_labels:
        value_to_positions = {}
        for i, x_value in enumerate(x_values):
            if x_value is None:
                continue
            value = values.get(x_value)
            if pd.notna(value):
                if value not in value_to_positions:
                    value_to_positions[value] = []
                value_to_positions[value].append(col_positions[i])

        for value, positions in value_to_positions.items():
            if positions:
                x_center = (min(positions) + max(positions) + cell_aspect) / 2
                text_color = (ann_config.label_text_colors or {}).get(str(value), "black")
                ax.text(
                    x_center,
                    annotation_y + height / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=ann_config.label_fontsize or fontsize,
                    color=text_color,
                    fontweight="normal",
                    clip_on=False,
                    zorder=100,
                )


def merge_labels_without_splits(
    ax,
    x_values,
    col_positions,
    annotation_y,
    height,
    values,
    ann_config,
    fontsize,
    cell_aspect=1.0,
) -> None:
    """
    Merge adjacent identical annotation values into contiguous labeled blocks.

    Args:
       ax: Matplotlib Axes to draw labels into.
       x_values: Sequence of x keys for columns.
       col_positions: Numeric positions of columns.
       annotation_y: Y coordinate used for label placement.
       height: Height of the annotation track.
       values: Mapping/Series of column key -> value.
       ann_config: Annotation configuration object.
       fontsize: Font size to use for labels.
       cell_aspect: Cell width multiplier.

    Returns:
       None (labels are drawn on `ax`).
    """

    last_value = None
    block_start_idx = None
    for i, x_value in enumerate(list(x_values) + [None]):
        value = values.get(x_value) if x_value is not None else None
        if value != last_value and block_start_idx is not None:
            block_end_idx = i - 1
            if (
                block_end_idx >= 0
                and block_start_idx <= block_end_idx
                and block_start_idx < len(col_positions)
            ):
                label_block(
                    ax,
                    col_positions,
                    block_start_idx,
                    block_end_idx,
                    annotation_y,
                    height,
                    last_value,
                    ann_config,
                    fontsize,
                    cell_aspect,
                )
            block_start_idx = None
        if value != last_value and x_value is not None:
            block_start_idx = i
        last_value = value


def label_block(
    ax,
    positions,
    start_idx,
    end_idx,
    annotation_y,
    height,
    value,
    ann_config,
    fontsize,
    cell_aspect=1.0,
):
    """
    Place a centered label over a contiguous block of columns.

    Args:
       ax: Matplotlib Axes to draw into.
       positions: Numeric column positions.
       start_idx: Start index of the block (inclusive).
       end_idx: End index of the block (inclusive).
       annotation_y: Y coordinate for the annotation track.
       height: Height of the annotation track.
       value: Value/text to display.
       ann_config: Annotation config (used for text color/fontsize).
       fontsize: Font size fallback if ann_config.label_fontsize is not set.
       cell_aspect: Cell width multiplier.

    Returns:
       None (text is drawn onto `ax`).
    """

    block_cols = positions[start_idx : end_idx + 1]
    x_center = (min(block_cols) + max(block_cols) + cell_aspect) / 2
    text_color = (ann_config.label_text_colors or {}).get(str(value), "white")
    ax.text(
        x_center,
        annotation_y + height / 2,
        str(value),
        ha="center",
        va="center",
        fontsize=ann_config.label_fontsize or fontsize,
        color=text_color,
        fontweight="bold",
        clip_on=False,
        zorder=100,
    )


def create_custom_legend_patch(
    color: str,
    shape: str,
    draw_border: bool = False,
    border_color: str = "black",
    border_width: float = 0.5,
) -> mpatches.Patch:
    """
    Create a legend patch shape used for custom legend entries.

    Args:
       color: Fill color for the legend patch.
       shape: One of 'upper_right', 'bottom_left', or 'rect' for full rectangle.
       draw_border: If True, draw an edge around the patch.
       border_color: Edge color to use when drawing border.
       border_width: Edge linewidth.

    Returns:
       A matplotlib patch instance appropriate for legend display.
    """

    border_args = {}
    if draw_border or is_white_color(color):
        border_args = {"edgecolor": border_color, "linewidth": border_width}
    if shape == "upper_right":
        return mpatches.Polygon([(0, 0), (1, 0), (1, 1)], color=color, closed=True, **border_args)
    if shape == "bottom_left":
        return mpatches.Polygon([(0, 0), (0, 1), (1, 0)], color=color, closed=True, **border_args)
    return mpatches.Rectangle((0, 0), 1, 1, color=color, **border_args)


class OncoPlotter:
    """
    High-level oncoplot drawing helper.

    Construct with mutation/event `df` and an `OncoplotConfig`. The `plot()` method
    renders a full oncoprint including heatmap cell fills, top annotations,
    legends, and optional row-group bars. Instances keep temporary references to
    created patches/texts to allow post-render adjustments (e.g., shifting labels).
    """

    def shift_row_group_bars_and_labels(
        self,
        ax,
        row_groups,
        bar_shift=-6,
        label_shift=-5.5,
        bar_shift_points=-240.0,
        label_shift_points=-220.0,
        use_points=True,
        bar_width: float | None = None,
        bar_width_points: float | None = None,
    ) -> None:
        """
        Shift row-group bar patches and their labels horizontally to avoid overlaps.

        This adjusts previously-stored `_row_group_bar_patches` and
        `_row_group_label_texts` positions, or finds matching patches/texts in
        the axes when not tracked. Coordinates can be supplied in data units or
        physical points; when `use_points` is True the provided point offsets
        are converted to data units for consistent visual spacing across figure
        sizes.

        Args:
           ax: Matplotlib Axes containing the row-group bars/labels.
           row_groups: DataFrame mapping genes/features to group/pathway names.
           bar_shift: Data-unit fallback horizontal shift for bars.
           label_shift: Data-unit fallback horizontal shift for labels.
           bar_shift_points: Horizontal shift for bars expressed in points (preferred).
           label_shift_points: Horizontal shift for labels expressed in points.
           use_points: If True convert the `_points` offsets to data units.
        """
        fig = ax.figure
        # Preserve current limits to avoid autoscale expanding the layout when users
        # call this post-plot (common in older usage). This keeps spacing stable.
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        autoscale_state = ax.get_autoscale_on()
        ax.set_autoscale_on(False)
        # Compute data-unit shifts from point targets using the x-scale so physical spacing stays constant.
        d0 = ax.transData.transform((0.0, 0.0))
        d1 = ax.transData.transform((1.0, 0.0))
        data_dx_px = max(abs(d1[0] - d0[0]), 1e-6)
        pts_per_data_unit_x = data_dx_px / (fig.dpi / 72.0)
        if use_points:
            bar_shift_data = bar_shift_points / pts_per_data_unit_x
            label_shift_data = label_shift_points / pts_per_data_unit_x
        else:
            bar_shift_data = bar_shift
            label_shift_data = label_shift

        # Compute optional bar width in data units if caller provided it.
        # Treat `bar_width` as a points-based alias for user convenience.
        bar_width_data = None
        if bar_width is not None or bar_width_points is not None:
            try:
                # Prefer explicit points arg; otherwise interpret `bar_width` as points.
                bw_pts = bar_width_points if bar_width_points is not None else float(bar_width)
                bar_width_data = bw_pts / pts_per_data_unit_x
            except Exception:
                bar_width_data = None

        referenced_bars = getattr(self, "_row_group_bar_patches", [])
        referenced_labels = getattr(self, "_row_group_label_texts", [])
        for patch in referenced_bars:
            patch.set_x(patch.get_x() + bar_shift_data)
            if bar_width_data is not None:
                try:
                    patch.set_width(bar_width_data)
                except Exception:
                    pass
        if not referenced_bars:
            for patch in ax.patches:
                if hasattr(patch, "_is_row_group_bar") and patch._is_row_group_bar:
                    patch.set_x(patch.get_x() + bar_shift_data)
                    if bar_width_data is not None:
                        try:
                            patch.set_width(bar_width_data)
                        except Exception:
                            pass
        for txt in referenced_labels:
            txt.set_x(txt.get_position()[0] + label_shift_data)
        if not referenced_labels and row_groups is not None:
            pathway_names = set(row_groups[self.row_group_col].unique())
            for txt in ax.texts:
                if hasattr(txt, "_is_row_label") and txt._is_row_label:
                    continue
                if (
                    hasattr(txt, "_is_row_group_label") and txt._is_row_group_label
                ) or txt.get_text() in pathway_names:
                    txt.set_x(txt.get_position()[0] + label_shift_data)
        leftmost_x = float("inf")
        for txt in referenced_labels:
            try:
                bbox = txt.get_window_extent().transformed(ax.transData.inverted())
                if bbox.x0 < leftmost_x:
                    leftmost_x = bbox.x0
            except Exception:
                continue
        if leftmost_x == float("inf"):
            for txt in ax.texts:
                if hasattr(txt, "_is_row_group_label") and txt._is_row_group_label:
                    try:
                        bbox = txt.get_window_extent().transformed(ax.transData.inverted())
                        if bbox.x0 < leftmost_x:
                            leftmost_x = bbox.x0
                    except Exception:
                        continue

        target_xlim = current_xlim
        if leftmost_x < current_xlim[0] and leftmost_x != float("inf"):
            padding = abs(current_xlim[0] - leftmost_x) + 1
            target_xlim = (current_xlim[0] - padding, current_xlim[1])

        ax.set_xlim(target_xlim)
        ax.set_ylim(current_ylim)
        ax.set_autoscale_on(autoscale_state)
        ax.figure.canvas.draw_idle()

    def move_row_group_labels(self, ax, new_bar_x, bar_width=None) -> None:
        """
        Move row-group bar patches and associated labels to a new X coordinate.

        Args:
           ax: Matplotlib Axes containing the bar patches and labels.
           new_bar_x: New X coordinate to place row-group bars.
           bar_width: Optional new width to set on bar patches.
        """
        current_xlim = ax.get_xlim()
        referenced_bars = getattr(self, "_row_group_bar_patches", [])
        referenced_labels = getattr(self, "_row_group_label_texts", [])
        for patch in referenced_bars:
            patch.set_x(new_bar_x)
            if bar_width is not None:
                patch.set_width(bar_width)
        for txt in referenced_labels:
            txt.set_x(new_bar_x)
        if not referenced_bars:
            for patch in ax.patches:
                if hasattr(patch, "_is_row_group_bar") and patch._is_row_group_bar:
                    patch.set_x(new_bar_x)
                    if bar_width is not None:
                        patch.set_width(bar_width)
        if not referenced_labels:
            for txt in ax.texts:
                if hasattr(txt, "_is_row_label") and txt._is_row_label:
                    continue
                if hasattr(txt, "_is_row_group_label") and txt._is_row_group_label:
                    txt.set_x(new_bar_x)
        leftmost_text_x = float("inf")
        text_offset = -0.2
        for txt in referenced_labels or []:
            x_pos = new_bar_x + text_offset
            if x_pos < leftmost_text_x:
                leftmost_text_x = x_pos
        if leftmost_text_x == float("inf"):
            for txt in ax.texts:
                if hasattr(txt, "_is_row_group_label") and txt._is_row_group_label:
                    try:
                        bbox = txt.get_window_extent().transformed(ax.transData.inverted())
                        if bbox.x0 < leftmost_text_x:
                            leftmost_text_x = bbox.x0
                    except Exception:
                        x_pos = new_bar_x + text_offset
                        if x_pos < leftmost_text_x:
                            leftmost_text_x = x_pos
        if leftmost_text_x < current_xlim[0] and leftmost_text_x != float("inf"):
            padding = abs(current_xlim[0] - leftmost_text_x) + 1
            ax.set_xlim(current_xlim[0] - padding, current_xlim[1])
        ax.figure.canvas.draw_idle()

    @staticmethod
    def redraw_row_group_labels(
        ax,
        row_groups,
        row_groups_color_dict,
        gene_to_idx,
        row_group_col,
        row_group_label_fontsize,
        rotate_left_annotation_label,
        bar_x,
        bar_width,
    ) -> None:
        # If no row_group_col provided or no row_groups, nothing to draw.
        if not row_group_col or row_groups is None:
            return
        """
        Draw row-group bars and labels on the left side of the oncoplot.

        This static helper will add rectangular colored bars corresponding to
        `row_groups` and place a rotated or unrotated label for each group.

        Args:
           ax: Matplotlib Axes to draw onto.
           row_groups: DataFrame with an index of gene names and a column
              `row_group_col` specifying group membership.
           row_groups_color_dict: Mapping of group name -> color for the bar.
           gene_to_idx: Mapping of gene name -> y-position index.
           row_group_col: Name of the column in `row_groups` holding group names.
           row_group_label_fontsize: Font size to use for group labels.
           rotate_left_annotation_label: If True rotate label text 90 degrees.
           bar_x: X coordinate for the left edge of the bars.
           bar_width: Width of the bar patches.
        """
        if (
            isinstance(row_groups, pd.DataFrame)
            and not row_groups.empty
            and row_group_col in row_groups.columns
        ):
            for pathway in row_groups[row_group_col].unique():
                color = row_groups_color_dict.get(pathway, "black")
                genes_in_group = (
                    row_groups[row_group_col == pathway].index.tolist()
                    if False
                    else row_groups[row_groups[row_group_col] == pathway].index.tolist()
                )
                genes_in_group = (
                    row_groups[row_group_col == pathway].index.tolist()
                    if False
                    else row_groups[row_groups[row_group_col] == pathway].index.tolist()
                )
                genes_in_group = row_groups[row_groups[row_group_col] == pathway].index.tolist()
                y_positions = [gene_to_idx[g] for g in genes_in_group if g in gene_to_idx]
                if not y_positions:
                    continue
                y_start, y_end = min(y_positions), max(y_positions)
                bar_height = y_end - y_start + 1
                bar_patch = mpatches.Rectangle(
                    (bar_x, y_start),
                    bar_width,
                    bar_height,
                    color=color,
                    clip_on=False,
                    zorder=5,
                )
                bar_patch._is_row_group_bar = True
                ax.add_patch(bar_patch)

                label_text = ax.text(
                    bar_x - 0.2,
                    (y_start + y_end) / 2 + 0.5,
                    pathway,
                    ha="right",
                    va="center",
                    fontsize=row_group_label_fontsize,
                    color=color,
                    clip_on=False,
                    rotation=90 if rotate_left_annotation_label else 0,
                )
                label_text._is_row_group_label = True
        ax.invert_yaxis()

    def get_dynamic_bar_x(ax, bar_offset, cell_aspect) -> Any:
        """
        Compute a dynamic X coordinate for row-group bars based on existing left-side text extents.

        Args:
            ax: Matplotlib Axes to inspect for left-aligned text items.
            bar_offset: Base offset to apply from the leftmost text position.
            cell_aspect: Cell width multiplier affecting scaled offset.

        Returns:
            Computed X coordinate where a row-group bar should be placed.
        """
        leftmost_x = float("inf")
        for text in ax.texts:
            if text.get_ha() == "right" and text.get_va() == "center":
                try:
                    bbox = text.get_window_extent().transformed(ax.transData.inverted())
                    if bbox.x0 < leftmost_x:
                        leftmost_x = bbox.x0
                except Exception:
                    continue
        if leftmost_x == float("inf"):
            leftmost_x = -0.3
        if cell_aspect < 1:
            scaled_offset = bar_offset * cell_aspect
        elif cell_aspect > 1:
            scaling_factor = min(cell_aspect, 1.5)
            scaled_offset = bar_offset * scaling_factor
        else:
            scaled_offset = bar_offset
        return leftmost_x + scaled_offset

    def __init__(
        self,
        df: pd.DataFrame,
        config: OncoplotConfig,
        row_groups: pd.DataFrame | None = None,
        row_groups_color_dict: dict | None = None,
        style: StyleBase | None = None,
    ) -> None:
        """
        Initialize an OncoPlotter.

        Args:
           df: DataFrame containing mutation/event records with columns matching
              `config.x_col`, `config.y_col`, and `config.value_col`.
           config: `OncoplotConfig` controlling layout, annotation defaults, and rendering options.
           row_groups: Optional DataFrame mapping features/genes to group names (index=gene).
           row_groups_color_dict: Optional mapping of group -> color for group bars.
           style: Optional `StyleBase` instance; when omitted `DefaultStyle()` is applied.
        """

        self.df = df
        self.row_groups = row_groups
        self.row_groups_color_dict = row_groups_color_dict
        self.config = config

        self._row_group_bar_patches = []
        self._row_group_label_texts = []

        # Validate required logical columns up front to avoid inscrutable KeyErrors later.
        # Allow callers to omit `row_group_col` when they don't want row-group bars/labels.
        required_fields = {
            "x_col": config.x_col,
            "y_col": config.y_col,
            "value_col": config.value_col,
        }
        missing_fields = [name for name, val in required_fields.items() if not val]
        if missing_fields:
            raise ValueError(
                "OncoplotConfig must set x_col (patient/sample ID), y_col (feature/gene), "
                "and value_col (mutation/value type). Missing: " + ", ".join(missing_fields)
            )

        # Normalize `row_group_col`: treat empty string/falsey as None so callers
        # may omit it to disable all row-group related drawing logic.
        self.row_group_col = (
            config.row_group_col if getattr(config, "row_group_col", None) else None
        )

        # Track whether the plotter should assemble row-groups. If the caller did
        # not provide a `row_group_col` (None) or did not supply a `row_groups`
        # mapping, we will skip row-group assembly entirely (no dummy columns
        # injected) and avoid drawing bars/labels.
        self._has_row_groups = False
        if self.row_group_col is not None and row_groups is not None:
            self._has_row_groups = True
        elif self.row_group_col is not None and self.row_group_col in self.df.columns:
            self._has_row_groups = True

        self.col_split_by = config.col_split_by
        self.col_split_order = config.col_split_order
        # If caller leaves col_sort_by empty, default to x_col so sort_values has a key.
        self.col_sort_by = config.col_sort_by or [config.x_col]
        self.x_col = config.x_col
        self.y_col = config.y_col
        self.figsize = config.figsize
        self.cell_aspect = config.cell_aspect
        self.top_annotations = config.top_annotations
        self.top_annotation_inter_spacer = config.top_annotation_inter_spacer
        self.top_annotation_intra_spacer = config.top_annotation_intra_spacer
        self.top_annotation_label_offset = getattr(config, "top_annotation_label_offset", 0.3)
        self.top_annotation_label_offset_points = getattr(
            config, "top_annotation_label_offset_points", 12.0
        )
        self.top_annotation_label_use_points = getattr(
            config, "top_annotation_label_use_points", True
        )
        self.col_split_gap = config.col_split_gap
        self.row_split_gap = config.row_split_gap
        self.bar_width = config.bar_width
        self.bar_width_points = getattr(config, "bar_width_points", 8.0)
        self.bar_width_use_points = getattr(config, "bar_width_use_points", True)
        self.bar_offset = config.bar_offset
        self.bar_buffer = config.bar_buffer
        self.bar_offset_use_points = getattr(config, "bar_offset_use_points", True)
        self.row_group_label_gap_use_points = getattr(
            config, "row_group_label_gap_use_points", True
        )
        self.row_group_label_fontsize = config.row_group_label_fontsize
        self.row_group_label_gap = getattr(config, "row_group_label_gap", 1.0)
        self.row_label_fontsize = config.row_label_fontsize
        self.column_label_fontsize = config.column_label_fontsize
        self.legend_fontsize = config.legend_fontsize
        self.legend_title_fontsize = config.legend_title_fontsize
        self.rotate_left_annotation_label = config.rotate_left_annotation_label
        self.legend_category_order = config.legend_category_order
        self.xticklabel_xoffset = config.xticklabel_xoffset
        self.xticklabel_yoffset = config.xticklabel_yoffset
        self.rowlabel_xoffset = getattr(config, "rowlabel_xoffset", -0.3)
        self.rowlabel_use_points = getattr(config, "rowlabel_use_points", True)
        self.legend_bbox_to_anchor = config.legend_bbox_to_anchor
        self.legend_offset = config.legend_offset
        self.legend_offset_points = getattr(config, "legend_offset_points", 18.0)
        self.legend_offset_use_points = getattr(config, "legend_offset_use_points", True)
        self.row_group_post_bar_shift = getattr(config, "row_group_post_bar_shift", -5.5)
        self.row_group_post_label_shift = getattr(config, "row_group_post_label_shift", -5.0)
        self.row_group_post_bar_shift_points = getattr(
            config, "row_group_post_bar_shift_points", -240.0
        )
        self.row_group_post_label_shift_points = getattr(
            config, "row_group_post_label_shift_points", -220.0
        )
        self.row_group_post_shift_use_points = getattr(
            config, "row_group_post_shift_use_points", True
        )
        self.fig_top_margin = config.fig_top_margin
        self.fig_bottom_margin = config.fig_bottom_margin
        self.fig_y_margin = config.fig_y_margin

        # Detect which config fields were explicitly set by the caller (pydantic)
        fields_set = getattr(
            config, "model_fields_set", getattr(config, "__pydantic_fields_set__", set())
        )

        # Treat `bar_width` as a shorthand alias for `bar_width_points` when the
        # user provided it explicitly and did not provide `bar_width_points`.
        # This makes `bar_width` consistently behave as a point-based width.
        if "bar_width" in fields_set and "bar_width_points" not in fields_set:
            try:
                self.bar_width_points = float(self.bar_width)
            except Exception:
                # Ignore invalid conversion; leave defaults in place
                pass

        # Apply provided style or default
        self.style = style or DefaultStyle()
        try:
            self.style.apply_theme()
        except Exception:
            # Style application is best-effort
            pass

        if config.heatmap_annotation is None:
            if config.row_values_color_dict is None:
                raise ValueError(
                    "Either heatmap_annotation or row_values_color_dict must be provided"
                )
            # Build a HeatmapAnnotationConfig from the oncoplot-level defaults.
            self.heatmap_annotation = HeatmapAnnotationConfig(
                values=config.value_col,
                colors=config.row_values_color_dict,
                legend_title=(
                    config.value_legend_title if config.value_legend_title else config.value_col
                ),
                bottom_left_triangle_values=getattr(
                    config, "heatmap_bottom_left_triangle_values", ["SNV"]
                ),
                upper_right_triangle_values=getattr(
                    config, "heatmap_upper_right_triangle_values", ["CNV"]
                ),
            )
        else:
            self.heatmap_annotation = config.heatmap_annotation

        # If the user supplied a HeatmapAnnotationConfig but left `colors` empty,
        # fall back to the oncoplot-level `row_values_color_dict` so defaults
        # from `OncoplotConfig` apply as expected.
        if getattr(self.heatmap_annotation, "colors", None) in (None, {}) and getattr(
            config, "row_values_color_dict", None
        ):
            self.heatmap_annotation.colors = config.row_values_color_dict

        if isinstance(self.heatmap_annotation.values, str):
            self.value_col = self.heatmap_annotation.values
        else:
            self.value_col = "__value__"

    def plot(self) -> plt.Figure:
        """
        Render the oncoplot and return the produced matplotlib Figure.

        This method computes layout (column positions, row ordering, cell sizing),
        draws background cells, heatmap/triangle fills, top annotations, mutation
        glyphs, legends, and optional row-group bars/labels according to the
        provided `OncoplotConfig` and supporting config objects.

        Returns:
           A matplotlib `Figure` instance containing the completed oncoplot.
        """
        df = self.df
        row_groups = self.row_groups
        row_groups_color_dict = self.row_groups_color_dict
        config = self.config
        col_split_by = self.col_split_by
        col_split_order = self.col_split_order
        col_sort_by = self.col_sort_by
        x_col = self.x_col
        y_col = self.y_col
        row_group_col = self.row_group_col
        figsize = self.figsize
        cell_aspect = self.cell_aspect
        top_annotations = self.top_annotations
        top_annotation_inter_spacer = self.top_annotation_inter_spacer
        top_annotation_intra_spacer = self.top_annotation_intra_spacer
        col_split_gap = self.col_split_gap
        row_split_gap = self.row_split_gap
        bar_offset = self.bar_offset
        bar_buffer = self.bar_buffer
        row_group_label_fontsize = self.row_group_label_fontsize
        row_label_fontsize = self.row_label_fontsize
        column_label_fontsize = self.column_label_fontsize
        legend_fontsize = self.legend_fontsize
        legend_title_fontsize = self.legend_title_fontsize
        rotate_left_annotation_label = self.rotate_left_annotation_label
        # use self.legend_category_order directly where needed
        heatmap_annotation = self.heatmap_annotation
        value_col = self.value_col
        fig_title = getattr(config, "fig_title", None)
        fig_title_fontsize = getattr(config, "fig_title_fontsize", 22)

        # Normalize split columns to a single value per x-axis value so splits do not duplicate entries
        if col_split_by:
            split_cols = [c for c in col_split_by if c in df.columns]
            if split_cols:
                split_map = (
                    df[[x_col] + split_cols]
                    .drop_duplicates()
                    .groupby(x_col)
                    .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA)
                )
                for col in split_cols:
                    df[col] = df[x_col].map(split_map[col])

        x_values = self._get_split_x_values(df, col_split_by, col_split_order, x_col, col_sort_by)

        col_positions = []
        pos = 0.0
        last_split_vals = None
        for x_val in x_values:
            split_vals = tuple(df.loc[df[x_col] == x_val, col].iloc[0] for col in col_split_by)
            if last_split_vals is not None:
                for i, (prev, curr) in enumerate(zip(last_split_vals, split_vals)):
                    if prev != curr:
                        # Scale split gaps with cell width so the gap-to-cell ratio stays consistent across aspects.
                        pos += col_split_gap * cell_aspect
                        break
            col_positions.append(pos)
            pos += cell_aspect
            last_split_vals = split_vals
        ncols = int(np.ceil(pos))

        split_maps = []
        if col_split_by:
            for i in range(1, len(col_split_by) + 1):
                split_cols = col_split_by[:i]
                split_map = df.set_index(x_col)[split_cols]
                if len(split_cols) == 1:
                    split_map = split_map[split_cols[0]]
                split_maps.append(split_map)
        else:
            split_maps = [None]

        genes_ordered, row_group_positions, row_positions = [], [], []
        pos, last_pathway = 0.0, None
        if self._has_row_groups and (
            row_groups is not None
            and isinstance(row_groups, pd.DataFrame)
            and not row_groups.empty
            and row_group_col in row_groups.columns
        ):
            group_values = row_groups[row_group_col].unique().tolist()
            custom_order = getattr(config, "row_group_order", None)
            if custom_order:
                seen = set()
                ordered = []
                for g in custom_order:
                    if g in group_values and g not in seen:
                        ordered.append(g)
                        seen.add(g)
                remaining = sorted([g for g in group_values if g not in seen])
                group_values = ordered + remaining

            for pathway in group_values:
                genes_in_group = row_groups[row_groups[row_group_col] == pathway].index.tolist()
                if last_pathway is not None:
                    pos += row_split_gap
                for gene in genes_in_group:
                    genes_ordered.append(gene)
                    row_group_positions.append(pathway)
                    row_positions.append(pos)
                    pos += 1.0
                last_pathway = pathway
            # Append any genes present in the dataframe but missing from row_groups
            missing_genes = [
                g for g in df[y_col].drop_duplicates().tolist() if g not in genes_ordered
            ]
            if missing_genes and genes_ordered:
                pos += row_split_gap
            for gene in missing_genes:
                genes_ordered.append(gene)
                row_group_positions.append(None)
                row_positions.append(pos)
                pos += 1.0
        else:
            unique_genes = df[y_col].drop_duplicates().tolist()
            for gene in unique_genes:
                genes_ordered.append(gene)
                row_positions.append(pos)
                pos += 1.0
        nrows = int(np.ceil(pos))
        gene_to_idx = {g: i for g, i in zip(genes_ordered, row_positions)}
        x_value_to_idx = {x_val: i for x_val, i in zip(x_values, col_positions)}

        auto_adjust = getattr(config, "auto_adjust_cell_size", False)

        # Default ratios in case auto_adjust is disabled
        cell_height_ratio = 1.0

        # If requested, auto-compute the figure size so each data cell is close to
        # `target_cell_width` x `target_cell_height` inches without having to pass
        # an explicit `figsize`. Padding accounts for row labels, row-group bars,
        # legends, and top annotations.
        if auto_adjust:
            cell_w = float(getattr(config, "target_cell_width", 1.5))
            cell_h = float(getattr(config, "target_cell_height", 1.5))

            # Keep logical cell geometry stable; aspect is applied on the axes.
            base_cell_aspect = float(getattr(config, "cell_aspect", 1.0) or 1.0)
            cell_aspect = base_cell_aspect

            # Approximate how much horizontal room row labels and row-group bars need.
            longest_row_label = max((len(str(g)) for g in genes_ordered), default=0)
            approx_char_width = 0.55 * row_label_fontsize / 72.0  # crude inches/char
            label_block_in = longest_row_label * approx_char_width
            bar_padding = max(0.0, abs(bar_offset) * 0.2) + bar_buffer
            post_shift_padding = 0.0
            if getattr(config, "apply_post_row_group_shift", False):
                post_shift_padding = abs(getattr(config, "row_group_post_label_shift", 0.0)) * 0.1
            left_padding_in = 0.6 + label_block_in + bar_padding + post_shift_padding
            right_padding_in = 0.8  # leave room for the legend gutter

            # Account for stacked top annotations, their spacers, and an optional title.
            top_annotations = top_annotations or {}
            num_top = len(top_annotations)
            top_blocks_in = sum(getattr(cfg, "height", 0.45) for cfg in top_annotations.values())
            top_spacers_in = 0.0
            if num_top:
                top_spacers_in += (num_top - 1) * top_annotation_intra_spacer
                top_spacers_in += top_annotation_inter_spacer
            title_padding_in = (fig_title_fontsize / 72.0) * 1.1 if fig_title else 0.0
            top_padding_in = 0.35 + top_blocks_in + top_spacers_in + title_padding_in

            # Leave space for rotated xtick labels and bottom gutter.
            bottom_padding_in = max(0.4, (column_label_fontsize / 72.0) * 1.6)

            fig_w = max(1.0, ncols * cell_w + left_padding_in + right_padding_in)
            fig_h = max(1.0, nrows * cell_h + top_padding_in + bottom_padding_in)
            # After sizing, apply aspect by shrinking/expanding width, then rescale to keep height.
            width_aspected = fig_w
            height_target = fig_h
            figsize = (width_aspected, height_target)

            # Derive proportional offsets from the effective cell size to keep spacing consistent
            # across aspect changes. Clamp to avoid extreme values on huge/small plots.
            cell_height_ratio = cell_h / getattr(config, "target_cell_height", cell_h)
            cell_height_ratio = max(0.5, min(cell_height_ratio, 2.5))

            fields_set = getattr(
                config,
                "model_fields_set",
                getattr(config, "__pydantic_fields_set__", set()),
            )
            spacing_aspect_scale = float(getattr(config, "spacing_aspect_scale", 0.0))
            xtick_aspect_scale = float(getattr(config, "xtick_aspect_scale", 0.0))
            # If the user did not set these, keep spacing/xticks stable across aspect changes.
            if "spacing_aspect_scale" not in fields_set:
                spacing_aspect_scale = 0.0
            if "xtick_aspect_scale" not in fields_set:
                xtick_aspect_scale = 0.0
            # Pure aspect scaling; users can turn it off via *_aspect_scale
            spacing_scale = (cell_aspect**spacing_aspect_scale) if spacing_aspect_scale else 1.0
            xtick_scale = (cell_aspect**xtick_aspect_scale) if xtick_aspect_scale else 1.0
            if "xticklabel_yoffset" not in fields_set:
                # If xticks use points, leave the value as-is (already interpreted as points);
                # otherwise scale the data-unit offset with cell height/aspect.
                if not getattr(config, "xticklabel_use_points", False):
                    scaled = self.xticklabel_yoffset * cell_height_ratio * xtick_scale
                    self.xticklabel_yoffset = max(0.1, scaled)
            if "bar_buffer" not in fields_set:
                self.bar_buffer = self.bar_buffer * spacing_scale
            if "bar_offset" not in fields_set:
                self.bar_offset = self.bar_offset * spacing_scale
            if "row_group_label_gap" not in fields_set:
                self.row_group_label_gap = self.row_group_label_gap * spacing_scale

        # Refresh spacing values after potential auto scaling to honor user inputs
        bar_offset = self.bar_offset
        bar_buffer = self.bar_buffer

        fig, ax = plt.subplots(figsize=figsize)
        fig_top_margin = self.fig_top_margin
        if not (
            row_groups is not None
            and isinstance(row_groups, pd.DataFrame)
            and not row_groups.empty
            and row_group_col in row_groups.columns
        ):
            fig_top_margin = max(0.0, fig_top_margin - 0.02)
        bottom_margin = self.fig_bottom_margin
        aspect_ratio = getattr(self.config, "aspect", 1.0)
        if aspect_ratio < 1.0 and nrows <= 10:
            bottom_margin = max(
                self.fig_bottom_margin, self.fig_bottom_margin + (10 - nrows) * 0.02
            )
        else:
            bottom_margin = self.fig_bottom_margin
        # Use standard subplot margins; auto-adjust only determines figsize so spacing
        # stays closer to the manual helper look without requiring a user-provided
        # figsize.
        fig.subplots_adjust(
            top=fig_top_margin,
            bottom=bottom_margin,
        )
        set_title_later = False
        if fig_title:
            set_title_later = True
        max_x = (max(col_positions) + cell_aspect) if col_positions else 0
        ax.set_xlim(-1, max(ncols, max_x))
        ax.set_ylim(-1, nrows)
        # Use configured aspect on the axes; cell geometry stays constant.
        ax.set_aspect(getattr(self.config, "aspect", 1.0) or 1.0)
        # Keep the axes facecolor opaque for correct cell rendering
        ax.set_facecolor("white")

        # Figure patch: configurable by OncoplotConfig. Use an explicit
        # figure_facecolor when provided; otherwise default to white.
        fig_face = getattr(config, "figure_facecolor", None)
        if fig_face is not None:
            fig.patch.set_facecolor(fig_face)
        else:
            fig.patch.set_facecolor("white")

        # If requested, make the figure background transparent while leaving
        # the axes background opaque (so cell fills remain visible).
        if getattr(config, "figure_transparent", False):
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_alpha(1.0)

        bottom_left_values = getattr(heatmap_annotation, "bottom_left_triangle_values", ["SNV"])
        upper_right_values = getattr(heatmap_annotation, "upper_right_triangle_values", ["CNV"])

        # Precompute row-label transform so top-annotation labels can align with row labels.
        rowlabel_use_points = bool(self.rowlabel_use_points)
        xlim_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-6)
        pts_per_data_unit_x = (fig.get_figwidth() * 72.0) / xlim_span
        if rowlabel_use_points:
            rowlabel_offset_pts = float(self.rowlabel_xoffset) * pts_per_data_unit_x
            rowlabel_translate = mtransforms.ScaledTranslation(
                rowlabel_offset_pts / 72.0, 0.0, fig.dpi_scale_trans
            )
            rowlabel_text_transform = ax.transData + rowlabel_translate
            rowlabel_base_x = 0.0
        else:
            rowlabel_text_transform = ax.transData
            rowlabel_base_x = float(self.rowlabel_xoffset)

        # Draw an opaque white background for every cell so empty cells
        # are filled (important for transparent PNG exports).
        bg_color = _ensure_opaque_color(None, default="white")
        for y in row_positions:
            for x in col_positions:
                bg_rect = mpatches.Rectangle(
                    (x, y),
                    cell_aspect,
                    1,
                    color=bg_color,
                    linewidth=0,
                    zorder=0,
                )
                ax.add_patch(bg_rect)

        for _, row in df.iterrows():
            gene, x_value = row.get(y_col), row.get(x_col)
            if gene in gene_to_idx and x_value in x_value_to_idx:
                if isinstance(heatmap_annotation.values, str):
                    value = row.get(heatmap_annotation.values)
                else:
                    try:
                        value = heatmap_annotation.values.get(x_value)
                    except Exception:
                        value = None
                color = heatmap_annotation.colors.get(value, "white")
                my_shape_func(
                    ax,
                    x_value_to_idx[x_value],
                    gene_to_idx[gene],
                    cell_aspect,
                    1,
                    color,
                    value,
                    bottom_left_values,
                    upper_right_values,
                )

        mutation_groups = {}
        for _, row in df.iterrows():
            gene, x_value = row.get(y_col), row.get(x_col)
            value = row.get(value_col)
            if gene in gene_to_idx and x_value in x_value_to_idx:
                key = (gene, x_value)
                if key not in mutation_groups:
                    mutation_groups[key] = []
                mutation_groups[key].append(value)

        for (gene, x_value), values in mutation_groups.items():
            x, y = x_value_to_idx[x_value], gene_to_idx[gene]
            for bl_value in bottom_left_values:
                if bl_value in values:
                    diagonal_fill(
                        ax,
                        x,
                        y,
                        cell_aspect,
                        1,
                        heatmap_annotation.colors.get(bl_value),
                        which_half="bottom_left",
                    )
                    break
            for ur_value in upper_right_values:
                if ur_value in values:
                    diagonal_fill(
                        ax,
                        x,
                        y,
                        cell_aspect,
                        1,
                        heatmap_annotation.colors.get(ur_value),
                        which_half="upper_right",
                    )
                    break
            if not any(val in values for val in bottom_left_values + upper_right_values):
                for value in values:
                    color = heatmap_annotation.colors.get(value, "white")
                    face = _ensure_opaque_color(color, default="white")
                    rect = mpatches.Rectangle((x, y), cell_aspect, 1, color=face, linewidth=0)
                    ax.add_patch(rect)

        for y in row_positions:
            for x in col_positions:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x, y),
                        cell_aspect,
                        1,
                        fill=False,
                        edgecolor="black",
                        lw=1,
                        zorder=2,
                    )
                )

        ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

        if top_annotations:
            # Align top-annotation labels with the row-label offset for consistent visual spacing.
            heatmap_left = min(col_positions) if col_positions else 0.0
            if self.top_annotation_label_use_points:
                top_label_transform = rowlabel_text_transform
                top_label_x = rowlabel_base_x
            else:
                top_label_offset_data = float(self.top_annotation_label_offset)
                top_label_transform = ax.transData
                top_label_x = heatmap_left - top_label_offset_data

            annotation_y = top_annotation_inter_spacer * -1
            if config.top_annotation_order:
                annotation_order = [
                    name for name in config.top_annotation_order if name in top_annotations
                ][::-1]
                for name in top_annotations:
                    if name not in annotation_order:
                        annotation_order.append(name)
            else:
                annotation_order = list(top_annotations.keys())
            for ann_idx, ann_name in enumerate(annotation_order):
                ann_config = top_annotations[ann_name]
                if ann_config.fontsize is None:
                    ann_config.fontsize = 12
                split_level = (
                    col_split_by.index(ann_name) + 1
                    if ann_name in col_split_by
                    else len(split_maps)
                )
                col_split_map = split_maps[split_level - 1]
                draw_top_annotation(
                    ax,
                    x_values,
                    col_positions,
                    annotation_y,
                    ann_config,
                    ann_name,
                    col_split_map=col_split_map,
                    cell_aspect=cell_aspect,
                    label_x=top_label_x,
                    label_transform=top_label_transform,
                )
                annotation_y -= ann_config.height + top_annotation_intra_spacer

        legend_categories = {}
        heatmap_legend_title = heatmap_annotation.legend_title
        # mutation handles: do not include a title patch here, assembly will add it
        mutation_handles = []
        mutation_value_order = heatmap_annotation.legend_value_order
        remove_unused_keys = getattr(config, "remove_unused_keys_in_legend", False)
        if remove_unused_keys:
            if isinstance(heatmap_annotation.values, str):
                series = df[heatmap_annotation.values]
            else:
                series = pd.Series(heatmap_annotation.values)
            if pd.api.types.is_categorical_dtype(series):
                series = series.cat.remove_unused_categories()
            # Use stringified values for comparisons to avoid mismatches between
            # numeric and string representations (e.g., 100 vs '100 mg') coming
            # from upstream configs or pandas categorical categories.
            present_values = set(str(v) for v in series.dropna().unique())
            # Filter mutation_value_order to only include observed values (string compare)
            if mutation_value_order:
                mutation_value_order = [v for v in mutation_value_order if str(v) in present_values]
        else:
            present_values = set(heatmap_annotation.colors.keys())

        heatmap_draw_border = getattr(heatmap_annotation, "draw_border", False)
        heatmap_border_categories = getattr(heatmap_annotation, "border_categories", None)
        heatmap_border_color = getattr(heatmap_annotation, "border_color", "black")
        heatmap_border_width = getattr(heatmap_annotation, "border_width", 0.5)

        if mutation_value_order:
            for label in mutation_value_order:
                if label in heatmap_annotation.colors:
                    if not remove_unused_keys or label in present_values:
                        color = heatmap_annotation.colors[label]
                        if heatmap_border_categories is not None:
                            needs_border = label in heatmap_border_categories
                        elif heatmap_draw_border:
                            needs_border = True
                        else:
                            needs_border = is_white_color(color)

                        face = _ensure_opaque_color(color, default="white")
                        if needs_border:
                            mutation_handles.append(
                                Patch(
                                    facecolor=face,
                                    edgecolor=heatmap_border_color,
                                    linewidth=heatmap_border_width,
                                    label=label,
                                )
                            )
                        else:
                            mutation_handles.append(Patch(facecolor=face, label=label))
        else:
            for label, color in heatmap_annotation.colors.items():
                if not remove_unused_keys or label in present_values:
                    if heatmap_border_categories is not None:
                        needs_border = label in heatmap_border_categories
                    elif heatmap_draw_border:
                        needs_border = True
                    else:
                        needs_border = is_white_color(color)

                    face = _ensure_opaque_color(color, default="white")
                    if needs_border:
                        mutation_handles.append(
                            Patch(
                                facecolor=face,
                                edgecolor=heatmap_border_color,
                                linewidth=heatmap_border_width,
                                label=label,
                            )
                        )
                    else:
                        mutation_handles.append(Patch(facecolor=face, label=label))
        legend_categories[heatmap_legend_title] = mutation_handles
        if top_annotations:
            for ann_name, ann_config in top_annotations.items():
                legend_title = ann_config.legend_title or ann_name
                # annotation handles: do not include a title patch here, assembly will add it
                annotation_handles = []
                value_order = ann_config.legend_value_order or sorted(ann_config.colors.keys())

                ann_draw_border = getattr(ann_config, "draw_border", False)
                ann_border_categories = getattr(ann_config, "border_categories", None)
                ann_border_color = getattr(ann_config, "border_color", "black")
                ann_border_width = getattr(ann_config, "border_width", 0.5)

                if remove_unused_keys:
                    # Build a Series aligned to the plotted x_values (patient columns)
                    if isinstance(ann_config.values, str):
                        series = df[ann_config.values]
                    else:
                        series = pd.Series(ann_config.values)

                    # Collect only values that correspond to the plotted x_values so
                    # legends reflect the subset actually rendered (not the full-study map).
                    observed = []
                    for xv in x_values:
                        try:
                            v = series.get(xv)
                        except Exception:
                            # fallback: positional/unnamed series
                            v = None
                        # Ensure we check element-wise nullness
                        if not (isinstance(v, (list, tuple, pd.Series, pd.Index))):
                            if pd.notna(v):
                                observed.append(v)
                        else:
                            # If the retrieved value is an array-like, extend with its non-null members
                            for elem in v:
                                if pd.notna(elem):
                                    observed.append(elem)

                    # If series has categorical dtype, remove unused categories then keep observed as-is
                    try:
                        dtype = getattr(series, "dtype", None)
                        if isinstance(dtype, pd.CategoricalDtype):
                            try:
                                series = series.cat.remove_unused_categories()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    present_ann_values = set(str(v) for v in observed)
                    # Filter value_order to only include observed values (string compare)
                    value_order = [v for v in value_order if str(v) in present_ann_values]
                else:
                    present_ann_values = set(ann_config.colors.keys())

                for value in value_order:
                    if str(value) in ann_config.colors:
                        if not remove_unused_keys or str(value) in present_ann_values:
                            color = ann_config.colors[str(value)]
                            if ann_border_categories is not None:
                                needs_border = str(value) in ann_border_categories
                            elif ann_draw_border:
                                needs_border = True
                            else:
                                needs_border = is_white_color(color)

                            face = _ensure_opaque_color(color, default="white")
                            if needs_border:
                                annotation_handles.append(
                                    Patch(
                                        facecolor=face,
                                        edgecolor=ann_border_color,
                                        linewidth=ann_border_width,
                                        label=str(value),
                                    )
                                )
                            else:
                                annotation_handles.append(Patch(facecolor=face, label=str(value)))
                if remove_unused_keys:
                    if any(pd.isna(v) for v in observed):
                        annotation_handles.append(Patch(color=ann_config.na_color, label="NA"))
                else:
                    if hasattr(ann_config, "na_color") and ann_config.na_color is not None:
                        annotation_handles.append(Patch(color=ann_config.na_color, label="NA"))
                legend_categories[legend_title] = annotation_handles
        legend_handles = []

        def legend_label_patch(label):
            return Patch(color="none", label=label)

        # Build legend order and deduplicate
        if self.legend_category_order:
            legend_order = []
            seen = set()
            for cat in self.legend_category_order:
                if cat in legend_categories and cat not in seen:
                    legend_order.append(cat)
                    seen.add(cat)
        else:
            legend_order = []
            if heatmap_legend_title in legend_categories:
                legend_order.append(heatmap_legend_title)
            for ann_name, ann_config in (top_annotations or {}).items():
                legend_title = ann_config.legend_title or ann_name
                if (
                    legend_title != heatmap_legend_title
                    and legend_title in legend_categories
                    and legend_title not in legend_order
                ):
                    legend_order.append(legend_title)

        # Determine the heatmap legend label
        # Prefer explicit override, then the configured legend title, then the values column name.
        heatmap_legend_label = (
            getattr(self.config, "value_legend_title", None)
            or heatmap_legend_title
            or getattr(self.heatmap_annotation, "values", None)
        )

        legend_handles = []
        for idx, cat in enumerate(legend_order):
            # Add a label patch for each category
            if cat == heatmap_legend_title:
                legend_handles.append(legend_label_patch(heatmap_legend_label))
            else:
                legend_handles.append(legend_label_patch(cat))
            # Add the actual legend handles for this category
            legend_handles.extend(legend_categories[cat])
            # Add a spacer between categories, except after the last
            if idx < len(legend_order) - 1:
                legend_handles.append(Patch(color="none", label=""))

        fig.canvas.draw()

        legend_kwargs = {}
        if self.legend_bbox_to_anchor is not None:
            bbox_to_anchor = self.legend_bbox_to_anchor
        else:
            if self.legend_offset_use_points:
                offset_pts = float(self.legend_offset_points)
                translate = mtransforms.ScaledTranslation(
                    offset_pts / 72.0, 0.0, fig.dpi_scale_trans
                )
                legend_kwargs["bbox_transform"] = ax.transAxes + translate
                bbox_to_anchor = (1.0, 0.5)
            else:
                bbox_to_anchor = (1 + self.legend_offset, 0.5)
                legend_kwargs["bbox_transform"] = ax.transAxes

        gene_labels = []
        text_transform = rowlabel_text_transform
        base_x = rowlabel_base_x
        for y, g in zip(row_positions, genes_ordered):
            text = ax.text(
                base_x,
                y + 0.55,
                g,
                ha="right",
                va="center",
                fontsize=row_label_fontsize,
                clip_on=False,
                transform=text_transform,
            )
            text._is_row_label = True
            gene_labels.append(text)

        use_point_offset = getattr(self.config, "xticklabel_use_points", False)
        offset_val = float(self.xticklabel_yoffset)
        if use_point_offset:
            # Point-based offset: interpret xticklabel_yoffset directly as points.
            offset_pts = offset_val
            base_transform = ax.get_xaxis_transform()
            translate = mtransforms.ScaledTranslation(0, -offset_pts / 72.0, fig.dpi_scale_trans)
            xtick_transform = base_transform + translate
            for x, p in zip(col_positions, x_values):
                ax.text(
                    x + cell_aspect / 2,
                    0.0,
                    p,
                    ha="center",
                    va="top",
                    fontsize=column_label_fontsize,
                    rotation=90,
                    clip_on=False,
                    transform=xtick_transform,
                )
        else:
            # Data-unit offset: use raw offset.
            y_xtick = nrows + offset_val
            for x, p in zip(col_positions, x_values):
                ax.text(
                    x + cell_aspect / 2,
                    y_xtick,
                    p,
                    ha="center",
                    va="top",
                    fontsize=column_label_fontsize,
                    rotation=90,
                    clip_on=False,
                )

        # Ensure newly added text objects have up-to-date extents before measuring spacing
        fig.canvas.draw()

        legend_family = resolve_font_family()
        lgd = ax.legend(
            handles=legend_handles,
            bbox_to_anchor=bbox_to_anchor,
            loc="center left",
            frameon=False,
            handlelength=1,
            handleheight=1,
            prop=font_manager.FontProperties(family=legend_family, size=legend_fontsize),
            ncol=1,
            title_fontsize=legend_title_fontsize,
            **legend_kwargs,
        )
        if legend_family:
            lgd.get_title().set_fontproperties(
                font_manager.FontProperties(family=legend_family, size=legend_title_fontsize)
            )
        # Bold the injected header labels (heatmap header + annotation headers)
        bold_labels = {heatmap_legend_label}
        bold_labels.update(
            (ann_config.legend_title or n) for n, ann_config in (top_annotations or {}).items()
        )
        for text in lgd.get_texts():
            if text.get_text() in bold_labels:
                text.set_fontweight("bold")

        heatmap_left = min(col_positions) if col_positions else 0.0

        # Convert bar offsets/gaps to data units using point-based spacing by default.
        # Use the x-scale in display units so physical spacing stays constant as aspect changes.
        d0 = ax.transData.transform((0.0, 0.0))
        d1 = ax.transData.transform((1.0, 0.0))
        data_dx_px = max(abs(d1[0] - d0[0]), 1e-6)
        pts_per_data_unit_x = data_dx_px / (fig.dpi / 72.0)
        if self.bar_offset_use_points:
            total_offset_data = (bar_offset + bar_buffer) / pts_per_data_unit_x
        else:
            total_offset_data = bar_offset + bar_buffer
        if self.bar_width_use_points:
            bar_width_draw = self.bar_width_points / pts_per_data_unit_x
        else:
            bar_width_draw = self.bar_width
        if self.row_group_label_gap_use_points:
            label_gap_pts = float(self.row_group_label_gap)
            label_gap = label_gap_pts / pts_per_data_unit_x
        else:
            label_gap = max(self.row_group_label_gap, 0.0)

        # Anchor bar to the heatmap edge so aspect changes do not move it relative to the grid.
        bar_x_shift = heatmap_left - total_offset_data - bar_width_draw
        self._row_group_bar_patches.clear()
        self._row_group_label_texts.clear()
        # Only draw row-group bars/labels when the plotter was configured
        # with a `row_group_col` and provided `row_groups` mapping.
        if self._has_row_groups:
            for pathway in row_groups[row_group_col].unique():
                color = (
                    row_groups_color_dict.get(pathway, "black")
                    if row_groups_color_dict
                    else "black"
                )
                genes_in_group = row_groups[row_groups[row_group_col] == pathway].index.tolist()
                y_positions = [gene_to_idx[g] for g in genes_in_group if g in gene_to_idx]
                if not y_positions:
                    continue
                y_start, y_end = min(y_positions), max(y_positions)
                bar_height = y_end - y_start + 1
                if not rotate_left_annotation_label:
                    label_x_shift = bar_x_shift - label_gap
                    label_ha = "right"
                else:
                    label_x_shift = bar_x_shift - 0.2
                    label_ha = "right"
                bar_patch = mpatches.Rectangle(
                    (bar_x_shift, y_start),
                    bar_width_draw,
                    bar_height,
                    color=color,
                    clip_on=False,
                    zorder=5,
                )
                bar_patch._is_row_group_bar = True
                ax.add_patch(bar_patch)
                self._row_group_bar_patches.append(bar_patch)

                label_text = ax.text(
                    label_x_shift,
                    (y_start + y_end) / 2 + 0.5,
                    pathway,
                    ha=label_ha,
                    va="center",
                    fontsize=row_group_label_fontsize,
                    color=color,
                    clip_on=False,
                    rotation=90 if rotate_left_annotation_label else 0,
                )
                label_text._is_row_group_label = True
                self._row_group_label_texts.append(label_text)
        ax.invert_yaxis()

        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=self.config.row_label_fontsize,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            fontsize=self.config.column_label_fontsize,
        )

        ax.margins(y=self.fig_y_margin)
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        margin = self.fig_y_margin
        if margin > 0 and yrange > 0:
            ax.set_ylim(ymin, ymax + yrange * margin)

        if col_positions and row_positions:
            bottom_y = max(row_positions) + 1
            block_starts = [col_positions[0]]
            block_ends = []
            for i in range(1, len(col_positions)):
                if not np.isclose(col_positions[i] - col_positions[i - 1], cell_aspect):
                    block_ends.append(col_positions[i - 1] + cell_aspect)
                    block_starts.append(col_positions[i])
            block_ends.append(col_positions[-1] + cell_aspect)
            for x_start, x_end in zip(block_starts, block_ends):
                ax.hlines(
                    bottom_y,
                    xmin=x_start,
                    xmax=x_end,
                    colors="black",
                    linewidth=1,
                    zorder=10,
                    clip_on=False,
                )

        if col_positions and row_positions:
            right_x = max(col_positions) + cell_aspect
            block_starts = [row_positions[0]]
            block_ends = []
            for i in range(1, len(row_positions)):
                if not np.isclose(row_positions[i] - row_positions[i - 1], 1.0):
                    block_ends.append(row_positions[i - 1] + 1.0)
                    block_starts.append(row_positions[i])
            block_ends.append(row_positions[-1] + 1.0)
            for y_start, y_end in zip(block_starts, block_ends):
                ax.vlines(
                    right_x,
                    ymin=y_start,
                    ymax=y_end,
                    colors="black",
                    linewidth=1,
                    zorder=10,
                    clip_on=False,
                )

        if set_title_later:
            fig.suptitle(fig_title, fontsize=fig_title_fontsize)

        if getattr(self.config, "apply_post_row_group_shift", False):
            # Ensure text positions are finalized before measuring/shifting
            fig.canvas.draw()
            self.shift_row_group_bars_and_labels(
                ax,
                row_groups,
                self.row_group_post_bar_shift,
                self.row_group_post_label_shift,
                self.row_group_post_bar_shift_points,
                self.row_group_post_label_shift_points,
                self.row_group_post_shift_use_points,
            )
            fig.canvas.draw_idle()
        return fig

    def _get_split_x_values(
        self, df, col_split_by, col_split_order, x_col, col_sort_by
    ) -> Any | list:
        """
        Return an ordered list of x-axis values, respecting nested `col_split_by` ordering.

        Args:
           df: DataFrame to extract x values from.
           col_split_by: List of columns to split/partition columns by (ordered).
           col_split_order: Mapping column -> ordered list of values to prioritize.
           x_col: Name of the x column containing timepoints or categories.
           col_sort_by: Fallback sort keys for unsplit lists.

        Returns:
           Ordered list of x values for plotting, recursively honoring split orders.
        """
        if not col_split_by:
            return df.sort_values(by=col_sort_by)[x_col].unique().tolist()
        col = col_split_by[0]
        order = col_split_order[col]
        x_values = []
        present_vals = df[col].dropna().unique().tolist()
        for val in order:
            if val not in present_vals:
                continue
            sub_df = df[df[col] == val]
            x_values.extend(
                self._get_split_x_values(
                    sub_df, col_split_by[1:], col_split_order, x_col, col_sort_by
                )
            )
        return x_values
