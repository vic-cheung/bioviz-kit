"""
Oncoplot utilities adapted from tm_toolbox, ported into bioviz.

This module implements oncoprint drawing helpers and the
`OncoplotPlotter` class. It accepts a `style` object implementing
the `StyleBase` protocol and calls `style.apply_theme()` when plotting.
"""

from typing import Any

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from matplotlib.patches import Patch  # type: ignore

from .plot_configs import HeatmapAnnotationConfig, OncoplotConfig
from .style import DefaultStyle, StyleBase

# Do not apply a global style at import time; callers should pass a style.

__all__ = [
    "diagonal_fill",
    "my_shape_func",
    "draw_top_annotation",
    "merge_labels_without_splits",
    "label_block",
    "create_custom_legend_patch",
    "is_white_color",
    "OncoplotPlotter",
]


def is_white_color(color) -> bool:
    if color is None:
        return False
    try:
        rgb = mcolors.to_rgb(color)
        return all(abs(c - 1.0) < 0.01 for c in rgb)
    except (ValueError, AttributeError):
        if isinstance(color, str):
            return color.lower() in ["white", "#ffffff", "#fff"]
        return False


def diagonal_fill(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    which_half: str = "bottom_left",
) -> None:
    if which_half == "bottom_left":
        coords = [(x, y + height), (x, y), (x + width, y + height)]
    elif which_half == "upper_right":
        coords = [(x + width, y), (x, y), (x + width, y + height)]
    else:
        raise ValueError("which_half must be 'bottom_left' or 'upper_right'")
    poly = mpatches.Polygon(coords, closed=True, color=color, linewidth=0)
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
    if value in bottom_left_values:
        diagonal_fill(ax, x, y, width, height, color, which_half="bottom_left")
    elif value in upper_right_values:
        diagonal_fill(ax, x, y, width, height, color, which_half="upper_right")
    else:
        rect = mpatches.Rectangle((x, y), width, height, color=color, linewidth=0)
        ax.add_patch(rect)


def draw_top_annotation(
    ax,
    patients,
    col_positions,
    annotation_y,
    ann_config,
    ann_name,
    col_split_map=None,
) -> Any:
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

    for j, patient in enumerate(patients):
        if patient is None:
            continue
        x = col_positions[j]
        value = values.get(patient)
        if pd.isna(value) or (isinstance(value, str) and value.strip().lower() == "nan"):
            color = ann_config.na_color
            value_str = "NA"
        else:
            color = colors.get(str(value), "white")
            value_str = str(value)

        ax.add_patch(
            mpatches.Rectangle((x, annotation_y), 1, height, color=color, clip_on=False, zorder=10)
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
        blocks_needing_borders.append((block_start, len(patients) - 1))

    border_color = getattr(ann_config, "border_color", "black")
    border_width = getattr(ann_config, "border_width", 0.5)

    for start, end in blocks_needing_borders:
        x0 = col_positions[start]
        x1 = col_positions[end] + 1
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
        ax.text(
            min(col_positions) - 0.3,
            annotation_y + height / 2,
            display_name,
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="normal",
            clip_on=False,
            zorder=13,
        )

    if merge_labels:
        patient_values = []
        for i, patient in enumerate(patients):
            if patient is not None:
                value = values.get(patient)
                if pd.notna(value):
                    patient_values.append((i, patient, value))

        if patient_values:
            blocks = []
            current_block = [patient_values[0]]

            for i in range(1, len(patient_values)):
                prev_idx, prev_patient, prev_value = patient_values[i - 1]
                curr_idx, curr_patient, curr_value = patient_values[i]

                if curr_value == prev_value and curr_idx == prev_idx + 1:
                    current_block.append(patient_values[i])
                else:
                    blocks.append(current_block)
                    current_block = [patient_values[i]]

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
                                (min(block_cols) + max(block_cols) + 1) / 2 if block_cols else 0
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
        for i, patient in enumerate(patients):
            if patient is None:
                continue
            value = values.get(patient)
            if pd.notna(value):
                if value not in value_to_positions:
                    value_to_positions[value] = []
                value_to_positions[value].append(col_positions[i])

        for value, positions in value_to_positions.items():
            if positions:
                x_center = (min(positions) + max(positions) + 1) / 2
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
    ax, patients, col_positions, annotation_y, height, values, ann_config, fontsize
) -> None:
    last_value = None
    block_start_idx = None
    for i, patient in enumerate(list(patients) + [None]):
        value = values.get(patient) if patient is not None else None
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
                )
            block_start_idx = None
        if value != last_value and patient is not None:
            block_start_idx = i
        last_value = value


def label_block(
    ax, positions, start_idx, end_idx, annotation_y, height, value, ann_config, fontsize
):
    block_cols = positions[start_idx : end_idx + 1]
    x_center = (min(block_cols) + max(block_cols) + 1) / 2
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
    border_args = {}
    if draw_border or is_white_color(color):
        border_args = {"edgecolor": border_color, "linewidth": border_width}
    if shape == "upper_right":
        return mpatches.Polygon([(0, 0), (1, 0), (1, 1)], color=color, closed=True, **border_args)
    if shape == "bottom_left":
        return mpatches.Polygon([(0, 0), (0, 1), (1, 0)], color=color, closed=True, **border_args)
    return mpatches.Rectangle((0, 0), 1, 1, color=color, **border_args)


class OncoplotPlotter:
    def shift_row_group_bars_and_labels(
        self, ax, row_groups, bar_shift=2.75, label_shift=2.5
    ) -> None:
        current_xlim = ax.get_xlim()
        referenced_bars = getattr(self, "_row_group_bar_patches", [])
        referenced_labels = getattr(self, "_row_group_label_texts", [])
        for patch in referenced_bars:
            patch.set_x(patch.get_x() + bar_shift)
        if not referenced_bars:
            for patch in ax.patches:
                if hasattr(patch, "_is_row_group_bar") and patch._is_row_group_bar:
                    patch.set_x(patch.get_x() + bar_shift)
        for txt in referenced_labels:
            txt.set_x(txt.get_position()[0] + label_shift)
        if not referenced_labels and row_groups is not None:
            pathway_names = set(row_groups[self.row_group_col].unique())
            for txt in ax.texts:
                if hasattr(txt, "_is_row_label") and txt._is_row_label:
                    continue
                if (
                    hasattr(txt, "_is_row_group_label") and txt._is_row_group_label
                ) or txt.get_text() in pathway_names:
                    txt.set_x(txt.get_position()[0] + label_shift)
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
        if leftmost_x < current_xlim[0] and leftmost_x != float("inf"):
            padding = abs(current_xlim[0] - leftmost_x) + 1
            ax.set_xlim(current_xlim[0] - padding, current_xlim[1])
        ax.figure.canvas.draw_idle()

    def move_row_group_labels(self, ax, new_bar_x, bar_width=None) -> None:
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
        self.df = df
        self.row_groups = row_groups
        self.row_groups_color_dict = row_groups_color_dict
        self.config = config

        self._row_group_bar_patches = []
        self._row_group_label_texts = []

        self.col_split_by = config.col_split_by
        self.col_split_order = config.col_split_order
        self.col_sort_by = config.col_sort_by
        self.x_col = config.x_col
        self.y_col = config.y_col
        self.row_group_col = config.row_group_col
        self.figsize = config.figsize
        self.cell_aspect = config.cell_aspect
        self.top_annotations = config.top_annotations
        self.top_annotation_inter_spacer = config.top_annotation_inter_spacer
        self.top_annotation_intra_spacer = config.top_annotation_intra_spacer
        self.col_split_gap = config.col_split_gap
        self.row_split_gap = config.row_split_gap
        self.bar_width = config.bar_width
        self.bar_offset = config.bar_offset
        self.bar_buffer = config.bar_buffer
        self.row_group_label_fontsize = config.row_group_label_fontsize
        self.row_label_fontsize = config.row_label_fontsize
        self.column_label_fontsize = config.column_label_fontsize
        self.legend_fontsize = config.legend_fontsize
        self.legend_title_fontsize = config.legend_title_fontsize
        self.rotate_left_annotation_label = config.rotate_left_annotation_label
        self.legend_category_order = config.legend_category_order
        self.xticklabel_xoffset = config.xticklabel_xoffset
        self.xticklabel_yoffset = config.xticklabel_yoffset
        self.legend_bbox_to_anchor = config.legend_bbox_to_anchor
        self.legend_offset = config.legend_offset
        self.fig_top_margin = config.fig_top_margin
        self.fig_bottom_margin = config.fig_bottom_margin
        self.fig_y_margin = config.fig_y_margin

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
            self.heatmap_annotation = HeatmapAnnotationConfig(
                values=config.value_col,
                colors=config.row_values_color_dict,
                legend_title=(
                    config.value_legend_title if config.value_legend_title else config.value_col
                ),
            )
        else:
            self.heatmap_annotation = config.heatmap_annotation

        if isinstance(self.heatmap_annotation.values, str):
            self.value_col = self.heatmap_annotation.values
        else:
            self.value_col = "__value__"

    def plot(self) -> plt.Figure:
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
        bar_width = self.bar_width
        bar_offset = self.bar_offset
        bar_buffer = self.bar_buffer
        row_group_label_fontsize = self.row_group_label_fontsize
        row_label_fontsize = self.row_label_fontsize
        column_label_fontsize = self.column_label_fontsize
        legend_fontsize = self.legend_fontsize
        legend_title_fontsize = self.legend_title_fontsize
        rotate_left_annotation_label = self.rotate_left_annotation_label
        legend_category_order = self.legend_category_order
        heatmap_annotation = self.heatmap_annotation
        value_col = self.value_col
        fig_title = getattr(config, "fig_title", None)
        fig_title_fontsize = getattr(config, "fig_title_fontsize", 22)

        patients = self._get_split_patients(df, col_split_by, col_split_order, x_col, col_sort_by)

        col_positions = []
        pos = 0.0
        last_split_vals = None
        for pid in patients:
            split_vals = tuple(df.loc[df[x_col] == pid, col].iloc[0] for col in col_split_by)
            if last_split_vals is not None:
                for i, (prev, curr) in enumerate(zip(last_split_vals, split_vals)):
                    if prev != curr:
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
        if (
            row_groups is not None
            and isinstance(row_groups, pd.DataFrame)
            and not row_groups.empty
            and row_group_col in row_groups.columns
        ):
            for pathway in row_groups[row_group_col].unique():
                genes_in_group = row_groups[row_groups[row_group_col] == pathway].index.tolist()
                if last_pathway is not None:
                    pos += row_split_gap
                for gene in genes_in_group:
                    genes_ordered.append(gene)
                    row_group_positions.append(pathway)
                    row_positions.append(pos)
                    pos += 1.0
                last_pathway = pathway
        else:
            unique_genes = df[y_col].drop_duplicates().tolist()
            for gene in unique_genes:
                genes_ordered.append(gene)
                row_positions.append(pos)
                pos += 1.0
        nrows = int(np.ceil(pos))
        gene_to_idx = {g: i for g, i in zip(genes_ordered, row_positions)}
        patient_to_idx = {p: i for p, i in zip(patients, col_positions)}

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
        fig.subplots_adjust(
            top=fig_top_margin,
            bottom=bottom_margin,
        )
        set_title_later = False
        if fig_title:
            set_title_later = True
        ax.set_xlim(-1, ncols)
        ax.set_ylim(-1, nrows)
        ax.set_aspect("auto")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        bottom_left_values = getattr(heatmap_annotation, "bottom_left_triangle_values", ["SV"])
        upper_right_values = getattr(heatmap_annotation, "upper_right_triangle_values", ["CNV"])

        for _, row in df.iterrows():
            gene, patient = row.get(y_col), row.get(x_col)
            if gene in gene_to_idx and patient in patient_to_idx:
                if isinstance(heatmap_annotation.values, str):
                    value = row.get(heatmap_annotation.values)
                else:
                    try:
                        value = heatmap_annotation.values.get(patient)
                    except Exception:
                        value = None
                color = heatmap_annotation.colors.get(value, "white")
                my_shape_func(
                    ax,
                    patient_to_idx[patient],
                    gene_to_idx[gene],
                    1,
                    1,
                    color,
                    value,
                    bottom_left_values,
                    upper_right_values,
                )

        mutation_groups = {}
        for _, row in df.iterrows():
            gene, patient = row.get(y_col), row.get(x_col)
            value = row.get(value_col)
            if gene in gene_to_idx and patient in patient_to_idx:
                key = (gene, patient)
                if key not in mutation_groups:
                    mutation_groups[key] = []
                mutation_groups[key].append(value)

        for (gene, patient), values in mutation_groups.items():
            x, y = patient_to_idx[patient], gene_to_idx[gene]
            for bl_value in bottom_left_values:
                if bl_value in values:
                    diagonal_fill(
                        ax,
                        x,
                        y,
                        1,
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
                        1,
                        1,
                        heatmap_annotation.colors.get(ur_value),
                        which_half="upper_right",
                    )
                    break
            if not any(val in values for val in bottom_left_values + upper_right_values):
                for value in values:
                    color = heatmap_annotation.colors.get(value, "white")
                    rect = mpatches.Rectangle((x, y), 1, 1, color=color, linewidth=0)
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
                    patients,
                    col_positions,
                    annotation_y,
                    ann_config,
                    ann_name,
                    col_split_map=col_split_map,
                )
                annotation_y -= ann_config.height + top_annotation_intra_spacer

        legend_categories = {}
        heatmap_legend_title = heatmap_annotation.legend_title
        mutation_handles = [Patch(facecolor="none", edgecolor="none", label=heatmap_legend_title)]
        mutation_value_order = heatmap_annotation.legend_value_order
        remove_unused_keys = getattr(config, "remove_unused_keys_in_legend", False)
        if remove_unused_keys:
            if isinstance(heatmap_annotation.values, str):
                present_values = set(df[heatmap_annotation.values].dropna().unique())
            else:
                present_values = set(pd.Series(heatmap_annotation.values).dropna().unique())
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

                        if needs_border:
                            mutation_handles.append(
                                Patch(
                                    facecolor=color,
                                    edgecolor=heatmap_border_color,
                                    linewidth=heatmap_border_width,
                                    label=label,
                                )
                            )
                        else:
                            mutation_handles.append(Patch(facecolor=color, label=label))
        else:
            for label, color in heatmap_annotation.colors.items():
                if not remove_unused_keys or label in present_values:
                    if heatmap_border_categories is not None:
                        needs_border = label in heatmap_border_categories
                    elif heatmap_draw_border:
                        needs_border = True
                    else:
                        needs_border = is_white_color(color)

                    if needs_border:
                        mutation_handles.append(
                            Patch(
                                facecolor=color,
                                edgecolor=heatmap_border_color,
                                linewidth=heatmap_border_width,
                                label=label,
                            )
                        )
                    else:
                        mutation_handles.append(Patch(facecolor=color, label=label))
        legend_categories[heatmap_legend_title] = mutation_handles
        if top_annotations:
            for ann_name, ann_config in top_annotations.items():
                legend_title = ann_config.legend_title or ann_name
                annotation_handles = [Patch(color="none", label=legend_title)]
                value_order = ann_config.legend_value_order or sorted(ann_config.colors.keys())

                ann_draw_border = getattr(ann_config, "draw_border", False)
                ann_border_categories = getattr(ann_config, "border_categories", None)
                ann_border_color = getattr(ann_config, "border_color", "black")
                ann_border_width = getattr(ann_config, "border_width", 0.5)

                if remove_unused_keys:
                    ann_values = pd.Series(ann_config.values)
                    present_ann_values = set(ann_values.dropna().unique())
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

                            if needs_border:
                                annotation_handles.append(
                                    Patch(
                                        facecolor=color,
                                        edgecolor=ann_border_color,
                                        linewidth=ann_border_width,
                                        label=str(value),
                                    )
                                )
                            else:
                                annotation_handles.append(Patch(facecolor=color, label=str(value)))
                if remove_unused_keys:
                    if ann_values.isna().any():
                        annotation_handles.append(Patch(color=ann_config.na_color, label="NA"))
                else:
                    if hasattr(ann_config, "na_color") and ann_config.na_color is not None:
                        annotation_handles.append(Patch(color=ann_config.na_color, label="NA"))
                legend_categories[legend_title] = annotation_handles
        legend_handles = []
        if legend_category_order:
            for category in legend_category_order:
                if category in legend_categories:
                    legend_handles.extend(legend_categories[category])
                    legend_handles.append(Patch(color="none", label=""))
                else:
                    print(
                        (
                            f"Warning: Legend category '{category}' not found."
                            f"Available categories: {list(legend_categories.keys())}"
                        )
                    )
            for category, handles in legend_categories.items():
                if category not in legend_category_order:
                    legend_handles.extend(handles)
                    legend_handles.append(Patch(color="none", label=""))
        else:
            legend_handles.extend(legend_categories.get(heatmap_legend_title, []))
            legend_handles.append(Patch(color="none", label=""))
            for ann_name, ann_config in (top_annotations or {}).items():
                legend_title = ann_config.legend_title or ann_name
                if legend_title != heatmap_legend_title:
                    legend_handles.extend(legend_categories.get(legend_title, []))
                    legend_handles.append(Patch(color="none", label=""))

        ax.set_aspect(cell_aspect)
        fig.canvas.draw()

        legend_kwargs = {}
        if self.legend_bbox_to_anchor is not None:
            bbox_to_anchor = self.legend_bbox_to_anchor
        else:
            bbox_to_anchor = (1 + self.legend_offset, 0.5)
            legend_kwargs["bbox_transform"] = ax.transAxes

        gene_labels = []
        for y, g in zip(row_positions, genes_ordered):
            text = ax.text(
                -0.3,
                y + 0.55,
                g,
                ha="right",
                va="center",
                fontsize=row_label_fontsize,
                clip_on=False,
            )
            text._is_row_label = True
            gene_labels.append(text)

        dynamic_xoffset = self.xticklabel_xoffset
        dynamic_yoffset = self.xticklabel_yoffset
        aspect_ratio = getattr(self.config, "aspect", 1.0)
        row_count = len(row_positions)
        col_count = len(col_positions)
        if row_count <= 10:
            dynamic_yoffset = max(0.15, self.xticklabel_yoffset * 0.8)
        elif row_count <= 20:
            dynamic_yoffset = max(0.25, self.xticklabel_yoffset * 1.2)
        else:
            dynamic_yoffset = max(0.4, self.xticklabel_yoffset * 1.5)
        if col_count <= 6:
            dynamic_xoffset = max(0.15, self.xticklabel_xoffset * 1.5)
        else:
            dynamic_xoffset = self.xticklabel_xoffset

        for i, (x, p) in enumerate(zip(col_positions, patients)):
            ax.text(
                x + cell_aspect / 2 + dynamic_xoffset,
                nrows + dynamic_yoffset,
                p,
                ha="right",
                va="center",
                fontsize=column_label_fontsize,
                rotation=90,
                rotation_mode="anchor",
                clip_on=False,
            )

        lgd = ax.legend(
            handles=legend_handles,
            bbox_to_anchor=bbox_to_anchor,
            loc="center left",
            frameon=False,
            handlelength=1,
            handleheight=1,
            fontsize=legend_fontsize,
            ncol=1,
            title_fontsize=legend_title_fontsize,
            **legend_kwargs,
        )
        for text in lgd.get_texts():
            if text.get_text() in [heatmap_legend_title] + [
                (ann_config.legend_title or n) for n, ann_config in (top_annotations or {}).items()
            ]:
                text.set_fontweight("bold")

        leftmost_x = float("inf")
        for label in gene_labels:
            bbox = label.get_window_extent().transformed(ax.transData.inverted())
            if bbox.x0 < leftmost_x:
                leftmost_x = bbox.x0
        total_offset = bar_offset + bar_buffer
        scaled_offset = total_offset * cell_aspect if cell_aspect < 1 else total_offset
        calculated_bar_x = leftmost_x + scaled_offset
        label_gap = 1.0
        self._row_group_bar_patches.clear()
        self._row_group_label_texts.clear()
        if (
            row_groups is not None
            and isinstance(row_groups, pd.DataFrame)
            and not row_groups.empty
            and row_group_col in row_groups.columns
        ):
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
                bar_x_shift = calculated_bar_x
                if not rotate_left_annotation_label:
                    label_x_shift = bar_x_shift - label_gap
                    label_ha = "right"
                else:
                    label_x_shift = bar_x_shift - 0.2
                    label_ha = "right"
                bar_patch = mpatches.Rectangle(
                    (bar_x_shift, y_start),
                    bar_width,
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

        aspect = getattr(self.config, "aspect", 1.0)
        if aspect != 1 and aspect != 1.0:
            ax = plt.gca()
            ax.set_aspect(aspect)
            bar_x = OncoplotPlotter.get_dynamic_bar_x(ax, self.bar_offset, self.cell_aspect)
            adjustment_factor = min(max(aspect, 0.5), 2.0)
            bar_offset = bar_x * adjustment_factor
            bar_width = self.bar_width * adjustment_factor
            self.move_row_group_labels(ax, bar_offset, bar_width)
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
        return fig

    def _get_split_patients(
        self, df, col_split_by, col_split_order, x_col, col_sort_by
    ) -> Any | list:
        if not col_split_by:
            return df.sort_values(by=col_sort_by)[x_col].unique().tolist()
        col = col_split_by[0]
        order = col_split_order[col]
        patients = []
        present_vals = df[col].dropna().unique().tolist()
        for val in order:
            if val not in present_vals:
                continue
            sub_df = df[df[col] == val]
            patients.extend(
                self._get_split_patients(
                    sub_df, col_split_by[1:], col_split_order, x_col, col_sort_by
                )
            )
        return patients
