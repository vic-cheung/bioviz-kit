from __future__ import annotations

from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from bioviz.configs import OncoplotConfig, TopAnnotationConfig
from bioviz.plots.oncoplot import (
    OncoPlotter,
    _ensure_opaque_color,
    is_white_color,
)
from bioviz.utils.plotting import resolve_font_family

__all__ = ["OncoPrevalencePlotter", "OncoPrevalenceRasterPlotter", "OncoGeneBarPlotter"]


class _OncoAggregatePlotterBase(OncoPlotter):
    def __init__(
        self,
        df: pd.DataFrame,
        config: OncoplotConfig,
        row_groups: pd.DataFrame | None = None,
        row_groups_color_dict: dict[str, str] | None = None,
        style=None,
        *,
        group_by: list[str] | None = None,
        include_overall: bool = True,
        show_all: bool | None = None,
        all_column_gap: float = 0.18,
        separate_all_columns: bool = False,
        percent_decimals: int = 0,
        annotate_values: bool = True,
        show_total_counts: bool = False,
        show_category_breakdown: bool = False,
        show_category_counts: bool = False,
        label_fontsize: float | int | None = None,
        label_text_color: str | None = None,
        label_bbox_mode: Literal["auto", "always", "multiline-only", "never"] = "auto",
        label_bbox_facecolor: str = "white",
        label_bbox_alpha: float = 0.78,
        label_bbox_edgecolor: str = "none",
        label_bbox_boxstyle: str = "round,pad=0.18",
    ) -> None:
        super().__init__(
            df=df,
            config=config,
            row_groups=row_groups,
            row_groups_color_dict=row_groups_color_dict,
            style=style,
        )
        self.group_by = list(group_by or config.col_split_by or [])
        self.include_overall = include_overall if show_all is None else bool(show_all)
        self.all_column_gap = max(float(all_column_gap), 0.0)
        self.separate_all_columns = bool(separate_all_columns)
        self.percent_decimals = percent_decimals
        self.annotate_values = annotate_values
        self.show_total_counts = show_total_counts
        self.show_category_breakdown = show_category_breakdown
        self.show_category_counts = show_category_counts
        self.aggregate_label_fontsize = label_fontsize
        self.aggregate_label_text_color = label_text_color
        self.aggregate_label_bbox_mode = label_bbox_mode
        self.aggregate_label_bbox_facecolor = label_bbox_facecolor
        self.aggregate_label_bbox_alpha = float(label_bbox_alpha)
        self.aggregate_label_bbox_edgecolor = label_bbox_edgecolor
        self.aggregate_label_bbox_boxstyle = label_bbox_boxstyle

    def _build_aggregate_columns(self) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        valid_group_by = [col for col in self.group_by if col in self.df.columns]
        sample_cols = [self.x_col] + valid_group_by
        sample_meta = self.df[sample_cols].drop_duplicates(subset=[self.x_col]).copy()
        sample_meta = sample_meta.drop_duplicates(subset=[self.x_col])

        columns: list[dict[str, Any]] = []
        all_samples = sample_meta[self.x_col].drop_duplicates().tolist()
        if self.include_overall or not valid_group_by:
            columns.append({"title": "All", "samples": all_samples, "meta": {}})
        if not valid_group_by:
            return columns, sample_meta

        sort_cols: list[str] = []
        for col in valid_group_by:
            if col in self.col_split_order and self.col_split_order[col]:
                order_col = f"__order__{col}"
                sample_meta[order_col] = pd.Categorical(
                    sample_meta[col],
                    categories=self.col_split_order[col],
                    ordered=True,
                )
                sort_cols.append(order_col)
            else:
                sort_cols.append(col)
        sample_meta = sample_meta.sort_values(sort_cols, kind="stable")

        grouped = sample_meta.groupby(valid_group_by, dropna=False, sort=False)
        for keys, subset in grouped:
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            meta: dict[str, str] = {}
            label_parts: list[str] = []
            for col, val in zip(valid_group_by, key_tuple, strict=True):
                display = str(val)
                if val is None or display.lower() == "nan":
                    display = "NA"
                meta[col] = display
                label_parts.append(display if len(valid_group_by) == 1 else f"{col}={display}")
            samples = subset[self.x_col].drop_duplicates().tolist()
            if samples:
                columns.append(
                    {
                        "title": " | ".join(label_parts),
                        "samples": samples,
                        "meta": meta,
                    }
                )
        return columns, sample_meta

    def _resolve_aggregate_annotation_values(
        self,
        ann_config: TopAnnotationConfig,
        columns: list[dict[str, Any]],
        sample_meta: pd.DataFrame,
    ) -> pd.Series:
        titles = [column["title"] for column in columns]
        values = ann_config.values

        if isinstance(values, str) and values in sample_meta.columns:
            series = sample_meta.set_index(self.x_col)[values]
            return self._aggregate_series_to_columns(series, columns)

        raw_series = pd.Series(values)
        raw_series = raw_series[~raw_series.index.duplicated(keep="first")]
        if titles and set(titles).issubset(set(raw_series.index.astype(str))):
            return pd.Series({title: raw_series.get(title) for title in titles})
        sample_index = pd.Index(sample_meta[self.x_col].astype(str))
        if sample_index.isin(raw_series.index.astype(str)).all():
            raw_series.index = raw_series.index.astype(str)
            return self._aggregate_series_to_columns(raw_series, columns)
        if len(raw_series) == len(titles):
            return pd.Series(raw_series.to_list(), index=titles)
        return pd.Series(index=titles, dtype=object)

    def _resolve_annotation_sample_series(
        self,
        ann_config: TopAnnotationConfig,
        sample_meta: pd.DataFrame,
    ) -> pd.Series:
        values = ann_config.values
        if isinstance(values, str):
            if values in sample_meta.columns:
                series = sample_meta.set_index(self.x_col)[values]
                series.index = series.index.astype(str)
                return pd.Series(series)
            if values in self.df.columns:
                series = self.df[[self.x_col, values]].drop_duplicates(subset=[self.x_col])
                series = series.set_index(self.x_col)[values]
                series.index = series.index.astype(str)
                return pd.Series(series)
            return pd.Series(dtype=object)

        raw_series = pd.Series(values)
        raw_series = raw_series[~raw_series.index.duplicated(keep="first")]
        sample_index = pd.Index(sample_meta[self.x_col].astype(str))
        if sample_index.isin(raw_series.index.astype(str)).all():
            raw_series.index = raw_series.index.astype(str)
            return pd.Series(raw_series)
        return pd.Series(dtype=object)

    def _aggregate_annotation_breakdowns(
        self,
        series: pd.Series,
        columns: list[dict[str, Any]],
    ) -> dict[str, dict[str, int]]:
        series = pd.Series(series)
        series.index = series.index.astype(str)
        breakdowns: dict[str, dict[str, int]] = {}
        for column in columns:
            observed: list[str] = []
            for sample in [str(sample) for sample in column["samples"]]:
                value = series.get(sample)
                if pd.notna(value):
                    observed.append(str(value))
            counts = pd.Series(observed, dtype=object).value_counts().to_dict() if observed else {}
            breakdowns[column["title"]] = {str(key): int(value) for key, value in counts.items()}
        return breakdowns

    def _annotation_value_order(
        self,
        ann_config: TopAnnotationConfig,
        breakdowns: dict[str, dict[str, int]],
    ) -> list[str]:
        ordered = [str(value) for value in (ann_config.legend_value_order or [])]
        color_keys = [str(value) for value in ann_config.colors]
        observed = [
            str(value)
            for counts in breakdowns.values()
            for value, count in counts.items()
            if count > 0
        ]
        result: list[str] = []
        for value in ordered + color_keys + observed:
            if value not in result:
                result.append(value)
        return result

    def _aggregate_series_to_columns(
        self,
        series: pd.Series,
        columns: list[dict[str, Any]],
    ) -> pd.Series:
        series = pd.Series(series)
        series.index = series.index.astype(str)
        aggregated: dict[str, Any] = {}
        for column in columns:
            column_samples = [str(sample) for sample in column["samples"]]
            observed = []
            for sample in column_samples:
                value = series.get(sample)
                if pd.notna(value):
                    observed.append(str(value))
            unique_values = pd.Index(observed).drop_duplicates().tolist()
            if not unique_values:
                aggregated[column["title"]] = pd.NA
            elif len(unique_values) == 1:
                aggregated[column["title"]] = unique_values[0]
            else:
                aggregated[column["title"]] = "Mixed"
        return pd.Series(aggregated)

    def _compute_gene_rows(self) -> tuple[list[Any], list[float], dict[Any, float]]:
        observed_genes = self.df[self.y_col].dropna().drop_duplicates().tolist()
        genes_ordered: list[Any] = []
        row_positions: list[float] = []
        pos = 0.0
        if self._has_row_groups and (
            self.row_groups is not None
            and isinstance(self.row_groups, pd.DataFrame)
            and not self.row_groups.empty
            and self.row_group_col in self.row_groups.columns
        ):
            group_values = self.row_groups[self.row_group_col].unique().tolist()
            custom_order = getattr(self.config, "row_group_order", None)
            if custom_order:
                seen = set()
                ordered = []
                for group in custom_order:
                    if group in group_values and group not in seen:
                        ordered.append(group)
                        seen.add(group)
                remaining = [group for group in group_values if group not in seen]
                group_values = ordered + remaining
            for idx, pathway in enumerate(group_values):
                genes_in_group = self.row_groups[
                    self.row_groups[self.row_group_col] == pathway
                ].index.tolist()
                if idx > 0:
                    pos += self.row_split_gap
                for gene in genes_in_group:
                    genes_ordered.append(gene)
                    row_positions.append(pos)
                    pos += 1.0
            missing_genes = [gene for gene in observed_genes if gene not in genes_ordered]
            if missing_genes and genes_ordered:
                pos += self.row_split_gap
            for gene in missing_genes:
                genes_ordered.append(gene)
                row_positions.append(pos)
                pos += 1.0
        else:
            for gene in observed_genes:
                genes_ordered.append(gene)
                row_positions.append(pos)
                pos += 1.0
        gene_to_idx = {gene: row for gene, row in zip(genes_ordered, row_positions, strict=True)}
        return genes_ordered, row_positions, gene_to_idx

    def _event_types(self, plot_df: pd.DataFrame) -> list[str]:
        event_types = [
            str(value)
            for value in getattr(self.heatmap_annotation, "legend_value_order", None) or []
        ]
        if event_types:
            return event_types
        if self.value_col in plot_df.columns:
            observed = [str(value) for value in plot_df[self.value_col].dropna().unique().tolist()]
        else:
            observed = []
        colors = getattr(self.heatmap_annotation, "colors", {}) or {}
        ordered = [label for label in colors if str(label) in observed]
        remaining = [label for label in observed if label not in ordered]
        return [str(label) for label in ordered + remaining]

    def _prepare_plot_df(self) -> pd.DataFrame:
        plot_df = self.df.copy()
        if self.value_col not in plot_df.columns and isinstance(
            self.heatmap_annotation.values, str
        ):
            source_col = self.heatmap_annotation.values
            if source_col in plot_df.columns:
                plot_df[self.value_col] = plot_df[source_col]
        return plot_df

    def _format_percent(self, count: float, denom: int) -> str:
        percent = 100.0 * float(count) / float(denom) if denom else 0.0
        return f"{percent:.{self.percent_decimals}f}%"

    def _build_cell_label(
        self,
        *,
        total_count: float,
        denom: int,
        counts_by_type: dict[str, float] | None = None,
        event_types: list[str] | None = None,
    ) -> str | None:
        if not self.annotate_values:
            return None

        total_label = self._format_percent(total_count, denom)
        if self.show_total_counts:
            total_label += f" ({int(total_count)}/{int(denom)})"

        lines = [total_label]
        if self.show_category_breakdown and counts_by_type and event_types:
            parts = []
            for event_type in event_types:
                event_count = float(counts_by_type.get(str(event_type), 0.0))
                if event_count <= 0:
                    continue
                part = f"{event_type}: {self._format_percent(event_count, denom)}"
                if self.show_category_counts:
                    part += f" ({int(event_count)})"
                parts.append(part)
            if not parts:
                return "\n".join(lines)
            if len(parts) <= 2:
                lines.append(", ".join(parts))
            else:
                for idx in range(0, len(parts), 2):
                    lines.append(", ".join(parts[idx : idx + 2]))

        return "\n".join(lines)

    def _get_label_style(self, facecolor: Any, label: str) -> tuple[str, dict[str, Any]]:
        multiline = "\n" in label

        bbox_mode = self.aggregate_label_bbox_mode
        use_bbox = False
        if bbox_mode == "always":
            use_bbox = True
        elif bbox_mode in {"auto", "multiline-only"}:
            use_bbox = multiline
        elif bbox_mode == "never":
            use_bbox = False

        bbox_kwargs = (
            {
                "bbox": {
                    "facecolor": self.aggregate_label_bbox_facecolor,
                    "alpha": self.aggregate_label_bbox_alpha,
                    "edgecolor": self.aggregate_label_bbox_edgecolor,
                    "boxstyle": self.aggregate_label_bbox_boxstyle,
                }
            }
            if use_bbox
            else {}
        )

        if self.aggregate_label_text_color is not None:
            return (str(self.aggregate_label_text_color), bbox_kwargs)

        if use_bbox:
            return ("black", bbox_kwargs)

        rgb = mcolors.to_rgb(facecolor)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return ("white" if luminance < 0.5 else "black", {})

    def _label_font_size(self) -> float:
        if self.aggregate_label_fontsize is not None:
            return float(self.aggregate_label_fontsize)
        if self.show_category_breakdown:
            return float(max(self.row_label_fontsize - 7, 6))
        if self.show_total_counts:
            return float(max(self.row_label_fontsize - 4, 7))
        return float(max(self.row_label_fontsize - 2, 8))

    def _setup_axes(
        self,
        ncols: int,
        nrows: int,
        max_x: float,
        fig_title: str | None,
        fig_title_fontsize: float | int,
    ) -> tuple[plt.Figure, plt.Axes, Any, float]:
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.subplots_adjust(top=self.fig_top_margin, bottom=self.fig_bottom_margin)
        ax.set_xlim(-1, max(ncols, max_x))
        ax.set_ylim(-1, nrows)
        if self.axes_aspect_mode == "auto":
            ax.set_aspect("auto")
        else:
            ax.set_aspect(getattr(self.config, "aspect", 1.0) or 1.0)
        ax.set_facecolor("white")
        fig.patch.set_facecolor(getattr(self.config, "figure_facecolor", None) or "white")
        fig.patch.set_alpha(0.0 if getattr(self.config, "figure_transparent", False) else 1.0)
        if fig_title:
            fig.suptitle(fig_title, fontsize=fig_title_fontsize)

        xlim_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1e-6)
        pts_per_data_unit_x = (fig.get_figwidth() * 72.0) / xlim_span
        if self.rowlabel_use_points:
            rowlabel_offset_pts = float(self.yticklabel_xoffset) * pts_per_data_unit_x
            rowlabel_translate = mtransforms.ScaledTranslation(
                rowlabel_offset_pts / 72.0, 0.0, fig.dpi_scale_trans
            )
            rowlabel_text_transform = ax.transData + rowlabel_translate
            rowlabel_base_x = 0.0
        else:
            rowlabel_text_transform = ax.transData
            rowlabel_base_x = -abs(float(self.yticklabel_xoffset))
        return fig, ax, rowlabel_text_transform, rowlabel_base_x

    def _draw_top_annotations(
        self,
        ax: plt.Axes,
        columns: list[dict[str, Any]],
        col_positions: list[float],
        sample_meta: pd.DataFrame,
        rowlabel_text_transform,
        rowlabel_base_x: float,
    ) -> dict[str, TopAnnotationConfig]:
        if not self.top_annotations:
            return {}
        annotation_y = self.top_annotation_inter_spacer * -1
        annotation_order = (
            [name for name in self.config.top_annotation_order if name in self.top_annotations][
                ::-1
            ]
            if self.config.top_annotation_order
            else list(self.top_annotations.keys())
        )
        for name in self.top_annotations:
            if name not in annotation_order:
                annotation_order.append(name)
        x_values = [column["title"] for column in columns]
        heatmap_left = min(col_positions) if col_positions else 0.0
        if self.top_annotation_label_use_points:
            label_transform = rowlabel_text_transform
            label_x = rowlabel_base_x
        else:
            label_transform = ax.transData
            label_x = heatmap_left - float(self.top_annotation_label_offset)

        resolved: dict[str, TopAnnotationConfig] = {}
        for ann_name in annotation_order:
            ann_config = self.top_annotations[ann_name]
            raw_series = self._resolve_annotation_sample_series(ann_config, sample_meta)
            breakdowns = self._aggregate_annotation_breakdowns(raw_series, columns)
            ann_values = self._resolve_aggregate_annotation_values(ann_config, columns, sample_meta)
            colors = dict(ann_config.colors)
            resolved_config = ann_config.model_copy(update={"values": ann_values, "colors": colors})
            self._draw_aggregate_top_annotation(
                ax=ax,
                x_values=x_values,
                col_positions=col_positions,
                annotation_y=annotation_y,
                ann_config=resolved_config,
                ann_name=ann_name,
                breakdowns=breakdowns,
                label_x=label_x,
                label_transform=label_transform,
            )
            resolved[ann_name] = resolved_config
            annotation_y -= resolved_config.height + self.top_annotation_intra_spacer
        return resolved

    def _draw_aggregate_top_annotation(
        self,
        *,
        ax: plt.Axes,
        x_values: list[str],
        col_positions: list[float],
        annotation_y: float,
        ann_config: TopAnnotationConfig,
        ann_name: str,
        breakdowns: dict[str, dict[str, int]],
        label_x: float,
        label_transform,
    ) -> None:
        if not col_positions:
            return

        height = ann_config.height
        display_name = ann_config.display_name or ann_name
        value_order = self._annotation_value_order(ann_config, breakdowns)
        border_color = getattr(ann_config, "border_color", "black")
        border_width = getattr(ann_config, "border_width", 0.5)

        for x_value, x in zip(x_values, col_positions, strict=True):
            counts = breakdowns.get(x_value, {})
            total = int(sum(counts.values()))
            if total <= 0:
                ax.add_patch(
                    plt.Rectangle(
                        (x, annotation_y),
                        self.cell_aspect,
                        height,
                        facecolor=ann_config.na_color,
                        edgecolor=border_color if ann_config.draw_border else "none",
                        linewidth=border_width if ann_config.draw_border else 0,
                        clip_on=False,
                        zorder=10,
                    )
                )
                continue

            left = x
            ordered_values = [value for value in value_order if counts.get(str(value), 0) > 0]
            if not ordered_values:
                ordered_values = [value for value, count in counts.items() if count > 0]
            needs_border = bool(ann_config.draw_border)
            for value in ordered_values:
                count = int(counts.get(str(value), 0))
                if count <= 0:
                    continue
                width = self.cell_aspect * (count / total)
                color = _ensure_opaque_color(ann_config.colors.get(str(value), ann_config.na_color))
                if is_white_color(color):
                    needs_border = True
                ax.add_patch(
                    plt.Rectangle(
                        (left, annotation_y),
                        width,
                        height,
                        facecolor=color,
                        edgecolor="none",
                        linewidth=0,
                        clip_on=False,
                        zorder=10,
                    )
                )
                left += width

            if needs_border:
                ax.add_patch(
                    plt.Rectangle(
                        (x, annotation_y),
                        self.cell_aspect,
                        height,
                        fill=False,
                        edgecolor=border_color,
                        linewidth=border_width,
                        clip_on=False,
                        zorder=11,
                    )
                )

        ax.text(
            label_x,
            annotation_y + height / 2,
            display_name,
            ha="right",
            va="center",
            fontsize=ann_config.fontsize,
            fontweight="normal",
            clip_on=False,
            zorder=13,
            transform=label_transform or ax.transData,
        )

    def _draw_gene_labels(
        self,
        ax: plt.Axes,
        fig: plt.Figure,
        genes_ordered: list[Any],
        row_positions: list[float],
        rowlabel_text_transform,
        rowlabel_base_x: float,
    ) -> None:
        for y, gene in zip(row_positions, genes_ordered, strict=True):
            text = ax.text(
                rowlabel_base_x,
                y + 0.55,
                gene,
                ha="right",
                va="center",
                fontsize=self.row_label_fontsize,
                clip_on=False,
                transform=rowlabel_text_transform,
            )
            text_marker: Any = text
            text_marker._is_row_label = True
        fig.canvas.draw()

    def _draw_row_groups(
        self,
        ax: plt.Axes,
        fig: plt.Figure,
        genes_ordered: list[Any],
        gene_to_idx: dict[Any, float],
        heatmap_left: float,
    ) -> None:
        if not self._has_row_groups or self.row_groups is None or self.row_group_col is None:
            ax.invert_yaxis()
            return
        d0 = ax.transData.transform((0.0, 0.0))
        d1 = ax.transData.transform((1.0, 0.0))
        data_dx_px = max(abs(d1[0] - d0[0]), 1e-6)
        pts_per_data_unit_x = data_dx_px / (fig.dpi / 72.0)
        total_offset_data = (
            (self.bar_offset + self.bar_buffer) / pts_per_data_unit_x
            if self.bar_offset_use_points
            else self.bar_offset + self.bar_buffer
        )
        bar_width_draw = (
            self.bar_width_points / pts_per_data_unit_x
            if self.bar_width_use_points
            else self.bar_width
        )
        label_gap = (
            float(self.row_group_label_gap) / pts_per_data_unit_x
            if self.row_group_label_gap_use_points
            else max(self.row_group_label_gap, 0.0)
        )
        bar_x = heatmap_left - total_offset_data - bar_width_draw
        self._row_group_bar_patches.clear()
        self._row_group_label_texts.clear()
        for pathway in self.row_groups[self.row_group_col].unique():
            color = (
                self.row_groups_color_dict.get(pathway, "black")
                if self.row_groups_color_dict
                else "black"
            )
            genes_in_group = self.row_groups[
                self.row_groups[self.row_group_col] == pathway
            ].index.tolist()
            y_positions = [gene_to_idx[gene] for gene in genes_in_group if gene in gene_to_idx]
            if not y_positions:
                continue
            y_start, y_end = min(y_positions), max(y_positions)
            bar_height = y_end - y_start + 1
            label_x = bar_x - label_gap if not self.rotate_left_annotation_label else bar_x - 0.2
            bar_patch = plt.Rectangle(
                (bar_x, y_start),
                bar_width_draw,
                bar_height,
                color=color,
                clip_on=False,
                zorder=5,
            )
            bar_patch_marker: Any = bar_patch
            bar_patch_marker._is_row_group_bar = True
            ax.add_patch(bar_patch)
            self._row_group_bar_patches.append(bar_patch)
            label_text = ax.text(
                label_x,
                (y_start + y_end) / 2 + 0.5,
                pathway,
                ha="right",
                va="center",
                fontsize=self.row_group_label_fontsize,
                color=color,
                clip_on=False,
                rotation=90 if self.rotate_left_annotation_label else 0,
            )
            label_marker: Any = label_text
            label_marker._is_row_group_label = True
            self._row_group_label_texts.append(label_text)
        ax.invert_yaxis()

    def _draw_column_labels(
        self,
        ax: plt.Axes,
        columns: list[dict[str, Any]],
        col_positions: list[float],
    ) -> None:
        if not self.show_column_labels:
            ax.set_xticks([])
            return
        centers = [position + self.cell_aspect / 2 for position in col_positions]
        ax.set_xticks(centers)
        ax.set_xticklabels(
            [column["title"] for column in columns], fontsize=self.column_label_fontsize
        )
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        ax.tick_params(axis="x", bottom=True, top=False, length=0, pad=8)

    def _finalize_layout(self, fig: plt.Figure, ax: plt.Axes) -> None:
        ax.margins(y=self.fig_y_margin)
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        if self.fig_y_margin > 0 and yrange > 0:
            ax.set_ylim(ymin, ymax + yrange * self.fig_y_margin)

        if getattr(self.config, "apply_post_row_group_shift", False):
            fig.canvas.draw()
            self.shift_row_group_bars_and_labels(
                ax,
                self.row_groups,
                self.row_group_post_bar_shift,
                self.row_group_post_label_shift,
                self.row_group_post_bar_shift_points,
                self.row_group_post_label_shift_points,
                self.row_group_post_shift_use_points,
            )
            fig.canvas.draw_idle()

    def _build_top_annotation_legend(
        self, top_annotations: dict[str, TopAnnotationConfig]
    ) -> dict[str, list[Patch]]:
        legend_categories: dict[str, list[Patch]] = {}
        for ann_name, ann_config in top_annotations.items():
            legend_title = ann_config.legend_title or ann_name
            handles: list[Patch] = []
            value_order = ann_config.legend_value_order or list(ann_config.colors.keys())
            for value in value_order:
                if str(value) not in ann_config.colors:
                    continue
                color = ann_config.colors[str(value)]
                needs_border = is_white_color(color)
                face = _ensure_opaque_color(color, default="white")
                if needs_border:
                    handles.append(
                        Patch(facecolor=face, edgecolor="black", linewidth=0.5, label=str(value))
                    )
                else:
                    handles.append(Patch(facecolor=face, label=str(value)))
            if "Mixed" in ann_config.colors and "Mixed" not in value_order:
                handles.append(Patch(facecolor=ann_config.colors["Mixed"], label="Mixed"))
            if handles:
                legend_categories[legend_title] = handles
        return legend_categories

    def _draw_legend(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        top_annotations: dict[str, TopAnnotationConfig],
        mutation_handles: list[Patch] | None = None,
        mutation_title: str | None = None,
        anchor_ax: plt.Axes | None = None,
        extra_offset_points: float = 0.0,
    ) -> None:
        legend_categories = self._build_top_annotation_legend(top_annotations)
        if mutation_handles:
            legend_categories = {
                **({mutation_title or "Mutation Type": mutation_handles}),
                **legend_categories,
            }
        if not legend_categories:
            return

        if self.legend_category_order:
            order = [
                category for category in self.legend_category_order if category in legend_categories
            ]
            for category in legend_categories:
                if category not in order:
                    order.append(category)
        else:
            order = list(legend_categories.keys())

        handles: list[Patch] = []
        label_headers = set()
        for idx, category in enumerate(order):
            label_headers.add(category)
            handles.append(Patch(color="none", label=category))
            handles.extend(legend_categories[category])
            if idx < len(order) - 1:
                handles.append(Patch(color="none", label=""))

        if self.legend_bbox_to_anchor is not None:
            bbox_to_anchor = self.legend_bbox_to_anchor
            legend_kwargs: dict[str, Any] = {}
        else:
            legend_anchor_ax = anchor_ax or ax
            if self.legend_offset_use_points:
                translate = mtransforms.ScaledTranslation(
                    (float(self.legend_offset_points) + float(extra_offset_points)) / 72.0,
                    0.0,
                    fig.dpi_scale_trans,
                )
                legend_kwargs = {"bbox_transform": legend_anchor_ax.transAxes + translate}
                bbox_to_anchor = (1.0, 0.5)
            else:
                legend_kwargs = {"bbox_transform": legend_anchor_ax.transAxes}
                bbox_to_anchor = (1 + self.legend_offset + extra_offset_points / 72.0, 0.5)

        legend_family = resolve_font_family()
        lgd = ax.legend(
            handles=handles,
            bbox_to_anchor=bbox_to_anchor,
            loc="center left",
            frameon=False,
            handlelength=1,
            handleheight=1,
            prop=font_manager.FontProperties(family=legend_family, size=self.legend_fontsize),
            title_fontsize=self.legend_title_fontsize,
            **legend_kwargs,
        )
        if legend_family:
            lgd.get_title().set_fontproperties(
                font_manager.FontProperties(family=legend_family, size=self.legend_title_fontsize)
            )
        for text in lgd.get_texts():
            if text.get_text() in label_headers:
                text.set_fontweight("bold")

    def _base_plot_state(
        self,
    ) -> tuple[
        pd.DataFrame,
        list[dict[str, Any]],
        pd.DataFrame,
        list[Any],
        list[float],
        dict[Any, float],
        list[float],
    ]:
        plot_df = self._prepare_plot_df()
        columns, sample_meta = self._build_aggregate_columns()
        genes_ordered, row_positions, gene_to_idx = self._compute_gene_rows()
        col_positions: list[float] = []
        current_x = 0.0
        for idx, column in enumerate(columns):
            col_positions.append(current_x)
            current_x += self.cell_aspect
            is_all_boundary = (
                idx == 0
                and column.get("title") == "All"
                and len(columns) > 1
                and self.include_overall
            )
            is_group_boundary = self.separate_all_columns and idx < len(columns) - 1
            if self.all_column_gap > 0 and (is_all_boundary or is_group_boundary):
                current_x += self.all_column_gap
        return (
            plot_df,
            columns,
            sample_meta,
            genes_ordered,
            row_positions,
            gene_to_idx,
            col_positions,
        )


class OncoPrevalencePlotter(_OncoAggregatePlotterBase):
    def __init__(
        self,
        df: pd.DataFrame,
        config: OncoplotConfig,
        row_groups: pd.DataFrame | None = None,
        row_groups_color_dict: dict[str, str] | None = None,
        style=None,
        *,
        group_by: list[str] | None = None,
        include_overall: bool = True,
        show_all: bool | None = None,
        all_column_gap: float = 0.18,
        separate_all_columns: bool = False,
        percent_decimals: int = 0,
        annotate_values: bool = True,
        show_total_counts: bool = False,
        show_category_breakdown: bool = False,
        show_category_counts: bool = False,
        label_fontsize: float | int | None = None,
        label_text_color: str | None = None,
        label_bbox_mode: Literal["auto", "always", "multiline-only", "never"] = "auto",
        label_bbox_facecolor: str = "white",
        label_bbox_alpha: float = 0.78,
        label_bbox_edgecolor: str = "none",
        label_bbox_boxstyle: str = "round,pad=0.18",
        cmap: str = "Blues",
        vmin: float = 0.0,
        vmax: float = 100.0,
        colorbar_title: str = "Altered (%)",
    ) -> None:
        super().__init__(
            df=df,
            config=config,
            row_groups=row_groups,
            row_groups_color_dict=row_groups_color_dict,
            style=style,
            group_by=group_by,
            include_overall=include_overall,
            show_all=show_all,
            all_column_gap=all_column_gap,
            separate_all_columns=separate_all_columns,
            percent_decimals=percent_decimals,
            annotate_values=annotate_values,
            show_total_counts=show_total_counts,
            show_category_breakdown=show_category_breakdown,
            show_category_counts=show_category_counts,
            label_fontsize=label_fontsize,
            label_text_color=label_text_color,
            label_bbox_mode=label_bbox_mode,
            label_bbox_facecolor=label_bbox_facecolor,
            label_bbox_alpha=label_bbox_alpha,
            label_bbox_edgecolor=label_bbox_edgecolor,
            label_bbox_boxstyle=label_bbox_boxstyle,
        )
        self.cmap = plt.get_cmap(cmap)
        self.norm = Normalize(vmin=vmin, vmax=vmax)
        self.colorbar_title = colorbar_title

    def plot(self) -> plt.Figure:
        plot_df, columns, sample_meta, genes_ordered, row_positions, gene_to_idx, col_positions = (
            self._base_plot_state()
        )
        nrows = int(np.ceil(max(row_positions) + 1 if row_positions else 1))
        max_x = (max(col_positions) + self.cell_aspect) if col_positions else self.cell_aspect
        fig, ax, rowlabel_transform, rowlabel_base_x = self._setup_axes(
            len(columns),
            nrows,
            max_x,
            getattr(self.config, "fig_title", None),
            getattr(self.config, "fig_title_fontsize", 22),
        )

        for x_idx, column in enumerate(columns):
            denom = len(pd.Index(column["samples"]).drop_duplicates())
            column_df = plot_df[plot_df[self.x_col].isin(column["samples"])]
            for gene, y in zip(genes_ordered, row_positions, strict=True):
                gene_df = column_df[column_df[self.y_col] == gene]
                altered = gene_df[self.x_col].drop_duplicates().nunique()
                percent = 100.0 * altered / denom if denom else 0.0
                face = self.cmap(self.norm(percent))
                rect = plt.Rectangle(
                    (col_positions[x_idx], y),
                    self.cell_aspect,
                    1.0,
                    facecolor=face,
                    edgecolor="#D9D9D9",
                    linewidth=0.8,
                )
                ax.add_patch(rect)
                label = self._build_cell_label(
                    total_count=altered,
                    denom=denom,
                    counts_by_type=(
                        gene_df[[self.x_col, self.value_col]]
                        .drop_duplicates()
                        .assign(_value=lambda frame: frame[self.value_col].astype(str))
                        .groupby("_value")
                        .size()
                        .to_dict()
                    ),
                    event_types=self._event_types(column_df),
                )
                if label is not None:
                    text_color, extra_kwargs = self._get_label_style(face, label)
                    ax.text(
                        col_positions[x_idx] + self.cell_aspect / 2,
                        y + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=self._label_font_size(),
                        color=text_color,
                        linespacing=1.15,
                        **extra_kwargs,
                    )

        resolved_top_annotations = self._draw_top_annotations(
            ax,
            columns,
            col_positions,
            sample_meta,
            rowlabel_transform,
            rowlabel_base_x,
        )
        self._draw_gene_labels(
            fig=fig,
            ax=ax,
            genes_ordered=genes_ordered,
            row_positions=row_positions,
            rowlabel_text_transform=rowlabel_transform,
            rowlabel_base_x=rowlabel_base_x,
        )
        heatmap_left = min(col_positions) if col_positions else 0.0
        self._draw_row_groups(ax, fig, genes_ordered, gene_to_idx, heatmap_left)
        self._draw_column_labels(ax, columns, col_positions)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False)
        ax.grid(False)
        self._finalize_layout(fig, ax)

        mappable = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        colorbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.06)
        colorbar.set_label(self.colorbar_title, fontsize=self.legend_title_fontsize)
        colorbar.ax.tick_params(labelsize=self.legend_fontsize)

        self._draw_legend(
            fig, ax, resolved_top_annotations, anchor_ax=colorbar.ax, extra_offset_points=34.0
        )
        fig.canvas.draw()
        return fig


class OncoPrevalenceRasterPlotter(_OncoAggregatePlotterBase):
    def __init__(
        self,
        df: pd.DataFrame,
        config: OncoplotConfig,
        row_groups: pd.DataFrame | None = None,
        row_groups_color_dict: dict[str, str] | None = None,
        style=None,
        *,
        group_by: list[str] | None = None,
        include_overall: bool = True,
        show_all: bool | None = None,
        all_column_gap: float = 0.18,
        separate_all_columns: bool = False,
        percent_decimals: int = 0,
        annotate_values: bool = True,
        show_total_counts: bool = True,
        show_category_breakdown: bool = False,
        show_category_counts: bool = False,
        label_fontsize: float | int | None = None,
        label_text_color: str | None = None,
        label_bbox_mode: Literal["auto", "always", "multiline-only", "never"] = "auto",
        label_bbox_facecolor: str = "white",
        label_bbox_alpha: float = 0.78,
        label_bbox_edgecolor: str = "none",
        label_bbox_boxstyle: str = "round,pad=0.18",
        event_band_order: list[str] | None = None,
        heatmap_width_fraction: float = 0.74,
        empty_cell_color: str = "#D9D9D9",
        empty_band_color: str | None = None,
        cell_border_color: str | None = "#D9D9D9",
        cell_border_linewidth: float = 0.8,
        row_separator_color: str | None = None,
        row_separator_linewidth: float = 0.0,
        slot_padding_fraction: float = 0.04,
        label_offset_fraction: float = 0.14,
    ) -> None:
        super().__init__(
            df=df,
            config=config,
            row_groups=row_groups,
            row_groups_color_dict=row_groups_color_dict,
            style=style,
            group_by=group_by,
            include_overall=include_overall,
            show_all=show_all,
            all_column_gap=all_column_gap,
            separate_all_columns=separate_all_columns,
            percent_decimals=percent_decimals,
            annotate_values=annotate_values,
            show_total_counts=show_total_counts,
            show_category_breakdown=show_category_breakdown,
            show_category_counts=show_category_counts,
            label_fontsize=label_fontsize,
            label_text_color=label_text_color,
            label_bbox_mode=label_bbox_mode,
            label_bbox_facecolor=label_bbox_facecolor,
            label_bbox_alpha=label_bbox_alpha,
            label_bbox_edgecolor=label_bbox_edgecolor,
            label_bbox_boxstyle=label_bbox_boxstyle,
        )
        self.event_band_order = (
            [str(value) for value in event_band_order] if event_band_order else []
        )
        self.heatmap_width_fraction = min(max(float(heatmap_width_fraction), 0.35), 1.0)
        self.empty_cell_color = _ensure_opaque_color(empty_cell_color, default="#D9D9D9")
        fallback_band = empty_band_color if empty_band_color is not None else empty_cell_color
        self.empty_band_color = _ensure_opaque_color(fallback_band, default=self.empty_cell_color)
        self.cell_border_color = cell_border_color
        self.cell_border_linewidth = max(float(cell_border_linewidth), 0.0)
        self.row_separator_color = row_separator_color
        self.row_separator_linewidth = max(float(row_separator_linewidth), 0.0)
        self.slot_padding_fraction = min(max(float(slot_padding_fraction), 0.0), 0.3)
        self.label_offset_fraction = min(max(float(label_offset_fraction), 0.0), 0.8)

    def plot(self) -> plt.Figure:
        plot_df, columns, sample_meta, genes_ordered, row_positions, gene_to_idx, col_positions = (
            self._base_plot_state()
        )
        ordered_event_types = self.event_band_order or self._event_types(plot_df)
        seen_event_types = {str(value) for value in ordered_event_types}
        event_types = list(ordered_event_types)
        for value in self._event_types(plot_df):
            if str(value) not in seen_event_types:
                event_types.append(str(value))
                seen_event_types.add(str(value))
        if not event_types:
            event_types = ["Altered"]

        nrows = int(np.ceil(max(row_positions) + 1 if row_positions else 1))
        max_x = (max(col_positions) + self.cell_aspect) if col_positions else self.cell_aspect
        fig, ax, rowlabel_transform, rowlabel_base_x = self._setup_axes(
            len(columns),
            nrows,
            max_x,
            getattr(self.config, "fig_title", None),
            getattr(self.config, "fig_title_fontsize", 22),
        )

        bar_colors = dict(getattr(self.heatmap_annotation, "colors", {}) or {})
        heatmap_width = self.cell_aspect * self.heatmap_width_fraction
        label_width = max(self.cell_aspect - heatmap_width, 0.0)
        heatmap_width = self.cell_aspect - label_width
        band_height = 1.0 / max(len(event_types), 1)
        seam_overlap = min(max(heatmap_width * 0.002, 0.0015), 0.01)

        for x_idx, column in enumerate(columns):
            samples = [
                str(sample) for sample in pd.Index(column["samples"]).drop_duplicates().tolist()
            ]
            denom = len(samples)
            column_df = plot_df[plot_df[self.x_col].isin(column["samples"])].copy()
            if self.x_col in column_df.columns:
                column_df[self.x_col] = column_df[self.x_col].astype(str)
            for gene, y in zip(genes_ordered, row_positions, strict=True):
                gene_df = column_df[column_df[self.y_col] == gene]
                border_color = (
                    self.cell_border_color if self.cell_border_color is not None else "none"
                )
                border_width = (
                    self.cell_border_linewidth if self.cell_border_color is not None else 0.0
                )
                heatmap_left = col_positions[x_idx]
                heatmap_right = heatmap_left + heatmap_width
                cell_patch = plt.Rectangle(
                    (heatmap_left - seam_overlap, y),
                    heatmap_width + (2.0 * seam_overlap),
                    1.0,
                    facecolor=self.empty_band_color,
                    edgecolor=border_color,
                    linewidth=border_width,
                    antialiased=False,
                    clip_on=False,
                )
                ax.add_patch(cell_patch)

                counts_by_type: dict[str, float] = {}
                altered_samples: set[str] = set()
                if denom > 0 and not gene_df.empty:
                    gene_events = (
                        gene_df[[self.x_col, self.value_col]]
                        .dropna(subset=[self.x_col])
                        .drop_duplicates()
                        .assign(
                            _sample=lambda frame: frame[self.x_col].astype(str),
                            _value=lambda frame: frame[self.value_col].astype(str),
                        )
                    )
                    sample_event_lookup: dict[str, set[str]] = {}
                    for sample, event_type in gene_events[["_sample", "_value"]].itertuples(
                        index=False
                    ):
                        sample_event_lookup.setdefault(str(sample), set()).add(str(event_type))

                    slot_edges = np.linspace(heatmap_left, heatmap_right, denom + 1)
                    for sample_idx, sample in enumerate(samples):
                        present_types = sample_event_lookup.get(sample, set())
                        if not present_types:
                            continue
                        altered_samples.add(sample)
                        slot_left = float(slot_edges[sample_idx])
                        slot_right = float(slot_edges[sample_idx + 1])
                        draw_left = max(heatmap_left, slot_left - seam_overlap)
                        draw_right = min(heatmap_right, slot_right + seam_overlap)
                        draw_width = max(draw_right - draw_left, 0.0)
                        for band_idx, event_type in enumerate(event_types):
                            if str(event_type) not in present_types:
                                continue
                            counts_by_type[str(event_type)] = (
                                counts_by_type.get(str(event_type), 0.0) + 1.0
                            )
                            band_patch = plt.Rectangle(
                                (draw_left, y + band_idx * band_height),
                                draw_width,
                                band_height,
                                facecolor=_ensure_opaque_color(
                                    bar_colors.get(str(event_type), "#808080"), default="#808080"
                                ),
                                edgecolor="none",
                                linewidth=0,
                                antialiased=False,
                                clip_on=False,
                            )
                            ax.add_patch(band_patch)

                label = self._build_cell_label(
                    total_count=len(altered_samples),
                    denom=denom,
                    counts_by_type=counts_by_type,
                    event_types=event_types,
                )
                if label is not None:
                    dominant_face = (
                        bar_colors.get(str(event_types[0]), self.empty_cell_color)
                        if event_types
                        else self.empty_cell_color
                    )
                    text_color, extra_kwargs = self._get_label_style(dominant_face, label)
                    ax.text(
                        heatmap_right + max(label_width * self.label_offset_fraction, 0.09),
                        y + 0.5,
                        label,
                        ha="left",
                        va="center",
                        fontsize=self._label_font_size(),
                        color=text_color,
                        linespacing=1.15,
                        **extra_kwargs,
                    )

        self._draw_row_separators(ax, col_positions, heatmap_width, row_positions, seam_overlap)

        resolved_top_annotations = self._draw_raster_top_annotations(
            ax,
            columns,
            col_positions,
            sample_meta,
            rowlabel_transform,
            rowlabel_base_x,
        )
        self._draw_gene_labels(
            fig=fig,
            ax=ax,
            genes_ordered=genes_ordered,
            row_positions=row_positions,
            rowlabel_text_transform=rowlabel_transform,
            rowlabel_base_x=rowlabel_base_x,
        )
        heatmap_left = min(col_positions) if col_positions else 0.0
        self._draw_row_groups(ax, fig, genes_ordered, gene_to_idx, heatmap_left)
        self._draw_column_labels(ax, columns, col_positions)

        mutation_handles: list[Patch] = []
        for event_type in event_types:
            if str(event_type) not in bar_colors:
                continue
            color = bar_colors[str(event_type)]
            if is_white_color(color):
                mutation_handles.append(
                    Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=str(event_type))
                )
            else:
                mutation_handles.append(Patch(facecolor=color, label=str(event_type)))

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False)
        ax.grid(False)
        self._finalize_layout(fig, ax)

        mutation_title = getattr(self.heatmap_annotation, "legend_title", None) or "Mutation Type"
        self._draw_legend(fig, ax, resolved_top_annotations, mutation_handles, mutation_title)
        fig.canvas.draw()
        return fig

    def _draw_row_separators(
        self,
        ax: plt.Axes,
        col_positions: list[float],
        heatmap_width: float,
        row_positions: list[float],
        seam_overlap: float,
    ) -> None:
        if (
            self.row_separator_color is None
            or self.row_separator_linewidth <= 0.0
            or not col_positions
            or len(row_positions) < 2
        ):
            return

        separator_ys: list[float] = []
        for current_y, next_y in zip(row_positions, row_positions[1:], strict=True):
            if np.isclose(next_y - current_y, 1.0):
                separator_ys.append(current_y + 1.0)

        if not separator_ys:
            return

        for heatmap_left in col_positions:
            heatmap_right = heatmap_left + heatmap_width
            for separator_y in separator_ys:
                ax.plot(
                    [heatmap_left - seam_overlap, heatmap_right + seam_overlap],
                    [separator_y, separator_y],
                    color=self.row_separator_color,
                    linewidth=self.row_separator_linewidth,
                    solid_capstyle="butt",
                    antialiased=False,
                    clip_on=False,
                    zorder=3,
                )

    def _draw_raster_top_annotations(
        self,
        ax: plt.Axes,
        columns: list[dict[str, Any]],
        col_positions: list[float],
        sample_meta: pd.DataFrame,
        rowlabel_text_transform,
        rowlabel_base_x: float,
    ) -> dict[str, TopAnnotationConfig]:
        if not self.top_annotations:
            return {}

        annotation_y = self.top_annotation_inter_spacer * -1
        annotation_order = (
            [name for name in self.config.top_annotation_order if name in self.top_annotations][
                ::-1
            ]
            if self.config.top_annotation_order
            else list(self.top_annotations.keys())
        )
        for name in self.top_annotations:
            if name not in annotation_order:
                annotation_order.append(name)

        heatmap_width = self.cell_aspect * self.heatmap_width_fraction
        heatmap_left = min(col_positions) if col_positions else 0.0
        if self.top_annotation_label_use_points:
            label_transform = rowlabel_text_transform
            label_x = rowlabel_base_x
        else:
            label_transform = ax.transData
            label_x = heatmap_left - float(self.top_annotation_label_offset)

        resolved: dict[str, TopAnnotationConfig] = {}
        for ann_name in annotation_order:
            ann_config = self.top_annotations[ann_name]
            sample_series = self._resolve_annotation_sample_series(ann_config, sample_meta)
            self._draw_patient_aligned_top_annotation(
                ax=ax,
                columns=columns,
                col_positions=col_positions,
                annotation_y=annotation_y,
                ann_config=ann_config,
                ann_name=ann_name,
                sample_series=sample_series,
                label_x=label_x,
                label_transform=label_transform,
                heatmap_width=heatmap_width,
            )
            resolved[ann_name] = ann_config
            annotation_y -= ann_config.height + self.top_annotation_intra_spacer
        return resolved

    def _draw_patient_aligned_top_annotation(
        self,
        *,
        ax: plt.Axes,
        columns: list[dict[str, Any]],
        col_positions: list[float],
        annotation_y: float,
        ann_config: TopAnnotationConfig,
        ann_name: str,
        sample_series: pd.Series,
        label_x: float,
        label_transform,
        heatmap_width: float,
    ) -> None:
        if not col_positions:
            return

        display_name = ann_config.display_name or ann_name
        height = ann_config.height
        border_color = getattr(ann_config, "border_color", "black")
        border_width = getattr(ann_config, "border_width", 0.5)
        sample_series = pd.Series(sample_series)
        if not sample_series.empty:
            sample_series.index = sample_series.index.astype(str)

        for column, x in zip(columns, col_positions, strict=True):
            samples = [
                str(sample) for sample in pd.Index(column["samples"]).drop_duplicates().tolist()
            ]
            if ann_config.na_color not in {None, "none", "None", "transparent"}:
                ax.add_patch(
                    plt.Rectangle(
                        (x, annotation_y),
                        heatmap_width,
                        height,
                        facecolor=ann_config.na_color,
                        edgecolor="none",
                        linewidth=0,
                        clip_on=False,
                        zorder=10,
                    )
                )

            if samples:
                slot_width = heatmap_width / max(len(samples), 1)
                run_start = None
                run_color = None
                for sample_idx, sample in enumerate(samples):
                    value = sample_series.get(sample)
                    color = None
                    if not pd.isna(value) and str(value).strip() != "":
                        color = ann_config.colors.get(str(value), ann_config.na_color)

                    if color == run_color:
                        continue

                    if (
                        run_color not in {None, "none", "None", "transparent"}
                        and run_start is not None
                    ):
                        ax.add_patch(
                            plt.Rectangle(
                                (x + run_start * slot_width, annotation_y),
                                (sample_idx - run_start) * slot_width,
                                height,
                                facecolor=run_color,
                                edgecolor="none",
                                linewidth=0,
                                antialiased=False,
                                clip_on=False,
                                zorder=11,
                            )
                        )

                    run_start = sample_idx
                    run_color = color

                if run_color not in {None, "none", "None", "transparent"} and run_start is not None:
                    ax.add_patch(
                        plt.Rectangle(
                            (x + run_start * slot_width, annotation_y),
                            (len(samples) - run_start) * slot_width,
                            height,
                            facecolor=run_color,
                            edgecolor="none",
                            linewidth=0,
                            antialiased=False,
                            clip_on=False,
                            zorder=11,
                        )
                    )

            if ann_config.draw_border:
                ax.add_patch(
                    plt.Rectangle(
                        (x, annotation_y),
                        heatmap_width,
                        height,
                        fill=False,
                        edgecolor=border_color,
                        linewidth=border_width,
                        clip_on=False,
                        zorder=12,
                    )
                )

        ax.text(
            label_x,
            annotation_y + height / 2,
            display_name,
            ha="right",
            va="center",
            fontsize=ann_config.fontsize,
            fontweight="normal",
            clip_on=False,
            zorder=13,
            transform=label_transform,
        )


class OncoGeneBarPlotter(_OncoAggregatePlotterBase):
    def __init__(
        self,
        df: pd.DataFrame,
        config: OncoplotConfig,
        row_groups: pd.DataFrame | None = None,
        row_groups_color_dict: dict[str, str] | None = None,
        style=None,
        *,
        group_by: list[str] | None = None,
        include_overall: bool = True,
        show_all: bool | None = None,
        all_column_gap: float = 0.18,
        separate_all_columns: bool = False,
        percent_decimals: int = 0,
        annotate_values: bool = True,
        show_total_counts: bool = False,
        show_category_breakdown: bool = False,
        show_category_counts: bool = False,
        label_fontsize: float | int | None = None,
        label_text_color: str | None = None,
        label_bbox_mode: Literal["auto", "always", "multiline-only", "never"] = "auto",
        label_bbox_facecolor: str = "white",
        label_bbox_alpha: float = 0.78,
        label_bbox_edgecolor: str = "none",
        label_bbox_boxstyle: str = "round,pad=0.18",
        bar_height_fraction: float = 1.0,
        empty_cell_color: str = "white",
        empty_label_text_color: str | None = None,
        cell_border_color: str | None = "#D9D9D9",
        cell_border_linewidth: float = 0.8,
    ) -> None:
        super().__init__(
            df=df,
            config=config,
            row_groups=row_groups,
            row_groups_color_dict=row_groups_color_dict,
            style=style,
            group_by=group_by,
            include_overall=include_overall,
            show_all=show_all,
            all_column_gap=all_column_gap,
            separate_all_columns=separate_all_columns,
            percent_decimals=percent_decimals,
            annotate_values=annotate_values,
            show_total_counts=show_total_counts,
            show_category_breakdown=show_category_breakdown,
            show_category_counts=show_category_counts,
            label_fontsize=label_fontsize,
            label_text_color=label_text_color,
            label_bbox_mode=label_bbox_mode,
            label_bbox_facecolor=label_bbox_facecolor,
            label_bbox_alpha=label_bbox_alpha,
            label_bbox_edgecolor=label_bbox_edgecolor,
            label_bbox_boxstyle=label_bbox_boxstyle,
        )
        self.bar_height_fraction = max(0.2, min(bar_height_fraction, 1.0))
        self.empty_cell_color = empty_cell_color
        self.empty_label_text_color = empty_label_text_color
        self.cell_border_color = cell_border_color
        self.cell_border_linewidth = max(float(cell_border_linewidth), 0.0)

    def plot(self) -> plt.Figure:
        plot_df, columns, sample_meta, genes_ordered, row_positions, gene_to_idx, col_positions = (
            self._base_plot_state()
        )
        event_types = self._event_types(plot_df)
        nrows = int(np.ceil(max(row_positions) + 1 if row_positions else 1))
        max_x = (max(col_positions) + self.cell_aspect) if col_positions else self.cell_aspect
        fig, ax, rowlabel_transform, rowlabel_base_x = self._setup_axes(
            len(columns),
            nrows,
            max_x,
            getattr(self.config, "fig_title", None),
            getattr(self.config, "fig_title_fontsize", 22),
        )

        bar_colors = dict(getattr(self.heatmap_annotation, "colors", {}) or {})
        bar_y_pad = (1.0 - self.bar_height_fraction) / 2
        for x_idx, column in enumerate(columns):
            denom = len(pd.Index(column["samples"]).drop_duplicates())
            column_df = plot_df[plot_df[self.x_col].isin(column["samples"])]
            for gene, y in zip(genes_ordered, row_positions, strict=True):
                gene_df = column_df[column_df[self.y_col] == gene]
                border_color = (
                    self.cell_border_color if self.cell_border_color is not None else "none"
                )
                border_width = (
                    self.cell_border_linewidth if self.cell_border_color is not None else 0.0
                )
                background = plt.Rectangle(
                    (col_positions[x_idx], y),
                    self.cell_aspect,
                    1.0,
                    facecolor=_ensure_opaque_color(self.empty_cell_color, default="white"),
                    edgecolor=border_color,
                    linewidth=border_width,
                    clip_on=False,
                )
                ax.add_patch(background)
                if denom == 0 or gene_df.empty:
                    if self.annotate_values:
                        label = f"{0:.{self.percent_decimals}f}%"
                        if self.empty_label_text_color is not None:
                            text_color = str(self.empty_label_text_color)
                            extra_kwargs: dict[str, Any] = {}
                        else:
                            text_color, extra_kwargs = self._get_label_style(
                                _ensure_opaque_color(self.empty_cell_color, default="white"),
                                label,
                            )
                        ax.text(
                            col_positions[x_idx] + self.cell_aspect / 2,
                            y + 0.5,
                            label,
                            ha="center",
                            va="center",
                            fontsize=self._label_font_size(),
                            color=text_color,
                            linespacing=1.15,
                            **extra_kwargs,
                        )
                    continue

                counts_by_type = (
                    gene_df[[self.x_col, self.value_col]]
                    .drop_duplicates()
                    .assign(_value=lambda frame: frame[self.value_col].astype(str))
                    .groupby("_value")
                    .size()
                )
                total_altered = gene_df[self.x_col].drop_duplicates().nunique()
                left = col_positions[x_idx]
                for event_type in event_types:
                    count = float(counts_by_type.get(str(event_type), 0.0))
                    if count <= 0:
                        continue
                    width = self.cell_aspect * (count / denom)
                    patch = plt.Rectangle(
                        (left, y + bar_y_pad),
                        width,
                        self.bar_height_fraction,
                        facecolor=_ensure_opaque_color(
                            bar_colors.get(str(event_type), "#808080"), default="#808080"
                        ),
                        edgecolor="none",
                        linewidth=0,
                    )
                    ax.add_patch(patch)
                    left += width
                label = self._build_cell_label(
                    total_count=total_altered,
                    denom=denom,
                    counts_by_type=counts_by_type.to_dict(),
                    event_types=event_types,
                )
                if label is not None:
                    dominant_face = (
                        bar_colors.get(str(event_types[0]), "white") if event_types else "white"
                    )
                    text_color, extra_kwargs = self._get_label_style(dominant_face, label)
                    ax.text(
                        col_positions[x_idx] + self.cell_aspect / 2,
                        y + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=self._label_font_size(),
                        color=text_color,
                        linespacing=1.15,
                        **extra_kwargs,
                    )

        resolved_top_annotations = self._draw_top_annotations(
            ax,
            columns,
            col_positions,
            sample_meta,
            rowlabel_transform,
            rowlabel_base_x,
        )
        self._draw_gene_labels(
            fig=fig,
            ax=ax,
            genes_ordered=genes_ordered,
            row_positions=row_positions,
            rowlabel_text_transform=rowlabel_transform,
            rowlabel_base_x=rowlabel_base_x,
        )
        heatmap_left = min(col_positions) if col_positions else 0.0
        self._draw_row_groups(ax, fig, genes_ordered, gene_to_idx, heatmap_left)
        self._draw_column_labels(ax, columns, col_positions)

        mutation_handles: list[Patch] = []
        for event_type in event_types:
            if str(event_type) not in bar_colors:
                continue
            color = bar_colors[str(event_type)]
            if is_white_color(color):
                mutation_handles.append(
                    Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=str(event_type))
                )
            else:
                mutation_handles.append(Patch(facecolor=color, label=str(event_type)))

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="y", left=False)
        ax.grid(False)
        self._finalize_layout(fig, ax)

        mutation_title = getattr(self.heatmap_annotation, "legend_title", None) or "Mutation Type"
        self._draw_legend(fig, ax, resolved_top_annotations, mutation_handles, mutation_title)
        fig.canvas.draw()
        return fig
