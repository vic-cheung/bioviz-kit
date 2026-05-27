"""Clinical Forest Plotter for oncology trial publications.

This plotter creates publication-ready forest plots with table-style layout
including Events/Patients, HR (95% CI), Median survival, and p-values.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from bioviz.configs.clinical_forest_cfg import ClinicalForestPlotConfig
from bioviz.configs.forest_cfg import ForestPlotConfig
from bioviz.plots.forest import ForestPlotter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class ClinicalForestPlotter:
    """Publication-ready clinical forest plot with table annotations.

    This plotter creates forest plots designed for clinical trial publications,
    featuring a table layout with Events/Patients counts, HR with CI, median
    survival times for both arms, and p-values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing hazard ratio and clinical outcome data.
    config : ClinicalForestPlotConfig
        Configuration options for the plot.

    Examples
    --------
    >>> from bioviz import ClinicalForestPlotter, ClinicalForestPlotConfig
    >>> config = ClinicalForestPlotConfig(title="OS Forest Plot")
    >>> plotter = ClinicalForestPlotter(df, config)
    >>> fig, ax = plotter.plot()
    >>> fig.savefig("forest_plot.pdf", bbox_inches="tight")
    """

    def __init__(self, data: pd.DataFrame, config: ClinicalForestPlotConfig | None = None):
        self.data = data.copy()
        self.config = config or ClinicalForestPlotConfig()
        self._base_config: ForestPlotConfig | None = None
        self._base_plotter: ForestPlotter | None = None

    # ==========================================================================
    # Public API
    # ==========================================================================

    def plot(self) -> tuple[Figure, Axes]:
        """Generate the clinical forest plot.

        Returns
        -------
        tuple[Figure, Axes]
            Matplotlib figure and axes containing the plot.
        """
        df = self._prepare_display_labels()
        n_rows = max(len(df), 1)

        xlim, xticks = self._compute_x_bounds(df)
        base_config = self._make_base_config(n_rows, xlim, xticks)
        base_plotter = ForestPlotter(df, base_config)

        fig, ax = base_plotter.plot()

        self._base_config = base_config
        self._base_plotter = base_plotter

        self._apply_clinical_layout(fig, ax, base_plotter, df)

        return fig, ax

    # ==========================================================================
    # Data preparation
    # ==========================================================================

    def _prepare_display_labels(self) -> pd.DataFrame:
        """Ensure display_label column exists."""
        df = self.data.copy()
        cfg = self.config

        if cfg.label_col in df.columns and cfg.label_col != "display_label":
            df["display_label"] = df[cfg.label_col]
        elif "display_label" not in df.columns:
            if cfg.comparator_label_col and cfg.comparator_label_col in df.columns:
                df["display_label"] = df[cfg.comparator_label_col].astype(str)
            else:
                df["display_label"] = df.index.astype(str)

        return df.reset_index(drop=True)

    def _compute_x_bounds(self, df: pd.DataFrame) -> tuple[tuple[float, float], list[float]]:
        """Compute bounded x-axis limits and tick positions."""
        cfg = self.config

        if cfg.xlim is not None:
            xlim = cfg.xlim
            if cfg.xticks is not None:
                return xlim, cfg.xticks
            xticks = self._generate_xticks(xlim[0], xlim[1])
            return xlim, xticks

        ci_lower_col = cfg.ci_lower_col
        ci_upper_col = cfg.ci_upper_col

        # Use CI upper bound (not just HR) to determine x_upper
        ci_upper = pd.to_numeric(df[ci_upper_col], errors="coerce")
        ci_upper_max = float(ci_upper.dropna().max()) if ci_upper.notna().any() else 1.0

        ci_lower = pd.to_numeric(df[ci_lower_col], errors="coerce")
        positive_lower = ci_lower[ci_lower > 0]
        lower_bound = float(positive_lower.min()) if not positive_lower.empty else 0.5
        x_lower = min(0.5, max(0.0, lower_bound * 0.85))

        # Use CI upper max with padding, minimum of 1.5 (enough to show reference line)
        proposed_upper = max(1.5, ci_upper_max * 1.15)
        x_upper = min(cfg.x_max_cap, proposed_upper)

        xticks = self._generate_xticks(x_lower, x_upper)

        return (x_lower, x_upper), xticks

    @staticmethod
    def _generate_xticks(x_lower: float, x_upper: float) -> list[float]:
        """Generate tick positions based on axis range."""
        tick_step = 0.5 if x_upper <= 4.0 else 1.0
        xticks = []
        current = 0.5
        # Include ticks up to and slightly beyond x_upper for a cleaner axis
        while current <= x_upper + 0.01:
            xticks.append(round(current, 2))
            current += tick_step
        return xticks

    def _make_base_config(
        self, n_rows: int, xlim: tuple[float, float], xticks: list[float]
    ) -> ForestPlotConfig:
        """Build a ForestPlotConfig for the underlying plotter."""
        cfg = self.config

        if cfg.figsize is not None:
            figsize = cfg.figsize
        else:
            height = min(
                cfg.max_figure_height,
                max(cfg.min_figure_height, cfg.base_height + cfg.row_height * n_rows),
            )
            figsize = (cfg.figure_width, height)

        if cfg.capsize is not None:
            capsize = cfg.capsize
        elif n_rows <= 1:
            capsize = 4
        elif n_rows <= 4:
            capsize = 5
        else:
            capsize = 6

        return ForestPlotConfig.model_validate(
            {
                "hr_col": cfg.hr_col,
                "ci_lower_col": cfg.ci_lower_col,
                "ci_upper_col": cfg.ci_upper_col,
                "label_col": "display_label",
                "pvalue_col": cfg.pvalue_col,
                "reference_col": None,
                "variable_col": None,
                "title": None,
                "xlabel": cfg.xlabel,
                "figsize": figsize,
                "log_scale": False,
                "show_reference_line": cfg.show_reference_line,
                "reference_line_color": cfg.reference_line_color,
                "reference_line_style": cfg.reference_line_style,
                "reference_line_width": cfg.reference_line_width,
                "color_significant": cfg.marker_color,
                "color_nonsignificant": cfg.marker_color,
                "marker_color_significant": cfg.marker_color,
                "marker_color_nonsignificant": cfg.marker_color,
                "show_stats_table": False,
                "show_section_separators": False,
                "marker_style": cfg.marker_style,
                "marker_size": cfg.marker_size,
                "linewidth": cfg.linewidth,
                "show_caps": cfg.show_caps,
                "capsize": capsize,
                "show_grid": False,
                "center_around_null": False,
                "xlim": xlim,
                "xticks": xticks,
                "show_y_spine": False,
                "show_yticks": False,
                "ytick_fontsize": int(cfg.axis_fontsize),
                "xtick_fontsize": int(cfg.axis_fontsize),
                "xlabel_fontsize": int(cfg.xlabel_fontsize),
                "title_fontsize": int(cfg.title_fontsize) if cfg.title_fontsize else 12,
                "stats_fontsize": int(cfg.cell_fontsize),
            }
        )

    # ==========================================================================
    # Layout and rendering
    # ==========================================================================

    def _apply_clinical_layout(
        self,
        fig: Figure,
        ax: Axes,
        plotter: ForestPlotter,
        df: pd.DataFrame,
    ) -> None:
        """Apply the clinical table layout to the forest plot."""
        cfg = self.config

        prepared_df = plotter.prepare_data()
        y_positions = plotter.compute_y_positions(prepared_df)
        n_rows = max(len(prepared_df), 1)

        title = cfg.title or ""
        title_wrap_width = cfg.title_wrap_width or (
            40 if n_rows <= 1 else 46 if n_rows <= 3 else 52
        )
        title = self._wrap_title(title, width=title_wrap_width)
        title_line_count = self._text_block_line_count(title)
        _fig_width_in, fig_height_in = fig.get_size_inches()

        # Scale vertical offsets down on very tall figures so title/header/footer
        # spacing remains visually consistent instead of stretching with height.
        vertical_scale = min(1.0, 12.0 / max(fig_height_in, 1.0))

        def _scale_about_anchor(value: float, anchor: float) -> float:
            return anchor + ((value - anchor) * vertical_scale)

        if len(y_positions) > 0:
            if n_rows <= 1:
                bottom_padding = 0.24
                top_padding = 0.32
            else:
                bottom_padding = 0.45
                # Keep the header-to-first-row gap visually stable on tall plots.
                # Padding tied to total row count creates a huge empty band above
                # the data block once the figure height grows.
                top_padding = 0.72 + (0.1 * max(title_line_count - 1, 0))
            ax.set_ylim(y_positions.min() - bottom_padding, y_positions.max() + top_padding)

        if n_rows <= 1:
            bottom_margin_in = 0.9
            top_margin_in = 1.35 + (0.18 * max(title_line_count - 2, 0))
            title_gap_in = 0.28
        else:
            bottom_margin_in = 1.05
            top_margin_in = 1.25 + (0.16 * max(title_line_count - 2, 0))
            title_gap_in = 0.22

        bottom_margin = min(0.3, bottom_margin_in / max(fig_height_in, 1.0))
        top_margin = max(0.72, 1.0 - (top_margin_in / max(fig_height_in, 1.0)))

        xlabel_y = _scale_about_anchor(cfg.footer_xlabel_offset, 0.0)
        footer_arrow_y = _scale_about_anchor(cfg.footer_arrow_offset, 0.0)
        footer_text_y = _scale_about_anchor(cfg.footer_text_offset, 0.0)

        auto_title_y = min(0.995, top_margin + (title_gap_in / max(fig_height_in, 1.0)))
        title_y = (
            _scale_about_anchor(cfg.title_y_position, 1.0)
            if cfg.title_y_position is not None
            else auto_title_y
        )
        title_fontsize = cfg.title_fontsize or (12.5 if title_line_count >= 3 else 14)
        fig.subplots_adjust(left=0.34, right=0.72, top=top_margin, bottom=bottom_margin)

        if title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight="bold", y=title_y)

        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

        # Use config positions (or defaults)
        label_x = cfg.label_x_position
        reference_x = cfg.reference_x_position
        comparator_x = cfg.comparator_x_position
        hr_x = cfg.hr_x_position
        median_ref_x = cfg.median_ref_x_position
        median_cmp_x = cfg.median_cmp_x_position
        pvalue_x = cfg.pvalue_x_position

        header_top_y = _scale_about_anchor(cfg.header_top_y, 1.0)
        header_sub_y = _scale_about_anchor(cfg.header_sub_y, 1.0)
        header_rule_y = header_sub_y - max(0.012, 0.03 * vertical_scale)

        ref_cmp_fontsize = self._compute_events_fontsize(
            fig, ax, prepared_df, reference_x, comparator_x
        )

        ax.set_xlabel("")

        self._draw_headers(
            ax,
            label_x,
            reference_x,
            comparator_x,
            hr_x,
            median_ref_x,
            median_cmp_x,
            pvalue_x,
            header_top_y,
            header_sub_y,
            header_rule_y,
        )

        self._draw_row_data(
            ax,
            prepared_df,
            y_positions,
            label_x,
            reference_x,
            comparator_x,
            hr_x,
            median_ref_x,
            median_cmp_x,
            pvalue_x,
            ref_cmp_fontsize,
            n_rows,
        )

        self._draw_footer(ax, xlabel_y, footer_arrow_y, footer_text_y)

    def _draw_headers(
        self,
        ax: Axes,
        label_x: float,
        reference_x: float,
        comparator_x: float,
        hr_x: float,
        median_ref_x: float,
        median_cmp_x: float,
        pvalue_x: float,
        header_top_y: float,
        header_sub_y: float,
        header_rule_y: float,
    ) -> None:
        """Draw table column headers."""
        cfg = self.config

        if cfg.show_events_patients:
            ax.text(
                (reference_x + comparator_x) / 2,
                header_top_y,
                "Events/Patients",
                transform=ax.transAxes,
                fontsize=cfg.header_fontsize,
                fontweight="bold",
                ha="center",
                va="bottom",
                clip_on=False,
            )
            ax.text(
                reference_x,
                header_sub_y,
                "Reference",
                transform=ax.transAxes,
                fontsize=cfg.label_fontsize,
                fontweight="bold",
                ha="center",
                va="bottom",
                clip_on=False,
            )
            ax.text(
                comparator_x,
                header_sub_y,
                "Comparator",
                transform=ax.transAxes,
                fontsize=cfg.label_fontsize,
                fontweight="bold",
                ha="center",
                va="bottom",
                clip_on=False,
            )

        if cfg.show_hr_column:
            ax.text(
                hr_x,
                header_top_y,
                "HR (95% CI)",
                transform=ax.transAxes,
                fontsize=cfg.header_fontsize,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

        if cfg.show_median_columns:
            ax.text(
                median_ref_x,
                header_top_y,
                "Median Ref\n(95% CI)",
                transform=ax.transAxes,
                fontsize=cfg.header_fontsize,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )
            ax.text(
                median_cmp_x,
                header_top_y,
                "Median Cmp\n(95% CI)",
                transform=ax.transAxes,
                fontsize=cfg.header_fontsize,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

        if cfg.show_pvalue_column:
            ax.text(
                pvalue_x,
                header_top_y,
                r"$\mathit{\mathbf{p}}$-value",
                transform=ax.transAxes,
                fontsize=cfg.header_fontsize,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

        ax.plot(
            [label_x, cfg.header_rule_end_x],
            [header_rule_y, header_rule_y],
            transform=ax.transAxes,
            color="black",
            linewidth=0.8,
            clip_on=False,
        )

    def _draw_row_data(
        self,
        ax: Axes,
        df: pd.DataFrame,
        y_positions: np.ndarray,
        label_x: float,
        reference_x: float,
        comparator_x: float,
        hr_x: float,
        median_ref_x: float,
        median_cmp_x: float,
        pvalue_x: float,
        ref_cmp_fontsize: float,
        n_rows: int,
    ) -> None:
        """Draw row labels and data cells."""
        cfg = self.config

        for i, (_, row) in enumerate(df.iterrows()):
            row_y = self._data_to_axes_y(ax, y_positions[i])

            ax.text(
                label_x,
                row_y,
                str(row.get("display_label", "")),
                transform=ax.transAxes,
                fontsize=cfg.label_fontsize,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )

            if cfg.show_events_patients:
                ax.text(
                    reference_x,
                    row_y,
                    self._format_reference_cell(row),
                    transform=ax.transAxes,
                    fontsize=ref_cmp_fontsize,
                    ha="center",
                    va="center",
                    linespacing=1.15,
                    clip_on=False,
                )
                ax.text(
                    comparator_x,
                    row_y,
                    self._format_comparator_cell(row),
                    transform=ax.transAxes,
                    fontsize=ref_cmp_fontsize,
                    ha="center",
                    va="center",
                    linespacing=1.15,
                    clip_on=False,
                )

            if cfg.show_hr_column:
                ax.text(
                    hr_x,
                    row_y,
                    self._format_hr_ci(row),
                    transform=ax.transAxes,
                    fontsize=cfg.cell_fontsize,
                    ha="left",
                    va="center",
                    clip_on=False,
                )

            if cfg.show_median_columns:
                ax.text(
                    median_ref_x,
                    row_y,
                    self._format_reference_median(row),
                    transform=ax.transAxes,
                    fontsize=cfg.cell_fontsize,
                    ha="left",
                    va="center",
                    clip_on=False,
                )
                ax.text(
                    median_cmp_x,
                    row_y,
                    self._format_comparator_median(row),
                    transform=ax.transAxes,
                    fontsize=cfg.cell_fontsize,
                    ha="left",
                    va="center",
                    clip_on=False,
                )

            if cfg.show_pvalue_column:
                ax.text(
                    pvalue_x,
                    row_y,
                    self._format_pvalue(row),
                    transform=ax.transAxes,
                    fontsize=cfg.cell_fontsize,
                    ha="left",
                    va="center",
                    clip_on=False,
                )

            self._draw_truncated_markers(ax, row, y_positions[i], n_rows)

    def _draw_truncated_markers(self, ax: Axes, row: pd.Series, y_pos: float, n_rows: int) -> None:
        """Draw truncation markers for clipped CI bars."""
        cfg = self.config

        if not cfg.show_truncation_markers:
            return

        hr_col = cfg.hr_col
        ci_lower_col = cfg.ci_lower_col
        ci_upper_col = cfg.ci_upper_col

        x_min, x_max = ax.get_xlim()
        marker_x = float(row[hr_col])

        if marker_x > x_max:
            ax.scatter(x_max, y_pos, marker="o", s=28, color=cfg.marker_color, zorder=4)
        elif marker_x < x_min:
            ax.scatter(x_min, y_pos, marker="o", s=28, color=cfg.marker_color, zorder=4)

        if n_rows <= 1:
            truncation_scale = 0.72
        elif n_rows <= 4:
            truncation_scale = 0.88
        else:
            truncation_scale = 1.0

        ci_hi = float(row[ci_upper_col])
        if ci_hi > x_max:
            self._draw_truncation_cap(
                ax, x_max, y_pos, direction="right", size_scale=truncation_scale
            )

        ci_lo = float(row[ci_lower_col])
        if ci_lo < x_min:
            self._draw_truncation_cap(
                ax, x_min, y_pos, direction="left", size_scale=truncation_scale
            )

    def _draw_truncation_cap(
        self,
        ax: Axes,
        x_value: float,
        y_value: float,
        *,
        direction: str,
        size_scale: float = 1.0,
    ) -> None:
        """Draw a display-fixed truncation cap marker."""
        figure = ax.figure

        def _offset_point(dx_pt: float, dy_pt: float) -> tuple[float, float]:
            base_x_px, base_y_px = ax.transData.transform((x_value, y_value))
            dx_px = dx_pt * figure.dpi / 72.0
            dy_px = dy_pt * figure.dpi / 72.0
            return tuple(ax.transData.inverted().transform((base_x_px + dx_px, base_y_px + dy_px)))

        cap_half_height_pt = 8.0 * size_scale
        slash_dx_pt = 7.0 * size_scale
        slash_dy_pt = 5.0 * size_scale
        slash_gap_pt = 4.0 * size_scale

        cap_bottom = _offset_point(0.0, -cap_half_height_pt)
        cap_top = _offset_point(0.0, cap_half_height_pt)
        ax.plot(
            [cap_bottom[0], cap_top[0]],
            [cap_bottom[1], cap_top[1]],
            color="black",
            linewidth=2.0,
            zorder=4,
            solid_capstyle="butt",
            clip_on=True,
        )

        if direction == "right":
            slash_offsets = [
                ((-slash_dx_pt, -slash_dy_pt), (0.0, slash_dy_pt)),
                ((-slash_dx_pt - slash_gap_pt, -slash_dy_pt), (-slash_gap_pt, slash_dy_pt)),
            ]
        else:
            slash_offsets = [
                ((0.0, -slash_dy_pt), (slash_dx_pt, slash_dy_pt)),
                ((slash_gap_pt, -slash_dy_pt), (slash_dx_pt + slash_gap_pt, slash_dy_pt)),
            ]

        for start_offset, end_offset in slash_offsets:
            start = _offset_point(*start_offset)
            end = _offset_point(*end_offset)
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black",
                linewidth=1.4,
                zorder=4,
                solid_capstyle="butt",
                clip_on=True,
            )

    def _draw_footer(
        self, ax: Axes, xlabel_y: float, footer_arrow_y: float, footer_text_y: float
    ) -> None:
        """Draw footer with directional arrows and labels."""
        cfg = self.config

        x_min, x_max = ax.get_xlim()
        null_x = 1.0
        xaxis_transform = ax.get_xaxis_transform()

        left_arrow_end = x_min + 0.06 * max(null_x - x_min, 0.0)
        right_arrow_end = x_max - 0.06 * max(x_max - null_x, 0.0)

        ax.text(
            null_x,
            xlabel_y,
            cfg.xlabel,
            transform=xaxis_transform,
            fontsize=cfg.xlabel_fontsize,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=False,
        )

        ax.annotate(
            "",
            xy=(left_arrow_end, footer_arrow_y),
            xytext=(null_x, footer_arrow_y),
            xycoords=xaxis_transform,
            textcoords=xaxis_transform,
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": "black"},
            annotation_clip=False,
        )
        ax.annotate(
            "",
            xy=(right_arrow_end, footer_arrow_y),
            xytext=(null_x, footer_arrow_y),
            xycoords=xaxis_transform,
            textcoords=xaxis_transform,
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.9, "color": "black"},
            annotation_clip=False,
        )
        ax.annotate(
            cfg.left_footer_label,
            xy=(null_x, footer_text_y),
            xycoords=xaxis_transform,
            xytext=(-6, 0),
            textcoords="offset points",
            fontsize=cfg.footer_fontsize,
            ha="right",
            va="top",
            annotation_clip=False,
        )
        ax.annotate(
            cfg.right_footer_label,
            xy=(null_x, footer_text_y),
            xycoords=xaxis_transform,
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=cfg.footer_fontsize,
            ha="left",
            va="top",
            annotation_clip=False,
        )

    # ==========================================================================
    # Formatting helpers
    # ==========================================================================

    def _format_reference_cell(self, row: pd.Series) -> str:
        """Format reference arm cell (label + events/patients)."""
        cfg = self.config
        label = self._canonicalize_arm_label(row.get(cfg.reference_label_col or "", ""))
        counts = self._format_events_patients(
            row.get(cfg.reference_events_col or ""),
            row.get(cfg.reference_n_col or ""),
        )
        return "\n".join(filter(None, [label, counts]))

    def _format_comparator_cell(self, row: pd.Series) -> str:
        """Format comparator arm cell (label + events/patients)."""
        cfg = self.config
        label = self._canonicalize_arm_label(row.get(cfg.comparator_label_col or "", ""))
        counts = self._format_events_patients(
            row.get(cfg.comparator_events_col or ""),
            row.get(cfg.comparator_n_col or ""),
        )
        return "\n".join(filter(None, [label, counts]))

    def _format_reference_median(self, row: pd.Series) -> str:
        """Format reference median survival with CI."""
        cfg = self.config
        return self._format_median_ci(
            row.get(cfg.median_ref_col or ""),
            row.get(cfg.median_ref_ci_lower_col or ""),
            row.get(cfg.median_ref_ci_upper_col or ""),
            row.get(cfg.median_ref_not_reached_col or "", False),
            row.get(cfg.median_ref_ci_lower_not_reached_col or "", False),
            row.get(cfg.median_ref_ci_upper_not_reached_col or "", False),
        )

    def _format_comparator_median(self, row: pd.Series) -> str:
        """Format comparator median survival with CI."""
        cfg = self.config
        return self._format_median_ci(
            row.get(cfg.median_cmp_col or ""),
            row.get(cfg.median_cmp_ci_lower_col or ""),
            row.get(cfg.median_cmp_ci_upper_col or ""),
            row.get(cfg.median_cmp_not_reached_col or "", False),
            row.get(cfg.median_cmp_ci_lower_not_reached_col or "", False),
            row.get(cfg.median_cmp_ci_upper_not_reached_col or "", False),
        )

    def _format_hr_ci(self, row: pd.Series) -> str:
        """Format HR with 95% CI."""
        cfg = self.config
        hr = row.get(cfg.hr_col)
        ci_lo = row.get(cfg.ci_lower_col)
        ci_hi = row.get(cfg.ci_upper_col)
        if pd.isna(hr) or pd.isna(ci_lo) or pd.isna(ci_hi):
            return ""
        return f"{hr:.2f} ({ci_lo:.2f} - {ci_hi:.2f})"

    def _format_pvalue(self, row: pd.Series) -> str:
        """Format p-value with significance threshold."""
        cfg = self.config
        p_value = row.get(cfg.pvalue_col or "")
        if pd.isna(p_value):
            return "N/A"
        if p_value < 0.001:
            return "<0.001"
        return f"{p_value:.4f}"

    @staticmethod
    def _format_events_patients(events: Any, patients: Any) -> str:
        """Format events/patients as 'n/N' string."""
        if pd.isna(events) or pd.isna(patients):
            return ""
        return f"{int(events)}/{int(patients)}"

    @staticmethod
    def _format_median_ci(
        median: Any,
        ci_lower: Any,
        ci_upper: Any,
        median_not_reached: Any = False,
        ci_lower_not_reached: Any = False,
        ci_upper_not_reached: Any = False,
    ) -> str:
        """Format median survival with CI, handling NR values."""
        if bool(median_not_reached):
            return "NR"
        if pd.isna(median):
            return ""

        lower_text = (
            "NR" if bool(ci_lower_not_reached) else ("" if pd.isna(ci_lower) else f"{ci_lower:.2f}")
        )
        upper_text = (
            "NR" if bool(ci_upper_not_reached) else ("" if pd.isna(ci_upper) else f"{ci_upper:.2f}")
        )

        if lower_text and upper_text:
            return f"{median:.2f} ({lower_text} - {upper_text})"
        return f"{median:.2f}"

    @staticmethod
    def _canonicalize_arm_label(label: str) -> str:
        """Normalize arm labels (e.g., 'Chemo' -> 'Chemotherapy').

        Only normalizes chemotherapy variants. Other labels like 'Control'
        and 'Placebo' are preserved as-is.
        """
        text = str(label).strip()
        upper = text.upper()
        if upper in {"CHEMOTHERAPY", "CHEMO"}:
            return "Chemotherapy"
        return text

    # ==========================================================================
    # Layout measurement helpers
    # ==========================================================================

    @staticmethod
    def _wrap_title(title: str, *, width: int = 52) -> str:
        """Wrap title text to specified character width."""
        lines = [line.strip() for line in str(title).splitlines()]
        wrapped_lines: list[str] = []
        for index, line in enumerate(lines):
            if not line:
                continue
            line = line.replace("_", " ") if index > 0 else line
            wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=False) or [line])
        return "\n".join(wrapped_lines)

    @staticmethod
    def _text_block_line_count(value: str) -> int:
        """Count lines in a text block."""
        stripped = str(value).strip()
        if not stripped:
            return 1
        return max(len(stripped.splitlines()), 1)

    def _compute_row_units(self, df: pd.DataFrame) -> list[float]:
        """Compute row height units based on multiline cells."""
        units: list[float] = []
        for _, row in df.iterrows():
            units.append(
                float(
                    max(
                        self._text_block_line_count(row.get("display_label", "")),
                        self._text_block_line_count(self._format_reference_cell(row)),
                        self._text_block_line_count(self._format_comparator_cell(row)),
                        1,
                    )
                )
            )
        return units or [1.0]

    def _compute_events_fontsize(
        self,
        fig: Figure,
        ax: Axes,
        df: pd.DataFrame,
        reference_x: float,
        comparator_x: float,
    ) -> float:
        """Compute adaptive font size for events/patients cells."""
        axes_width_inches = fig.get_figwidth() * ax.get_position().width
        available_width_inches = max((comparator_x - reference_x - 0.035) * axes_width_inches, 0.25)

        ref_width = max(
            self._estimate_text_width(self._format_reference_cell(row), 9.0)
            for _, row in df.iterrows()
        )
        cmp_width = max(
            self._estimate_text_width(self._format_comparator_cell(row), 9.0)
            for _, row in df.iterrows()
        )
        combined_width = ref_width + cmp_width

        if combined_width <= available_width_inches:
            return 9.0
        scale = max(0.78, available_width_inches / max(combined_width, 1e-6))
        return round(9.0 * scale, 2)

    @staticmethod
    def _estimate_text_width(text: str, fontsize: float) -> float:
        """Estimate rendered text width in inches."""
        stripped = str(text).strip()
        if not stripped:
            return 0.0
        font_props = FontProperties(size=fontsize)
        max_width = 0.0
        for segment in stripped.splitlines() or [stripped]:
            candidate = segment if segment else " "
            try:
                text_path = TextPath((0, 0), candidate, size=fontsize, prop=font_props)
                max_width = max(max_width, text_path.get_extents().width / 72.0)
            except Exception:
                max_width = max(max_width, (len(candidate) * fontsize * 0.55) / 72.0)
        return max_width

    @staticmethod
    def _data_to_axes_y(ax: Axes, y_value: float) -> float:
        """Convert data y-coordinate to axes fraction."""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if y_range == 0:
            return 0.5
        return (y_value - y_min) / y_range


__all__ = ["ClinicalForestPlotter"]
