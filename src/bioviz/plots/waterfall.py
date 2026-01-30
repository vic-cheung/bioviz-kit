"""Waterfall plot implementation with Config + Plotter pattern.

Provides both a class-based API (WaterfallPlotter) and a legacy function
(plot_waterfall) for backward compatibility.

Supports two main modes:
1. **Standard/Ungrouped**: All samples sorted together by value
2. **Grouped**: Samples sorted within groups, with gaps between groups
"""

from __future__ import annotations

from typing import Optional, Tuple, Callable, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..configs.waterfall_cfg import WaterfallConfig
from .plot_composite_helpers import get_default_palette


class WaterfallPlotter:
    """Class-based waterfall plotter following bioviz-kit conventions.

    Supports two main modes:

    1. **Standard/Ungrouped mode** (default):
       All samples sorted together by value.

       config = WaterfallConfig(
           value_col="VAF_Change",
           color_col="BOR",
           palette=BOR_COLORS,
       )

    2. **Grouped mode** (with sort_within_group_col):
       Samples sorted within each group, with gaps between groups.

       config = WaterfallConfig(
           value_col="VAF_Change",
           color_col="BOR",
           sort_within_group_col="Dose",
           group_order=["100mg", "200mg", "400mg"],
           group_gap=10,
           show_group_counts=True,
       )

    Additional features:
    - `edgecolor_col`: Color bar edges by a different column than fill
    - `bar_annotation_col`: Add per-bar text annotations
    - `threshold_lines`: Draw horizontal reference lines
    - `legend_bbox_to_anchor`: Position legend outside plot
    """

    def __init__(self, df: pd.DataFrame, config: WaterfallConfig):
        """Initialize the WaterfallPlotter.

        Args:
            df: DataFrame containing the data.
            config: WaterfallConfig with plot settings.
        """
        self.df = df.copy()
        self.cfg = config

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Generate the waterfall plot.

        Args:
            ax: Optional matplotlib Axes to plot on. Creates new figure if None.
            show: Whether to call plt.show() after plotting.

        Returns:
            Tuple of (figure, axes).
        """
        cfg = self.cfg

        # Handle aggregation mode
        if cfg.group_col is not None and cfg.aggregate is not None:
            return self._plot_aggregated(ax, show)

        # Handle facet mode
        if cfg.facet_col is not None and cfg.facet_col in self.df.columns:
            return self._plot_faceted(show)

        # Grouped waterfall mode
        if cfg.sort_within_group_col is not None:
            return self._plot_grouped(ax, show)

        # Default per-sample waterfall (ungrouped)
        return self._plot_standard(ax, show)

    def _plot_grouped(
        self,
        ax: Optional[plt.Axes],
        show: bool,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a grouped waterfall with gaps between groups."""
        cfg = self.cfg
        created_fig = False

        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)
            try:
                fig.patch.set_facecolor("white")
                fig.patch.set_alpha(0.0)
            except Exception:
                pass
            created_fig = True
        else:
            fig = ax.figure

        # Determine group order
        if cfg.group_order is not None:
            group_order = cfg.group_order
        else:
            group_order = list(self.df[cfg.sort_within_group_col].unique())

        # Sort within each group
        df_sorted = self.df.copy()
        df_sorted["_sort_key"] = pd.Categorical(
            df_sorted[cfg.sort_within_group_col],
            categories=group_order,
            ordered=True,
        )
        df_sorted = df_sorted.sort_values(
            [cfg.sort_within_group_col, cfg.value_col],
            ascending=[True, not cfg.sort_ascending],  # Descending within groups for waterfall
        ).reset_index(drop=True)

        # Create x-axis positions with gaps between groups
        x_positions = []
        current_pos = 0
        group_centers = []
        group_labels = []
        group_boundaries = []

        for group in group_order:
            group_data = df_sorted[df_sorted[cfg.sort_within_group_col] == group]
            if len(group_data) == 0:
                continue

            # Assign positions for this group
            group_positions = list(range(current_pos, current_pos + len(group_data)))
            x_positions.extend(group_positions)

            # Track group center for label
            group_centers.append(current_pos + len(group_data) / 2)
            if cfg.show_group_counts:
                group_labels.append(f"{group}\n(n={len(group_data)})")
            else:
                group_labels.append(str(group))

            # Move to next position with gap
            current_pos += len(group_data)
            group_boundaries.append(current_pos)
            current_pos += cfg.group_gap

        df_sorted = df_sorted[df_sorted[cfg.sort_within_group_col].isin(group_order)].reset_index(
            drop=True
        )
        df_sorted["_x_position"] = x_positions

        values = df_sorted[cfg.value_col].values

        # Get fill colors
        fill_colors, legend_handles = self._get_colors(df_sorted)

        # Get edge colors
        edge_colors = self._get_edge_colors(df_sorted)

        # Zero line (draw first so bars are on top)
        if cfg.show_zero_line:
            ax.axhline(y=0, color=cfg.zero_line_color, linewidth=cfg.zero_line_width, zorder=1)

        # Threshold lines
        if cfg.threshold_lines:
            for tl in cfg.threshold_lines:
                ax.axhline(
                    y=tl.value,
                    color=tl.color,
                    linestyle=tl.linestyle,
                    linewidth=tl.linewidth,
                    label=tl.label,
                    zorder=1,
                )

        # Plot bars
        if cfg.edgecolor_col is not None:
            # Plot bars individually for different edge colors
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                ax.bar(
                    row["_x_position"],
                    row[cfg.value_col],
                    color=fill_colors[i] if isinstance(fill_colors, list) else fill_colors,
                    edgecolor=edge_colors[i] if isinstance(edge_colors, list) else edge_colors,
                    linewidth=cfg.linewidth,
                    width=cfg.bar_width,
                    zorder=2,
                )
        else:
            ax.bar(
                df_sorted["_x_position"],
                values,
                color=fill_colors,
                edgecolor=edge_colors,
                linewidth=cfg.linewidth,
                width=cfg.bar_width,
                zorder=2,
            )

        # Group separators
        if cfg.show_group_separators:
            for i, boundary in enumerate(group_boundaries[:-1]):
                gap_center = boundary + cfg.group_gap / 2
                ax.axvline(
                    x=gap_center,
                    color=cfg.group_separator_color,
                    linestyle=cfg.group_separator_style,
                    linewidth=cfg.group_separator_width,
                    alpha=cfg.group_separator_alpha,
                )

        # X-axis labels at group centers
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            group_labels,
            rotation=cfg.xtick_rotation,
            ha="right",
            fontsize=cfg.tick_fontsize,
            rotation_mode="anchor",
        )

        # Labels and title
        self._apply_labels_and_title(ax)

        # Axis limits
        if cfg.ylim:
            ax.set_ylim(cfg.ylim)
        if cfg.yticks is not None:
            ax.set_yticks(cfg.yticks)

        # Legend
        self._add_legend(ax, legend_handles)

        # Style spines
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)

        ax.grid(False)

        if created_fig:
            plt.tight_layout()
            if show:
                plt.show()

        return fig, ax

    def _plot_standard(
        self,
        ax: Optional[plt.Axes],
        show: bool,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a standard per-sample waterfall (ungrouped)."""
        cfg = self.cfg
        created_fig = False

        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)
            try:
                fig.patch.set_facecolor("white")
                fig.patch.set_alpha(0.0)
            except Exception:
                pass
            created_fig = True
        else:
            fig = ax.figure

        # Sort data
        df_sorted = self.df.sort_values(by=cfg.value_col, ascending=cfg.sort_ascending).reset_index(
            drop=True
        )
        x = np.arange(len(df_sorted))
        values = df_sorted[cfg.value_col].values

        # Get colors
        fill_colors, legend_handles = self._get_colors(df_sorted)
        edge_colors = self._get_edge_colors(df_sorted)

        # Zero line (draw first)
        if cfg.show_zero_line:
            ax.axhline(y=0, color=cfg.zero_line_color, linewidth=cfg.zero_line_width, zorder=1)

        # Threshold lines
        if cfg.threshold_lines:
            for tl in cfg.threshold_lines:
                ax.axhline(
                    y=tl.value,
                    color=tl.color,
                    linestyle=tl.linestyle,
                    linewidth=tl.linewidth,
                    label=tl.label,
                    zorder=1,
                )

        # Plot bars
        if cfg.edgecolor_col is not None or cfg.bar_annotation_col is not None:
            # Plot bars individually
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                fc = fill_colors[i] if isinstance(fill_colors, list) else fill_colors
                ec = edge_colors[i] if isinstance(edge_colors, list) else edge_colors

                ax.bar(
                    x[i],
                    row[cfg.value_col],
                    color=fc,
                    edgecolor=ec,
                    linewidth=cfg.linewidth,
                    width=cfg.bar_width,
                    zorder=2,
                )

                # Per-bar annotation
                if cfg.bar_annotation_col and cfg.bar_annotation_col in row.index:
                    val = row[cfg.value_col]
                    if val >= 0:
                        text_y = -cfg.bar_annotation_offset
                        va = "top"
                    else:
                        text_y = cfg.bar_annotation_offset
                        va = "bottom"

                    ax.text(
                        x[i],
                        text_y,
                        str(row[cfg.bar_annotation_col]),
                        rotation=cfg.bar_annotation_rotation,
                        ha="center",
                        va=va,
                        fontsize=cfg.bar_annotation_fontsize,
                        alpha=cfg.bar_annotation_alpha,
                        zorder=3,
                    )
        else:
            ax.bar(
                x,
                values,
                color=fill_colors,
                edgecolor=edge_colors,
                linewidth=cfg.linewidth,
                width=cfg.bar_width,
                zorder=2,
            )

        # X-axis ticks
        if cfg.show_xticks and cfg.id_col and cfg.id_col in df_sorted.columns:
            ax.set_xticks(x)
            ax.set_xticklabels(
                df_sorted[cfg.id_col].astype(str),
                rotation=cfg.xtick_rotation,
                fontsize=cfg.tick_fontsize,
            )
        else:
            ax.set_xticks([])

        # Labels and title
        self._apply_labels_and_title(ax)

        # Axis limits
        if cfg.xlim:
            ax.set_xlim(cfg.xlim)
        if cfg.ylim:
            ax.set_ylim(cfg.ylim)
        if cfg.yticks is not None:
            ax.set_yticks(cfg.yticks)

        # Legend
        self._add_legend(ax, legend_handles)

        # Style spines
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.5)

        ax.grid(False)

        if created_fig:
            plt.tight_layout()
            if show:
                plt.show()

        return fig, ax

    def _plot_aggregated(
        self,
        ax: Optional[plt.Axes],
        show: bool,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot aggregated values by group."""
        cfg = self.cfg
        created_fig = False

        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)
            try:
                fig.patch.set_facecolor("white")
                fig.patch.set_alpha(0.0)
            except Exception:
                pass
            created_fig = True
        else:
            fig = ax.figure

        # Aggregate
        agg_df = (
            self.df.groupby(cfg.group_col)[cfg.value_col]
            .agg(cfg.aggregate)
            .reset_index()
            .rename(columns={cfg.value_col: "agg_value"})
        )
        plot_df = agg_df.sort_values(by="agg_value", ascending=cfg.sort_ascending).reset_index(
            drop=True
        )

        x = np.arange(len(plot_df))
        values = plot_df["agg_value"].values

        # Colors
        if cfg.color_col and cfg.color_col in plot_df.columns:
            colors, legend_handles = self._get_colors(plot_df)
        else:
            colors = cfg.default_color
            legend_handles = []

        ax.bar(x, values, color=colors, width=cfg.bar_width)
        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df[cfg.group_col].astype(str),
            rotation=cfg.xtick_rotation,
            fontsize=cfg.tick_fontsize,
        )

        agg_label = cfg.aggregate if isinstance(cfg.aggregate, str) else "agg"
        ylabel = cfg.ylabel or f"{cfg.value_col} ({agg_label})"
        xlabel = cfg.xlabel or cfg.group_col
        ax.set_ylabel(ylabel, fontsize=cfg.ylabel_fontsize)
        ax.set_xlabel(xlabel, fontsize=cfg.xlabel_fontsize)

        if cfg.title:
            ax.set_title(
                cfg.title,
                fontsize=cfg.title_fontsize,
                fontweight=cfg.title_fontweight,
            )

        if created_fig:
            plt.tight_layout()
            if show:
                plt.show()

        return fig, ax

    def _plot_faceted(self, show: bool) -> Tuple[plt.Figure, plt.Axes]:
        """Plot faceted (small multiples) waterfall."""
        cfg = self.cfg
        facet_vals = list(pd.Categorical(self.df[cfg.facet_col]).categories)
        n = len(facet_vals)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=cfg.figsize, squeeze=False)
        try:
            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(0.0)
        except Exception:
            pass
        axes = axes.flatten()

        for i, fv in enumerate(facet_vals):
            sub = self.df[self.df[cfg.facet_col] == fv].sort_values(
                by=cfg.value_col, ascending=cfg.sort_ascending
            )
            x = np.arange(len(sub))
            vals = sub[cfg.value_col].values
            colors, _ = self._get_colors(sub)

            ax_i = axes[i]
            ax_i.bar(x, vals, color=colors, width=cfg.bar_width)
            ax_i.set_title(str(fv))
            ax_i.set_xticks([])

        # Hide unused axes
        for j in range(len(facet_vals), len(axes)):
            axes[j].axis("off")

        if cfg.title:
            fig.suptitle(
                cfg.title,
                fontsize=cfg.title_fontsize,
                fontweight=cfg.title_fontweight,
            )

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axes[0]

    def _get_colors(self, df_plot: pd.DataFrame) -> Tuple[list | str, list[Patch]]:
        """Determine bar fill colors and legend handles.

        Returns:
            Tuple of (colors, legend_handles).
        """
        cfg = self.cfg
        legend_handles = []

        if cfg.color_col and cfg.color_col in df_plot.columns:
            groups = pd.Categorical(df_plot[cfg.color_col])
            categories = list(groups.categories)

            # Determine palette
            if cfg.palette is not None:
                if isinstance(cfg.palette, dict):
                    # Dict mapping category -> color
                    pal = [cfg.palette.get(cat, cfg.default_color) for cat in categories]
                else:
                    # List of colors
                    pal = cfg.palette
            else:
                pal = get_default_palette(len(categories))

            colors = [pal[groups.codes[i] % len(pal)] for i in range(len(groups))]

            # Build legend handles
            for idx, cat in enumerate(categories):
                color = pal[idx % len(pal)]
                legend_handles.append(Patch(facecolor=color, label=str(cat)))

            return colors, legend_handles
        else:
            return cfg.default_color, []

    def _get_edge_colors(self, df_plot: pd.DataFrame) -> list | str:
        """Determine bar edge colors.

        If edgecolor_col is set, colors edges by that column using
        edgecolor_palette. Otherwise, uses the default edgecolor.

        Returns:
            List of colors per bar, or single color string.
        """
        cfg = self.cfg

        if cfg.edgecolor_col is not None and cfg.edgecolor_col in df_plot.columns:
            groups = pd.Categorical(df_plot[cfg.edgecolor_col])
            categories = list(groups.categories)

            if cfg.edgecolor_palette is not None:
                # Dict mapping category -> color
                edge_colors = [
                    cfg.edgecolor_palette.get(cat, cfg.edgecolor)
                    for cat in df_plot[cfg.edgecolor_col]
                ]
            else:
                # Default to a grayscale palette
                pal = get_default_palette(len(categories))
                edge_colors = [pal[groups.codes[i] % len(pal)] for i in range(len(groups))]

            return edge_colors
        else:
            return cfg.edgecolor

    def _apply_labels_and_title(self, ax: plt.Axes) -> None:
        """Apply xlabel, ylabel, and title to axes."""
        cfg = self.cfg

        ylabel = cfg.ylabel if cfg.ylabel is not None else cfg.value_col
        xlabel = cfg.xlabel if cfg.xlabel is not None else (cfg.id_col or "Sample")
        ax.set_ylabel(ylabel, fontsize=cfg.ylabel_fontsize)
        ax.set_xlabel(xlabel, fontsize=cfg.xlabel_fontsize)

        if cfg.title:
            ax.set_title(
                cfg.title,
                fontsize=cfg.title_fontsize,
                fontweight=cfg.title_fontweight,
            )

    def _add_legend(
        self,
        ax: plt.Axes,
        fill_handles: list[Patch],
    ) -> None:
        """Add legend with fill colors (and optionally edge colors)."""
        cfg = self.cfg

        if not cfg.show_legend:
            return

        all_handles = []
        all_labels = []

        # Add fill color handles
        if fill_handles:
            for h in fill_handles:
                all_handles.append(h)
                all_labels.append(h.get_label())

        # Add edge color handles if using edgecolor_col
        if cfg.edgecolor_col is not None and cfg.edgecolor_palette is not None:
            for cat, color in cfg.edgecolor_palette.items():
                patch = Patch(
                    facecolor="white",
                    edgecolor=color,
                    linewidth=cfg.linewidth,
                    label=str(cat),
                )
                all_handles.append(patch)
                all_labels.append(str(cat))

        if not all_handles:
            return

        legend_kwargs = dict(
            handles=all_handles,
            title=cfg.legend_title or cfg.color_col,
            loc=cfg.legend_loc,
            fontsize=cfg.legend_fontsize,
            frameon=cfg.legend_frameon,
        )

        if cfg.legend_title_fontsize is not None:
            legend_kwargs["title_fontsize"] = cfg.legend_title_fontsize

        if cfg.legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = cfg.legend_bbox_to_anchor
            # Adjust loc for outside positioning if bbox is set
            if legend_kwargs["loc"] == "best":
                legend_kwargs["loc"] = "center left"

        ax.legend(**legend_kwargs)


# =============================================================================
# Legacy function interface (backward compatibility)
# =============================================================================


def plot_waterfall(
    df: pd.DataFrame,
    value_col: str,
    id_col: Optional[str] = None,
    color_col: Optional[str] = None,
    group_col: Optional[str] = None,
    aggregate: Optional[Union[Callable, str]] = None,
    facet_by: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    ax=None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a simple waterfall (sorted bar) chart.

    This is the legacy function interface. For more control, use WaterfallPlotter
    with WaterfallConfig.

    Args:
        df: DataFrame with values
        value_col: column name for values (numeric)
        id_col: optional id column to label bars
        color_col: optional column for colors (mapped to categorical palette)
        group_col: optional column for aggregation grouping
        aggregate: aggregation function ('mean', 'sum', etc.) when using group_col
        facet_by: optional column for faceted plots
        figsize: figure size
        ax: optional axes
        show: whether to call plt.show()

    Returns:
        (fig, ax)
    """
    # Build config from legacy args
    config = WaterfallConfig(
        value_col=value_col,
        id_col=id_col,
        color_col=color_col,
        group_col=group_col,
        aggregate=aggregate,
        facet_col=facet_by,
        figsize=figsize,
        show_xticks=id_col is not None,
    )

    plotter = WaterfallPlotter(df, config)
    return plotter.plot(ax=ax, show=show)


def waterfall_with_distribution(
    df: pd.DataFrame,
    value_col: str,
    id_col: Optional[str] = None,
    color_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stat_pairs: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Draw a waterfall on top and a grouped boxplot + annotations below.

    Returns (fig, (ax_waterfall, ax_box))
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
    )
    try:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(0.0)
    except Exception:
        pass
    plot_waterfall(df, value_col, id_col=id_col, color_col=color_col, ax=ax_top, show=False)
    from .grouped import plot_grouped_boxplots

    plot_grouped_boxplots(
        df,
        x=group_col if group_col is not None else color_col,
        y=value_col,
        ax=ax_bot,
        stat_pairs=stat_pairs,
        show=False,
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax_top, ax_bot)
