"""Forest plot visualization for hazard ratios.

Provides a plotter class for generating forest plots that display hazard ratios
with confidence intervals from survival analysis (Cox PH models).

Example
-------
>>> from bioviz.configs import ForestPlotConfig
>>> from bioviz.plots import ForestPlotter
>>> cfg = ForestPlotConfig(hr_col="hr", ci_lower_col="ci_lower", ci_upper_col="ci_upper")
>>> plotter = ForestPlotter(hr_df, cfg)
>>> fig, ax = plotter.plot()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..configs.forest_cfg import ForestPlotConfig

__all__ = ["ForestPlotter"]


def _resolve_fontsize(
    config_value: int | None, rcparam_key: str, default: float = 10
) -> float:
    """Return config value if set, else fall back to rcParams or default."""
    if config_value is not None:
        return float(config_value)
    return float(plt.rcParams.get(rcparam_key, default))


class ForestPlotter:
    """Generate forest plots for hazard ratio visualization.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing HR data with columns for HR, CI bounds, labels, p-values.
    config : ForestPlotConfig
        Configuration object specifying plot options.

    Attributes
    ----------
    data : pd.DataFrame
    config : ForestPlotConfig
    """

    def __init__(self, data: pd.DataFrame, config: ForestPlotConfig) -> None:
        self.data = data.copy()
        self.config = config
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate that required columns exist."""
        cfg = self.config
        required = [cfg.hr_col, cfg.ci_lower_col, cfg.ci_upper_col, cfg.label_col]
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare and order data for plotting."""
        cfg = self.config
        df = self.data.dropna(subset=[cfg.hr_col, cfg.ci_lower_col, cfg.ci_upper_col])
        if df.empty:
            raise ValueError("No valid data rows after removing NaN values")

        # Apply category ordering
        if cfg.category_order and cfg.variable_col and cfg.variable_col in df.columns:
            ordered_dfs = []
            ordered_vars = list(cfg.category_order.keys())[::-1]
            remaining = [
                v for v in df[cfg.variable_col].unique() if v not in ordered_vars
            ]
            for var in ordered_vars + remaining:
                var_df = df[df[cfg.variable_col] == var].copy()
                if var in cfg.category_order:
                    order_list = cfg.category_order[var]
                    var_df[cfg.label_col] = pd.Categorical(
                        var_df[cfg.label_col], categories=order_list, ordered=True
                    )
                    var_df = var_df.sort_values(cfg.label_col)
                    var_df[cfg.label_col] = var_df[cfg.label_col].astype(str)
                ordered_dfs.append(var_df)
            df = pd.concat(ordered_dfs, ignore_index=True)
        elif cfg.category_order and cfg.label_col in df.columns:
            order_list = next(iter(cfg.category_order.values()), [])
            if order_list:
                df[cfg.label_col] = pd.Categorical(
                    df[cfg.label_col], categories=order_list, ordered=True
                )
                df = df.sort_values(cfg.label_col)
                df[cfg.label_col] = df[cfg.label_col].astype(str)

        # Reverse for matplotlib (y=0 at bottom)
        if cfg.variable_col and cfg.variable_col in df.columns:
            df = (
                df.groupby(cfg.variable_col, sort=False, group_keys=False)
                .apply(lambda g: g.iloc[::-1])
                .reset_index(drop=True)
            )
        else:
            df = df.iloc[::-1].reset_index(drop=True)

        return df

    def _compute_y_positions(self, df: pd.DataFrame) -> np.ndarray:
        """Compute y-positions with optional section gaps."""
        cfg = self.config
        n_rows = len(df)
        y_positions = np.arange(n_rows, dtype=float)

        if (
            cfg.variable_col
            and cfg.variable_col in df.columns
            and cfg.section_gap != 0.0
        ):
            current_var = None
            cumulative = 0.0
            for i, row in df.iterrows():
                var = row[cfg.variable_col]
                if var != current_var and current_var is not None:
                    cumulative += cfg.section_gap
                y_positions[i] += cumulative
                current_var = var

        return y_positions

    def _get_colors(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Determine CI bar and marker colors based on significance."""
        cfg = self.config
        n = len(df)
        ci_colors = []
        marker_colors = []

        sig_color = cfg.color_significant
        nonsig_color = cfg.color_nonsignificant
        marker_sig = cfg.marker_color_significant or sig_color
        marker_nonsig = cfg.marker_color_nonsignificant or nonsig_color

        if cfg.pvalue_col in df.columns:
            for _, row in df.iterrows():
                pval = row[cfg.pvalue_col]
                if pd.notna(pval) and pval < cfg.alpha_threshold:
                    ci_colors.append(sig_color)
                    marker_colors.append(marker_sig)
                else:
                    ci_colors.append(nonsig_color)
                    marker_colors.append(marker_nonsig)
        else:
            ci_colors = [nonsig_color] * n
            marker_colors = [marker_nonsig] * n

        return ci_colors, marker_colors

    def _set_xlim(self, ax, df: pd.DataFrame) -> None:
        """Set x-axis limits."""
        cfg = self.config
        if cfg.xlim is not None:
            if cfg.center_around_null and cfg.log_scale:
                log_min = np.log10(cfg.xlim[0])
                log_max = np.log10(cfg.xlim[1])
                max_dist = max(abs(log_min), abs(log_max))
                ax.set_xlim(10 ** (-max_dist), 10**max_dist)
            else:
                ax.set_xlim(cfg.xlim)
        else:
            all_vals = pd.concat([df[cfg.ci_lower_col], df[cfg.ci_upper_col]])
            all_vals = all_vals.replace([np.inf, -np.inf], np.nan)
            all_vals = all_vals[all_vals > 0].dropna()
            if len(all_vals) == 0:
                ax.set_xlim(0.1, 10)
            else:
                min_v, max_v = all_vals.min(), all_vals.max()
                if cfg.log_scale:
                    if cfg.center_around_null:
                        max_dist = max(abs(np.log10(min_v)), abs(np.log10(max_v))) * 1.2
                        ax.set_xlim(10 ** (-max_dist), 10**max_dist)
                    else:
                        log_range = np.log10(max_v) - np.log10(min_v)
                        ax.set_xlim(
                            10 ** (np.log10(min_v) - 0.2 * log_range),
                            10 ** (np.log10(max_v) + 0.2 * log_range),
                        )
                else:
                    rng = max_v - min_v
                    ax.set_xlim(max(0, min_v - 0.1 * rng), max_v + 0.1 * rng)

    def _add_stats_table(
        self, ax, df: pd.DataFrame, y_positions: np.ndarray, fontsize: float
    ) -> None:
        """Add HR/CI/p-value table on the right side."""
        cfg = self.config
        if cfg.pvalue_col not in df.columns:
            return

        table_x = cfg.stats_table_x_position
        col_spacing = max(cfg.stats_table_col_spacing, fontsize * 0.015)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        def data_to_axes_y(y_data):
            return (y_data - y_min) / y_range

        # Header
        header_y = data_to_axes_y(len(df) - 1 + 0.8)
        for i, txt in enumerate(["HR", "95% CI", "p-value"]):
            ax.text(
                table_x + i * col_spacing,
                header_y,
                txt,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight="bold",
                ha="center",
                va="center",
            )

        # Data rows
        for i, (_, row) in enumerate(df.iterrows()):
            hr_str = f"{row[cfg.hr_col]:.2f}"
            ci_str = f"({row[cfg.ci_lower_col]:.2f}, {row[cfg.ci_upper_col]:.2f})"
            pval = row[cfg.pvalue_col]
            pval_str = (
                "<0.001"
                if pd.notna(pval) and pval < 0.001
                else (f"{pval:.4f}" if pd.notna(pval) else "N/A")
            )
            row_y = data_to_axes_y(y_positions[i])
            for j, txt in enumerate([hr_str, ci_str, pval_str]):
                ax.text(
                    table_x + j * col_spacing,
                    row_y,
                    txt,
                    transform=ax.transAxes,
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                )

    def _add_section_separators_and_labels(
        self, ax, df: pd.DataFrame, y_positions: np.ndarray, ytick_fs: float
    ) -> None:
        """Add section separators and labels for multi-variable plots."""
        cfg = self.config
        if not cfg.variable_col or cfg.variable_col not in df.columns:
            return
        unique_vars = df[cfg.variable_col].unique()
        if len(unique_vars) <= 1:
            return

        # Build section ranges
        var_ranges: dict[Any, dict[str, Any]] = {}
        for i, (_, row) in enumerate(df.iterrows()):
            var = row[cfg.variable_col]
            if var not in var_ranges:
                var_ranges[var] = {"indices": []}
            var_ranges[var]["indices"].append(i)

        # Compute min/max y for each section
        section_bounds = []
        for var, info in var_ranges.items():
            indices = info["indices"]
            min_y = y_positions[min(indices, key=lambda x: y_positions[x])]
            max_y = y_positions[max(indices, key=lambda x: y_positions[x])]
            var_ranges[var]["min_y"] = min_y
            var_ranges[var]["max_y"] = max_y
            var_ranges[var]["label_y"] = max_y
            section_bounds.append((min_y, var))

        section_bounds.sort(key=lambda x: x[0])

        # Separator lines
        if cfg.show_section_separators:
            for i in range(1, len(section_bounds)):
                cur_min = section_bounds[i][0]
                prev_var = section_bounds[i - 1][1]
                prev_max = var_ranges[prev_var]["max_y"]
                sep_y = (cur_min + prev_max) / 2
                ax.axhline(
                    y=sep_y,
                    color=cfg.section_separator_color,
                    linestyle="-",
                    linewidth=1,
                    alpha=cfg.section_separator_alpha,
                )

        # Section labels
        for var, info in var_ranges.items():
            label = (cfg.section_labels or {}).get(var, str(var))
            ax.text(
                cfg.section_label_x_position,
                info["label_y"],
                label,
                transform=ax.get_yaxis_transform(),
                fontsize=ytick_fs + 1,
                fontweight="bold",
                va="center",
                ha="right",
            )

    def plot(
        self,
        ax=None,
        fig=None,
        output_path: str | Path | None = None,
    ) -> tuple[Any, Any]:
        """Generate the forest plot.

        Parameters
        ----------
        ax : Axes, optional
            Existing axes; if None, a new figure/axes is created.
        fig : Figure, optional
            Existing figure.
        output_path : str or Path, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
        ax : Axes
        """
        cfg = self.config
        df = self._prepare_data()

        # Font sizes
        ytick_fs = _resolve_fontsize(cfg.ytick_fontsize, "ytick.labelsize", 10)
        xtick_fs = _resolve_fontsize(cfg.xtick_fontsize, "xtick.labelsize", 10)
        xlabel_fs = _resolve_fontsize(cfg.xlabel_fontsize, "axes.labelsize", 11)
        title_fs = _resolve_fontsize(cfg.title_fontsize, "axes.titlesize", 12)
        stats_fs = _resolve_fontsize(cfg.stats_fontsize, "font.size", 9)

        # Create figure
        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)

        y_positions = self._compute_y_positions(df)
        ci_colors, marker_colors = self._get_colors(df)

        # Plot error bars and markers
        for i, (_, row) in enumerate(df.iterrows()):
            hr = row[cfg.hr_col]
            ci_lo = row[cfg.ci_lower_col]
            ci_hi = row[cfg.ci_upper_col]
            y = y_positions[i]

            # Error bar
            ax.plot(
                [ci_lo, ci_hi],
                [y, y],
                color=ci_colors[i],
                linewidth=cfg.linewidth,
                solid_capstyle="round",
            )
            # Caps
            if cfg.show_caps:
                cap_h = cfg.capsize * 0.01
                for x in [ci_lo, ci_hi]:
                    ax.plot(
                        [x, x],
                        [y - cap_h, y + cap_h],
                        color=ci_colors[i],
                        linewidth=cfg.linewidth,
                    )
            # Marker
            ax.scatter(
                hr,
                y,
                s=cfg.marker_size**2,
                color=marker_colors[i],
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
                marker=cfg.marker_style,
            )

        # Reference line
        if cfg.show_reference_line:
            ax.axvline(
                x=1,
                color=cfg.reference_line_color,
                linestyle=cfg.reference_line_style,
                linewidth=cfg.reference_line_width,
                alpha=0.7,
                zorder=1,
            )

        # Scale and limits
        if cfg.log_scale:
            ax.set_xscale("log")
        self._set_xlim(ax, df)

        # X-ticks
        if cfg.xticks:
            ax.set_xticks(cfg.xticks)
            fmt = "{:.2g}" if cfg.log_scale else "{:.2f}"
            ax.set_xticklabels([fmt.format(x) for x in cfg.xticks], fontsize=xtick_fs)
        else:
            ax.tick_params(axis="x", labelsize=xtick_fs)

        # Y-axis
        ax.set_yticks(y_positions)
        labels = df[cfg.label_col].astype(str).tolist()
        if cfg.reference_col and cfg.reference_col in df.columns:
            labels = [
                f"{lbl} (vs {row[cfg.reference_col]})"
                if pd.notna(row[cfg.reference_col])
                else lbl
                for lbl, (_, row) in zip(labels, df.iterrows())
            ]
        ax.set_yticklabels(labels, fontsize=ytick_fs)
        if not cfg.show_yticks:
            ax.tick_params(axis="y", length=0)

        # Y-limits
        if len(y_positions) > 0:
            ax.set_ylim(
                y_positions.min() - cfg.y_margin, y_positions.max() + cfg.y_margin
            )

        # Labels/title
        ax.set_xlabel(cfg.xlabel, fontsize=xlabel_fs, fontweight="bold")
        ax.set_ylabel("")
        if cfg.title:
            ax.set_title(cfg.title, fontsize=title_fs, fontweight="bold", pad=20)

        # Grid
        if cfg.show_grid:
            ax.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.8)
            ax.set_axisbelow(True)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not cfg.show_y_spine:
            ax.spines["left"].set_visible(False)

        # Stats table
        if cfg.show_stats_table:
            self._add_stats_table(ax, df, y_positions, stats_fs)

        # Section separators/labels
        self._add_section_separators_and_labels(ax, df, y_positions, ytick_fs)

        # Save
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=300)

        return fig, ax
