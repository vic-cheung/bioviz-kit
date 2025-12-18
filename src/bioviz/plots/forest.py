"""
Forest plot functions ported to bioviz (pure plotting: df + kwargs -> fig, ax).

This module contains plotting-only code. It intentionally avoids external
filesystem helpers and uses pathlib directly when saving output.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_forest_plot(
    hr_data: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (10, 8),
    hr_col: str = "hr",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    label_col: str = "comparator",
    pvalue_col: str = "p_value",
    reference_col: Optional[str] = "reference",
    variable_col: Optional[str] = "variable",
    section_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Hazard Ratio (95% CI)",
    show_reference_line: bool = True,
    reference_line_color: str = "#D32F2F",
    reference_line_style: str = "--",
    reference_line_width: float = 1.5,
    show_stats_table: bool = True,
    log_scale: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    xticks: Optional[list] = None,
    color_significant: str = "#2E7D32",
    color_nonsignificant: str = "#757575",
    marker_color_significant: Optional[str] = None,
    marker_color_nonsignificant: Optional[str] = None,
    alpha_threshold: float = 0.05,
    marker_size: float = 8,
    marker_style: str = "s",
    linewidth: float = 2,
    show_caps: bool = False,
    capsize: float = 2,
    show_section_separators: bool = True,
    section_separator_color: str = "blue",
    section_separator_alpha: float = 0.25,
    section_gap: float = 0.0,
    y_margin: float = 0.5,
    section_label_x_position: float = -0.35,
    stats_table_x_position: float = 1.05,
    stats_table_col_spacing: float = 0.15,
    show_grid: bool = False,
    center_around_null: bool = False,
    show_y_spine: bool = False,
    show_yticks: bool = False,
    ytick_fontsize: float = 10,
    xtick_fontsize: float = 10,
    xlabel_fontsize: float = 11,
    stats_fontsize: float = 9,
    title_fontsize: float = 12,
    category_order: Optional[Dict[str, list]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # The implementation mirrors the original tm_modeling behaviour but is
    # self-contained and focuses only on plotting.
    required_cols = [hr_col, ci_lower_col, ci_upper_col, label_col]
    missing = [col for col in required_cols if col not in hr_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = hr_data.copy()
    df = df.dropna(subset=[hr_col, ci_lower_col, ci_upper_col])

    if df.empty:
        raise ValueError("No valid data rows after removing NaN values")

    # Category ordering (kept verbatim to preserve original behaviour)
    if category_order and variable_col and variable_col in df.columns:
        ordered_dfs = []
        ordered_variables = list(category_order.keys())[::-1]
        remaining_variables = [
            v for v in df[variable_col].unique() if v not in category_order.keys()
        ]
        all_variables = ordered_variables + remaining_variables

        for variable in all_variables:
            var_df = df[df[variable_col] == variable].copy()
            if variable in category_order:
                order_list = category_order[variable]
                var_df[label_col] = pd.Categorical(
                    var_df[label_col], categories=order_list, ordered=True
                )
                var_df = var_df.sort_values(label_col)
                var_df[label_col] = var_df[label_col].astype(str)
            ordered_dfs.append(var_df)

        df = pd.concat(ordered_dfs, ignore_index=True)
    elif category_order and label_col in df.columns:
        if category_order:
            order_list = next(iter(category_order.values()))
            df[label_col] = pd.Categorical(df[label_col], categories=order_list, ordered=True)
            df = df.sort_values(label_col)
            df[label_col] = df[label_col].astype(str)

    # Reverse order so top-to-bottom reads naturally
    if variable_col and variable_col in df.columns:
        df = (
            df.groupby(variable_col, sort=False, group_keys=False)
            .apply(lambda group: group.iloc[::-1])
            .reset_index(drop=True)
        )
    else:
        df = df.iloc[::-1].reset_index(drop=True)

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(n_rows, dtype=float)

    if variable_col and variable_col in df.columns and section_gap != 0.0:
        current_variable = None
        cumulative_gap = 0.0
        for i, (idx, row) in enumerate(df.iterrows()):
            variable = row[variable_col]
            if variable != current_variable and current_variable is not None:
                cumulative_gap += section_gap
            y_positions[i] += cumulative_gap
            current_variable = variable

    # Colors based on p-value significance
    colors = []
    if pvalue_col in df.columns:
        for _, row in df.iterrows():
            pval = row[pvalue_col]
            if pd.notna(pval) and pval < alpha_threshold:
                colors.append(color_significant)
            else:
                colors.append(color_nonsignificant)
    else:
        colors = [color_nonsignificant] * n_rows

    if marker_color_significant is None:
        marker_color_significant = color_significant
    if marker_color_nonsignificant is None:
        marker_color_nonsignificant = color_nonsignificant

    marker_colors = []
    if pvalue_col in df.columns:
        for idx, row in df.iterrows():
            pval = row[pvalue_col]
            if pd.notna(pval) and pval < alpha_threshold:
                marker_colors.append(marker_color_significant)
            else:
                marker_colors.append(marker_color_nonsignificant)
    else:
        marker_colors = [marker_color_nonsignificant] * n_rows

    # Plot CI bars and points
    for i, (idx, row) in enumerate(df.iterrows()):
        hr = row[hr_col]
        ci_lower = row[ci_lower_col]
        ci_upper = row[ci_upper_col]
        ci_color = colors[i]
        marker_color = marker_colors[i]

        ax.plot(
            [ci_lower, ci_upper],
            [y_positions[i], y_positions[i]],
            color=ci_color,
            linewidth=linewidth,
            solid_capstyle="round",
        )

        if show_caps:
            cap_height = capsize * 0.01
            ax.plot(
                [ci_lower, ci_lower],
                [y_positions[i] - cap_height, y_positions[i] + cap_height],
                color=ci_color,
                linewidth=linewidth,
            )
            ax.plot(
                [ci_upper, ci_upper],
                [y_positions[i] - cap_height, y_positions[i] + cap_height],
                color=ci_color,
                linewidth=linewidth,
            )

        ax.scatter(
            hr,
            y_positions[i],
            s=marker_size**2,
            color=marker_color,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            marker=marker_style,
        )

    if show_reference_line:
        ax.axvline(
            x=1,
            color=reference_line_color,
            linestyle=reference_line_style,
            linewidth=reference_line_width,
            alpha=0.7,
            zorder=1,
        )

    if log_scale:
        ax.set_xscale("log")

    # X limits computation (kept from original)
    if xlim is not None:
        if center_around_null and log_scale:
            xlim_min, xlim_max = xlim
            log_min = np.log10(xlim_min)
            log_max = np.log10(xlim_max)
            max_distance = max(abs(log_min), abs(log_max))
            xlim_min = 10 ** (-max_distance)
            xlim_max = 10**max_distance
            ax.set_xlim(xlim_min, xlim_max)
        else:
            ax.set_xlim(xlim)
    else:
        all_values = pd.concat([df[ci_lower_col], df[ci_upper_col]])
        all_values = all_values.replace([np.inf, -np.inf], np.nan)
        all_values = all_values[all_values > 0]
        all_values = all_values.dropna()

        if len(all_values) == 0:
            xlim_min, xlim_max = 0.1, 10
        else:
            min_val = all_values.min()
            max_val = all_values.max()
            if log_scale:
                if center_around_null:
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    max_distance = max(abs(log_min), abs(log_max))
                    padding = 0.2
                    max_distance_padded = max_distance * (1 + padding)
                    xlim_min = 10 ** (-max_distance_padded)
                    xlim_max = 10**max_distance_padded
                else:
                    padding = 0.2
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    log_range = log_max - log_min
                    xlim_min = 10 ** (log_min - padding * log_range)
                    xlim_max = 10 ** (log_max + padding * log_range)
            else:
                range_val = max_val - min_val
                xlim_min = max(0, min_val - 0.1 * range_val)
                xlim_max = max_val + 0.1 * range_val

        ax.set_xlim(xlim_min, xlim_max)

    if xticks is not None:
        ax.set_xticks(xticks)
        if log_scale:
            ax.set_xticklabels([f"{x:.2g}" for x in xticks], fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels([f"{x:.2f}" for x in xticks], fontsize=xtick_fontsize)
    else:
        ax.tick_params(axis="x", labelsize=xtick_fontsize)

    ax.set_yticks(y_positions)
    labels = df[label_col].astype(str).tolist()
    if reference_col and reference_col in df.columns:
        labels = [
            f"{label} (vs {row[reference_col]})" if pd.notna(row[reference_col]) else label
            for label, (_, row) in zip(labels, df.iterrows())
        ]

    ax.set_yticklabels(labels, fontsize=ytick_fontsize)
    if not show_yticks:
        ax.tick_params(axis="y", length=0)

    if len(y_positions) > 0:
        y_min = y_positions.min() - y_margin
        y_max = y_positions.max() + y_margin
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight="bold")
    ax.set_ylabel("")

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=20)

    if show_grid:
        ax.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not show_y_spine:
        ax.spines["left"].set_visible(False)

    if show_stats_table and pvalue_col in df.columns:
        table_data = []
        for _, row in df.iterrows():
            hr = row[hr_col]
            ci_lower = row[ci_lower_col]
            ci_upper = row[ci_upper_col]
            pval = row[pvalue_col]

            hr_str = f"{hr:.2f}"
            ci_str = f"({ci_lower:.2f}, {ci_upper:.2f})"
            pval_str = f"{pval:.4f}" if pd.notna(pval) else "N/A"
            if pd.notna(pval) and pval < 0.001:
                pval_str = "<0.001"

            table_data.append([hr_str, ci_str, pval_str])

        table_x = stats_table_x_position
        col_spacing = max(stats_table_col_spacing, stats_fontsize * 0.015)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        def data_to_axes_y(y_data):
            return (y_data - y_min) / y_range

        header_y_data = n_rows - 1 + 0.8
        header_y_axes = data_to_axes_y(header_y_data)

        ax.text(
            table_x,
            header_y_axes,
            "HR",
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            table_x + col_spacing,
            header_y_axes,
            "95% CI",
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            table_x + col_spacing * 2,
            header_y_axes,
            "p-value",
            transform=ax.transAxes,
            fontsize=stats_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
        )

        for i, row_data in enumerate(table_data):
            row_y_axes = data_to_axes_y(y_positions[i])
            ax.text(
                table_x,
                row_y_axes,
                row_data[0],
                transform=ax.transAxes,
                fontsize=stats_fontsize,
                ha="center",
                va="center",
            )
            ax.text(
                table_x + col_spacing,
                row_y_axes,
                row_data[1],
                transform=ax.transAxes,
                fontsize=stats_fontsize,
                ha="center",
                va="center",
            )
            ax.text(
                table_x + col_spacing * 2,
                row_y_axes,
                row_data[2],
                transform=ax.transAxes,
                fontsize=stats_fontsize,
                ha="center",
                va="center",
            )

    if variable_col and variable_col in df.columns:
        unique_variables = df[variable_col].unique()
        if len(unique_variables) > 1:
            variable_ranges = {}
            for i, (idx, row) in enumerate(df.iterrows()):
                variable = row[variable_col]
                if variable not in variable_ranges:
                    variable_ranges[variable] = {"indices": []}
                variable_ranges[variable]["indices"].append(i)

            section_boundaries = []
            for variable, data in variable_ranges.items():
                indices = data["indices"]
                min_idx = min(indices, key=lambda idx: y_positions[idx])
                max_idx = max(indices, key=lambda idx: y_positions[idx])
                min_y = y_positions[min_idx]
                max_y = y_positions[max_idx]

                variable_ranges[variable]["min_y"] = min_y
                variable_ranges[variable]["max_y"] = max_y
                variable_ranges[variable]["label_y"] = max_y
                section_boundaries.append((min_y, variable))

            section_boundaries.sort(key=lambda x: x[0])

            if show_section_separators:
                for i in range(1, len(section_boundaries)):
                    current_min = section_boundaries[i][0]
                    prev_var = section_boundaries[i - 1][1]
                    prev_max = variable_ranges[prev_var]["max_y"]
                    separator_y = (current_min + prev_max) / 2
                    ax.axhline(
                        y=separator_y,
                        color=section_separator_color,
                        linestyle="-",
                        linewidth=1,
                        alpha=section_separator_alpha,
                    )

            for variable, data in variable_ranges.items():
                if section_labels and variable in section_labels:
                    section_label = section_labels[variable]
                else:
                    section_label = str(variable)

                ax.text(
                    section_label_x_position,
                    data["label_y"],
                    section_label,
                    transform=ax.get_yaxis_transform(),
                    fontsize=ytick_fontsize + 1,
                    fontweight="bold",
                    va="center",
                    ha="right",
                )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        print(f"Forest plot saved to: {output_file}")

    return fig, ax


def generate_multi_variable_forest_plot(
    hr_data_dict: Dict[str, pd.DataFrame],
    *,
    figsize: Tuple[float, float] = (12, 10),
    hr_col: str = "hr",
    ci_lower_col: str = "ci_lower",
    ci_upper_col: str = "ci_upper",
    label_col: str = "comparator",
    pvalue_col: str = "p_value",
    reference_col: Optional[str] = "reference",
    title: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    combined_data = []
    for variable_name, df in hr_data_dict.items():
        df_copy = df.copy()
        df_copy["variable"] = variable_name
        combined_data.append(df_copy)

    combined_df = pd.concat(combined_data, ignore_index=True)
    section_labels = {key: key for key in hr_data_dict.keys()}

    return generate_forest_plot(
        combined_df,
        figsize=figsize,
        hr_col=hr_col,
        ci_lower_col=ci_lower_col,
        ci_upper_col=ci_upper_col,
        label_col=label_col,
        pvalue_col=pvalue_col,
        reference_col=reference_col,
        variable_col="variable",
        section_labels=section_labels,
        title=title,
        **kwargs,
    )


class ForestPlotter:
    """Class wrapper around the forest plotting functions for API parity with KMPlotter.

    Example:
        p = ForestPlotter(df)
        fig, ax = p.plot(figsize=(10,8))
    """

    def __init__(self, hr_data: pd.DataFrame):
        self.hr_data = hr_data

    @classmethod
    def from_dict(cls, hr_data_dict: Dict[str, pd.DataFrame]):
        combined = []
        for name, df in hr_data_dict.items():
            df_copy = df.copy()
            df_copy["variable"] = name
            combined.append(df_copy)
        combined_df = pd.concat(combined, ignore_index=True)
        return cls(combined_df)

    def plot(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        return generate_forest_plot(self.hr_data, **kwargs)
