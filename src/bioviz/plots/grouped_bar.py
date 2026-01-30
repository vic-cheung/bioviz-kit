"""
Grouped bar plots with optional confidence intervals.

Provides bar plots for comparing values across categories with optional grouping
and Clopper-Pearson or bootstrap confidence intervals for proportions.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from scipy.stats import beta
from statsmodels.stats.proportion import proportion_confint

from bioviz.configs.grouped_bar_cfg import GroupedBarConfig

__all__ = [
    "GroupedBarPlotter",
    "clopper_pearson_ci",
    "bootstrap_proportion_ci",
    "compute_proportion_summary",
    "plot_grouped_bars",
]


# =============================================================================
# CI Computation Functions
# =============================================================================


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute Clopper-Pearson (exact) confidence interval for a proportion.

    Recommended for binomial proportions, especially for:
    - Small sample sizes
    - Proportions near 0% or 100%
    - When conservative (wider) CIs are preferred

    Parameters
    ----------
    k : int
        Number of successes (numerator).
    n : int
        Total number of trials (denominator).
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) as proportions (0-1 scale).

    Examples
    --------
    >>> clopper_pearson_ci(15, 100)  # 15% with 95% CI
    (0.0867, 0.2395)
    """
    if n == 0:
        return 0.0, 0.0

    try:
        lo, hi = proportion_confint(count=k, nobs=n, alpha=alpha, method="beta")
        return float(lo), float(hi)
    except Exception:
        pass

    # Fallback to manual beta distribution calculation
    try:
        if k == 0:
            lo = 0.0
        else:
            lo = beta.ppf(alpha / 2, k, n - k + 1)
        if k == n:
            hi = 1.0
        else:
            hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
        return float(lo), float(hi)
    except Exception:
        # Normal approximation fallback (last resort)
        p = k / n
        se = (p * (1 - p) / n) ** 0.5
        lo = max(0.0, p - 1.96 * se)
        hi = min(1.0, p + 1.96 * se)
        return lo, hi


def bootstrap_proportion_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Compute bootstrap percentile confidence interval for a proportion.

    Parameters
    ----------
    k : int
        Number of successes (numerator).
    n : int
        Total number of trials (denominator).
    alpha : float
        Significance level (default 0.05 for 95% CI).
    n_boot : int
        Number of bootstrap samples (default 10000).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) as proportions (0-1 scale).
    """
    if n == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_state)
    arr = np.concatenate([np.ones(k, dtype=int), np.zeros(n - k, dtype=int)])
    boots = rng.choice(arr, size=(n_boot, n), replace=True)
    props = boots.mean(axis=1)
    lower = np.percentile(props, 100 * (alpha / 2))
    upper = np.percentile(props, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_proportion_summary(
    category_list: list[str],
    group_configs: list[dict[str, Any]],
    method: str = "clopper",
    alpha: float = 0.05,
    n_boot: int = 10000,
    random_state: int | None = 12345,
    value_scale: float = 100.0,
) -> pd.DataFrame:
    """
    Compute proportion summary with confidence intervals for multiple groups.

    Parameters
    ----------
    category_list : list[str]
        List of categories to analyze (e.g., genes, pathways).
    group_configs : list[dict]
        Each dict should contain:
            - 'name': str, group name
            - 'k': dict or Series, category -> count mapping
            - 'n': int, total group size
    method : str
        'clopper' for Clopper-Pearson or 'bootstrap' for bootstrap CI.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    n_boot : int
        Number of bootstrap samples (only used if method='bootstrap').
    random_state : int, optional
        Random seed for bootstrap reproducibility.
    value_scale : float
        Scale factor for values (100.0 for percentages, 1.0 for proportions).

    Returns
    -------
    pd.DataFrame
        Summary with columns: Category, Group, k, n, value, ci_low, ci_high

    Examples
    --------
    >>> group_configs = [
    ...     {"name": "Treatment", "k": {"TP53": 15, "KRAS": 30}, "n": 100},
    ...     {"name": "Control", "k": {"TP53": 10, "KRAS": 25}, "n": 80},
    ... ]
    >>> df = compute_proportion_summary(["TP53", "KRAS"], group_configs)
    """
    summary_rows = []

    for category in category_list:
        for group_config in group_configs:
            group_name = group_config["name"]
            k_dict = group_config["k"]
            n_total = group_config["n"]

            # Get count for this category
            if hasattr(k_dict, "get"):
                k = int(k_dict.get(category, 0))
            else:
                # Handle pandas Series
                k = int(k_dict[category]) if category in k_dict.index else 0

            # Compute CI
            if method == "clopper-pearson":
                ci_low, ci_high = clopper_pearson_ci(k, n_total, alpha=alpha)
            elif method == "bootstrap":
                ci_low, ci_high = bootstrap_proportion_ci(
                    k, n_total, alpha=alpha, n_boot=n_boot, random_state=random_state
                )
            else:
                ci_low, ci_high = 0.0, 0.0

            # Compute value (proportion)
            value = (k / n_total) if n_total > 0 else 0.0

            summary_rows.append(
                {
                    "Category": category,
                    "Group": group_name,
                    "k": k,
                    "n": n_total,
                    "value": value_scale * value,
                    "ci_low": value_scale * ci_low,
                    "ci_high": value_scale * ci_high,
                }
            )

    return pd.DataFrame(summary_rows)


# =============================================================================
# Plotting Functions
# =============================================================================


def _resolve_fontsize(config_value: float | None, rcparam_key: str) -> float:
    """
    Resolve fontsize: use config value if set, otherwise fall back to rcParams.

    Parameters
    ----------
    config_value : float | None
        Value from config. If None, use rcParams.
    rcparam_key : str
        Key in matplotlib.rcParams to fall back to.

    Returns
    -------
    float
        The resolved fontsize.
    """
    if config_value is not None:
        return config_value
    return plt.rcParams.get(rcparam_key, 12)


def plot_grouped_bars(
    df: pd.DataFrame,
    config: GroupedBarConfig,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a grouped bar plot from data.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns matching config column names.
    config : GroupedBarConfig
        Configuration object controlling visual aspects.
    ax : plt.Axes, optional
        Axes to draw into. If None, creates new figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        The figure and axes objects.
    """
    # Resolve fontsizes from config or rcParams
    title_fontsize = _resolve_fontsize(config.title_fontsize, "axes.titlesize")
    xlabel_fontsize = _resolve_fontsize(config.xlabel_fontsize, "axes.labelsize")
    ylabel_fontsize = _resolve_fontsize(config.ylabel_fontsize, "axes.labelsize")
    tick_fontsize = _resolve_fontsize(config.tick_fontsize, "xtick.labelsize")
    legend_fontsize = _resolve_fontsize(config.legend_fontsize, "legend.fontsize")
    annot_fontsize = _resolve_fontsize(
        config.annot_fontsize,
        "font.size",
    )
    # Scale annotation fontsize slightly smaller than base if using default
    if config.annot_fontsize is None:
        annot_fontsize = annot_fontsize * 0.9

    # Get categories
    categories = df[config.category_col].unique()
    if config.invert_categories and config.orientation == "horizontal":
        categories = categories[::-1]
    n_categories = len(categories)

    # Determine if grouped or simple bars
    is_grouped = config.group_col is not None and config.group_col in df.columns

    if is_grouped:
        groups = df[config.group_col].unique()
        if config.group_order is not None:
            groups = [g for g in config.group_order if g in groups]
        n_groups = len(groups)
    else:
        groups = [None]
        n_groups = 1

    # Calculate positions and offsets
    positions = np.arange(n_categories) * config.group_spacing

    if n_groups > 1:
        offsets = np.linspace(
            -(n_groups - 1) * config.bar_width / 2,
            (n_groups - 1) * config.bar_width / 2,
            n_groups,
        )
    else:
        offsets = [0]

    # Check if we have CI data
    has_ci = (config.ci_low_col is not None and config.ci_low_col in df.columns) and (
        config.ci_high_col is not None and config.ci_high_col in df.columns
    )

    # Extract data for each group
    group_data = {}
    for group in groups:
        if is_grouped and group is not None:
            group_df = df[df[config.group_col] == group].set_index(config.category_col)
        else:
            group_df = df.set_index(config.category_col)

        # Reindex to match category order
        group_df = group_df.reindex(categories)

        values = group_df[config.value_col].fillna(0).values

        if has_ci:
            ci_low = group_df[config.ci_low_col].fillna(0).values
            ci_high = group_df[config.ci_high_col].fillna(0).values
            err_low = values - ci_low
            err_high = ci_high - values
        else:
            err_low = None
            err_high = None

        group_data[group] = {
            "values": values,
            "err_low": err_low,
            "err_high": err_high,
        }

    # Create figure if needed
    if ax is None:
        if config.figsize is not None:
            figsize = config.figsize
        else:
            if config.orientation == "horizontal":
                figsize = (9, max(6, n_categories * config.group_spacing * 0.4))
            else:
                figsize = (max(8, n_categories * config.group_spacing * 0.5), 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine colors
    if config.group_colors is not None:
        group_colors = config.group_colors
    else:
        default_colors = plt.cm.tab10.colors
        group_colors = {
            g: default_colors[i % len(default_colors)]
            if g is not None
            else config.default_color
            for i, g in enumerate(groups)
        }

    # Plot bars for each group
    for i, group in enumerate(groups):
        data = group_data[group]

        # Get label
        if config.group_labels is not None and group is not None:
            label = config.group_labels.get(group, group)
        elif group is not None:
            label = group
        else:
            label = None

        color = group_colors.get(group, config.default_color)

        # Prepare error bars
        if data["err_low"] is not None and data["err_high"] is not None:
            xerr_or_yerr = [data["err_low"], data["err_high"]]
        else:
            xerr_or_yerr = None

        bar_positions = positions + offsets[i]

        if config.orientation == "horizontal":
            ax.barh(
                bar_positions,
                data["values"],
                height=config.bar_width,
                xerr=xerr_or_yerr,
                label=label,
                color=color,
                edgecolor=config.bar_edgecolor,
                linewidth=config.bar_linewidth,
                ecolor=config.error_color,
                capsize=config.capsize if xerr_or_yerr else 0,
            )
        else:
            ax.bar(
                bar_positions,
                data["values"],
                width=config.bar_width,
                yerr=xerr_or_yerr,
                label=label,
                color=color,
                edgecolor=config.bar_edgecolor,
                linewidth=config.bar_linewidth,
                ecolor=config.error_color,
                capsize=config.capsize if xerr_or_yerr else 0,
            )

    # Configure axes
    if config.orientation == "horizontal":
        ax.set_yticks(positions)
        ax.set_yticklabels(categories, fontsize=tick_fontsize)

        # Y-axis limits
        per_group_half_span = max(abs(o) for o in offsets) + config.bar_width / 2
        margin = 0.15 * config.group_spacing
        y_lower = positions.min() - per_group_half_span - margin
        y_upper = positions.max() + per_group_half_span + margin
        ax.set_ylim(y_lower, y_upper)

        if config.invert_categories:
            ax.invert_yaxis()

        # X-axis limits (value axis) - explicit xlim takes precedence
        if config.xlim is not None:
            ax.set_xlim(config.xlim)
        else:
            if config.value_max is None:
                max_val = 0
                for data in group_data.values():
                    vals = data["values"]
                    errs = (
                        data["err_high"]
                        if data["err_high"] is not None
                        else np.zeros_like(vals)
                    )
                    max_val = max(max_val, (vals + errs).max())
                # Add annotation padding if annotations enabled
                annot_extra = config.annot_padding if config.show_annotations else 0
                padding = max(5, max_val * config.value_padding_pct) + annot_extra
                value_max = max(10, max_val + padding)
            else:
                value_max = config.value_max

            value_min = config.value_min if config.value_min is not None else 0
            ax.set_xlim(value_min, value_max)

        # Custom x-ticks
        if config.xticks is not None:
            ax.set_xticks(config.xticks)
            if config.xtick_labels is not None:
                ax.set_xticklabels(config.xtick_labels, fontsize=tick_fontsize)

        # Labels
        if config.xlabel:
            ax.set_xlabel(config.xlabel, fontsize=xlabel_fontsize)
        if config.ylabel:
            ax.set_ylabel(config.ylabel, fontsize=ylabel_fontsize)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, fontsize=tick_fontsize)

        # Y-axis limits (value axis) - explicit ylim takes precedence
        if config.ylim is not None:
            ax.set_ylim(config.ylim)
        else:
            if config.value_max is None:
                max_val = 0
                for data in group_data.values():
                    vals = data["values"]
                    errs = (
                        data["err_high"]
                        if data["err_high"] is not None
                        else np.zeros_like(vals)
                    )
                    max_val = max(max_val, (vals + errs).max())
                # Add annotation padding if annotations enabled
                annot_extra = config.annot_padding if config.show_annotations else 0
                padding = max(5, max_val * config.value_padding_pct) + annot_extra
                value_max = max(10, max_val + padding)
            else:
                value_max = config.value_max

            value_min = config.value_min if config.value_min is not None else 0
            ax.set_ylim(value_min, value_max)

        # Custom y-ticks
        if config.yticks is not None:
            ax.set_yticks(config.yticks)
            if config.ytick_labels is not None:
                ax.set_yticklabels(config.ytick_labels, fontsize=tick_fontsize)

        # Labels
        if config.xlabel:
            ax.set_xlabel(config.xlabel, fontsize=xlabel_fontsize)
        if config.ylabel:
            ax.set_ylabel(config.ylabel, fontsize=ylabel_fontsize)

    # Title
    if config.title:
        ax.set_title(
            config.title, fontsize=title_fontsize, fontweight=config.title_fontweight
        )

    # Legend (only if grouped and show_legend is True)
    if is_grouped and config.show_legend and n_groups > 1:
        legend_kwargs = {
            "loc": config.legend_loc,
            "frameon": False,
            "handlelength": 2.0,
            "handleheight": 1.0,
            "handletextpad": 0.6,
            "labelspacing": 0.6,
            "borderaxespad": 0.5,
            "prop": {"size": legend_fontsize},
        }
        if config.legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = config.legend_bbox_to_anchor
        if config.legend_title:
            legend_kwargs["title"] = config.legend_title
        ax.legend(**legend_kwargs)

    # Annotations
    if config.show_annotations:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for i in range(n_categories):
            for j, group in enumerate(groups):
                data = group_data[group]
                val = data["values"][i]
                err = data["err_high"][i] if data["err_high"] is not None else 0
                pos = positions[i] + offsets[j]

                if config.orientation == "horizontal":
                    xpos = val + err + config.annot_offset
                    ypos = pos
                    t = ax.text(
                        xpos,
                        ypos,
                        config.annot_format.format(val),
                        va="center",
                        fontsize=annot_fontsize,
                    )
                    # Adjust vertical position slightly
                    bbox = t.get_window_extent(renderer=renderer)
                    text_h_pixels = bbox.height
                    dy_pixels = -0.1 * text_h_pixels
                    trans = ax.transData + mtrans.ScaledTranslation(
                        0, dy_pixels / fig.dpi, fig.dpi_scale_trans
                    )
                    t.set_transform(trans)
                else:
                    xpos = pos
                    ypos = val + err + config.annot_offset
                    ax.text(
                        xpos,
                        ypos,
                        config.annot_format.format(val),
                        ha="center",
                        fontsize=annot_fontsize,
                    )

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Plotter Class
# =============================================================================


class GroupedBarPlotter:
    """
    Class-based interface for grouped bar plots.

    Supports:
    - Direct plotting from pre-computed DataFrames
    - Computation + plotting from raw group configurations (for proportion CIs)
    - Both grouped and simple (ungrouped) bar plots
    - Horizontal and vertical orientations

    Examples
    --------
    >>> # Simple bar plot (no grouping)
    >>> config = GroupedBarConfig(
    ...     group_col=None,
    ...     orientation="vertical",
    ... )
    >>> plotter = GroupedBarPlotter(df, config)
    >>> fig, ax = plotter.plot()

    >>> # Grouped horizontal bars with CI
    >>> config = GroupedBarConfig(
    ...     ci_low_col="ci_low",
    ...     ci_high_col="ci_high",
    ...     orientation="horizontal",
    ... )
    >>> plotter = GroupedBarPlotter(df, config)
    >>> fig, ax = plotter.plot()

    >>> # From raw counts (computes Clopper-Pearson CI)
    >>> plotter = GroupedBarPlotter.from_proportions(
    ...     category_list=["TP53", "KRAS", "CDKN2A"],
    ...     group_configs=[
    ...         {"name": "Treatment", "k": gene_counts_trt, "n": n_treatment},
    ...         {"name": "Control", "k": gene_counts_ctrl, "n": n_control},
    ...     ],
    ...     config=GroupedBarConfig(ci_method="clopper"),
    ... )
    >>> fig, ax = plotter.plot()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: GroupedBarConfig | None = None,
    ):
        """
        Initialize with data.

        Parameters
        ----------
        data : pd.DataFrame
            Data with columns matching config column names.
        config : GroupedBarConfig, optional
            Configuration. Uses defaults if not provided.
        """
        self.data = data
        self.config = config if config is not None else GroupedBarConfig()

    @classmethod
    def from_proportions(
        cls,
        category_list: list[str],
        group_configs: list[dict[str, Any]],
        config: GroupedBarConfig | None = None,
        value_scale: float = 100.0,
    ) -> "GroupedBarPlotter":
        """
        Create plotter by computing proportion summary from group configurations.

        Parameters
        ----------
        category_list : list[str]
            List of categories to analyze.
        group_configs : list[dict]
            Each dict should contain:
                - 'name': str, group name
                - 'k': dict or Series, category -> count mapping
                - 'n': int, total group size
        config : GroupedBarConfig, optional
            Configuration for CI method and visual settings.
        value_scale : float
            Scale factor (100.0 for percentages).

        Returns
        -------
        GroupedBarPlotter
            Instance with computed summary data.
        """
        if config is None:
            config = GroupedBarConfig()

        # Determine CI method
        ci_method = (
            config.ci_method if config.ci_method != "none" else "clopper-pearson"
        )

        summary_df = compute_proportion_summary(
            category_list=category_list,
            group_configs=group_configs,
            method=ci_method,
            alpha=config.alpha,
            n_boot=config.n_boot,
            random_state=config.random_state,
            value_scale=value_scale,
        )

        # Update config to use computed columns
        updated_config = config.model_copy(
            update={
                "category_col": "Category",
                "group_col": "Group",
                "value_col": "value",
                "ci_low_col": "ci_low",
                "ci_high_col": "ci_high",
                "k_col": "k",
                "n_col": "n",
            }
        )

        return cls(summary_df, updated_config)

    def plot(self, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Generate the bar plot.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to draw into. Creates new figure if None.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The figure and axes objects.
        """
        return plot_grouped_bars(self.data, self.config, ax=ax)
