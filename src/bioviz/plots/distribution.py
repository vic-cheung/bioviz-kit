"""
Distribution plot helpers for bioviz.

Pure plotting helpers: functions accept Axes and data and draw the plot.
These helpers do not perform DataFrame checks or file IO.
"""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from bioviz.utils.style import DefaultStyle
from bioviz.utils.style import StyleBase
from typing import Any
from bioviz.configs.distribution_cfg import DistributionConfig
import pandas as pd


def _default_color(style: Optional[StyleBase] = None) -> str:
    s = style or DefaultStyle()
    vals = list(getattr(s, "palette", {}).values())
    return vals[0] if vals else "#009E73"


def generate_histogram(
    ax,
    plot_data,
    variable_name: str,
    indication: str,
    median_value: float,
    title_prefix: Optional[str] = None,
    bins: int = 20,
    alpha: float = 0.7,
    grid_alpha: float = 0.3,
    hist_grid: bool = False,
    hist_alpha: Optional[float] = None,
    color: Optional[str] = None,
    edgecolor: str = "black",
    median_color: str = "black",
    median_linestyle: str = "--",
    median_linewidth: int = 2,
    median_alpha: float = 0.8,
    style: Optional[StyleBase] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    title_template: Optional[str] = None,
    median_label_fmt: str = "Median = {median:.2f}",
    show_median_label: bool = True,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    median_label_fontsize: int = 10,
    median_label_location: str = "auto",
    **kwargs,
):
    """
    Render a histogram with median line and annotation onto `ax`.

    This function is intentionally pure plotting: it does not perform data
    validation or saving.
    """
    # If grouped data handling: caller may pass `hue` and grouped data via kwargs
    hue = kwargs.get("hue", None)
    hue_palette = kwargs.get("hue_palette", None)
    hist_mode = kwargs.get("hist_mode", "bar")
    hist_hue_overlap = kwargs.get("hist_hue_overlap", True)
    hue_alpha = kwargs.get("hue_alpha", None)

    if color is None:
        color = _default_color(style)

    ax.set_axisbelow(True)
    if hist_grid:
        ax.grid(True, alpha=grid_alpha)

    # If hue requested and plot_data is a DataFrame/Series with groups, handle grouped hist/KDE
    if hue is None:
        ax.hist(
            plot_data,
            bins=bins,
            alpha=alpha if hist_alpha is None else hist_alpha,
            color=color,
            edgecolor=edgecolor,
            zorder=2,
        )
    else:
        # Expect plot_data as a DataFrame when hue is provided
        try:
            groups = plot_data.groupby(hue, observed=False)
        except Exception:
            # fallback to simple histogram
            ax.hist(
                plot_data,
                bins=bins,
                alpha=alpha if hist_alpha is None else hist_alpha,
                color=color,
                edgecolor=edgecolor,
                zorder=2,
            )
        else:
            # resolve colors
            group_names = list(groups.groups.keys())
            if hue_palette is None:
                s = style or DefaultStyle()
                palette_vals = list(getattr(s, "palette", {}).values())
                # cycle palette if not enough
                colors = {g: palette_vals[i % len(palette_vals)] for i, g in enumerate(group_names)}
            # shared bins
            all_vals = plot_data[kwargs.get("value_col")] if kwargs.get("value_col") else plot_data
            bins_edges = np.histogram_bin_edges(all_vals.dropna(), bins=bins)

            for g in group_names:
                vals = groups.get_group(g)
                if kwargs.get("value_col"):
                    vals = vals[kwargs.get("value_col")]
                vals = vals.dropna()
                c = colors.get(g, _default_color(style))
                a = (
                    hue_alpha
                    if hue_alpha is not None
                    else (alpha if hist_alpha is None else hist_alpha)
                )
                if hist_mode in ("bar", "both"):
                    ax.hist(
                        vals,
                        bins=bins_edges,
                        alpha=a,
                        color=c,
                        edgecolor=edgecolor,
                        density=False,
                        zorder=2,
                    )
                if hist_mode in ("kde", "both"):
                    # try scipy if available
                    try:
                        from scipy.stats import gaussian_kde

                        kde = gaussian_kde(vals)
                        xs = np.linspace(bins_edges[0], bins_edges[-1], 200)
                        ys = kde(xs)
                    except Exception:
                        # fallback simple gaussian KDE via numpy
                        xs = np.linspace(bins_edges[0], bins_edges[-1], 200)
                        from math import sqrt

                        bw = (
                            1.06 * vals.std(ddof=1) * (len(vals) ** (-1 / 5))
                            if len(vals) > 1
                            else 1.0
                        )
                        ys = np.zeros_like(xs)
                        for v in vals:
                            ys += np.exp(-0.5 * ((xs - v) / bw) ** 2) / (bw * sqrt(2 * np.pi))
                        ys = ys / len(vals)
                    ax.plot(xs, ys, color=c, linewidth=1.25, zorder=3)
            if kwargs.get("hue_swarm_legend", True):
                # add legend entries with group medians
                # compute medians per group and format labels
                group_median_fmt = kwargs.get("group_median_fmt", "{group} (median = {median:.2f})")
                handles = []
                labels = []
                for g in group_names:
                    vals = groups.get_group(g)
                    if kwargs.get("value_col"):
                        vals = vals[kwargs.get("value_col")]
                    vals = vals.dropna()
                    m = float(np.nanmedian(vals)) if len(vals) > 0 else float("nan")
                    labels.append(f"{g} (Median = {m:.2f})")
                    handles.append(plt.Line2D([0], [0], color=colors[g], lw=4))
                # place legend outside the axes on the right centered vertically with no frame
                ax.legend(
                    handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False
                )

    ax.axvline(
        median_value,
        color=median_color,
        linestyle=median_linestyle,
        linewidth=median_linewidth,
        alpha=median_alpha,
        zorder=3,
    )
    if show_median_label:
        try:
            # when hue/grouping is present, move the global median label off-plot near legend
            if hue is not None:
                label_text = f"Global Median = {median_value:.2f}"
                ax.annotate(
                    label_text,
                    xy=(1.02, 0.95),
                    xycoords="axes fraction",
                    xytext=(6, 0),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=median_label_fontsize,
                    color=median_color,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
            else:
                label_text = median_label_fmt.format(median=median_value)
                if median_label_location == "upper_right":
                    ax.annotate(
                        label_text,
                        xy=(0.98, 0.95),
                        xycoords="axes fraction",
                        xytext=(-4, -4),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=median_label_fontsize,
                        color=median_color,
                        horizontalalignment="right",
                        verticalalignment="top",
                    )
                elif median_label_location == "off_right":
                    ax.annotate(
                        label_text,
                        xy=(1.01, 0.95),
                        xycoords="axes fraction",
                        xytext=(6, 0),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=median_label_fontsize,
                        color=median_color,
                        horizontalalignment="left",
                        verticalalignment="top",
                    )
                else:
                    ax.annotate(
                        label_text,
                        xy=(median_value, ax.get_ylim()[1] * 0.9),
                        xytext=(10, -10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=median_label_fontsize,
                        color=median_color,
                    )
        except Exception:
            pass

    ax.set_xlabel(xlabel or variable_name, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel or "Frequency", fontsize=ylabel_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(xtick_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(ytick_fontsize)

    if title is not None:
        final_title = title
    elif title_template is not None:
        try:
            final_title = title_template.format(indication=indication, variable=variable_name)
        except Exception:
            final_title = f"{indication}: {variable_name} Distribution"
    else:
        final_title = f"{indication}: {variable_name} Distribution"

    if title_prefix:
        final_title = f"{title_prefix} - {final_title}"

    ax.set_title(final_title, fontsize=title_fontsize)


def generate_horizontal_boxplot_with_swarm(
    ax,
    plot_data,
    variable_name: str,
    indication: str,
    title_prefix: Optional[str] = None,
    alpha: float = 0.7,
    grid_alpha: float = 0.3,
    box_grid: bool = False,
    box_alpha: Optional[float] = None,
    box_color: Optional[str] = None,
    median_color: str = "red",
    median_linewidth: int = 2,
    swarm_facecolor: str = "white",
    swarm_edgecolor: str = "black",
    swarm_linewidth: float = 0.5,
    swarm_size: int = 40,
    swarm_alpha: float = 0.8,
    jitter_std: float = 0.02,
    random_seed: int = 42,
    style: Optional[StyleBase] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    y_ticks: Optional[list] = None,
    y_ticklabels: Optional[list] = None,
    ylim: Optional[tuple] = None,
    title: Optional[str] = None,
    title_template: Optional[str] = None,
    color: Optional[str] = None,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    show_box_median_label: bool = False,
    box_median_label_location: str = "auto",
    median_label_fontsize: int = 10,
    **kwargs,
):
    """
    Render a horizontal boxplot with a jittered swarm overlay onto `ax`.
    """
    # Support 'color' alias (e.g., passed from plot_distribution as hist color)
    if box_color is None:
        # If grouping/hue is present and caller didn't set box_color, use a neutral light gray
        if kwargs.get("hue") and isinstance(plot_data, pd.DataFrame):
            box_color = "#f0f0f0"
        elif color is not None:
            box_color = color
        else:
            box_color = _default_color(style)

    ax.set_axisbelow(True)
    if box_grid:
        ax.grid(True, alpha=grid_alpha)

    # If this is a single grouped boxplot (hue present and plot_data is DataFrame), use neutral white background
    if kwargs.get("hue") and isinstance(plot_data, pd.DataFrame):
        try:
            ax.set_facecolor("white")
        except Exception:
            pass

    # Determine numeric values to feed to boxplot (support DataFrame + value_col)
    if kwargs.get("hue") and kwargs.get("value_col") and isinstance(plot_data, pd.DataFrame):
        box_vals = plot_data[kwargs.get("value_col")].dropna()
    elif isinstance(plot_data, pd.DataFrame) and kwargs.get("value_col"):
        box_vals = plot_data[kwargs.get("value_col")].dropna()
    else:
        # attempt to coerce to numeric if needed
        try:
            box_vals = pd.to_numeric(plot_data, errors="coerce").dropna()
        except Exception:
            # fallback to list conversion
            try:
                box_vals = pd.Series(list(plot_data)).dropna()
            except Exception:
                box_vals = pd.Series([])

    ax.boxplot(
        box_vals,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=box_color, alpha=box_alpha if box_alpha is not None else alpha),
        medianprops=dict(color=median_color, linewidth=median_linewidth),
        showfliers=False,
        zorder=1,
    )

    np.random.seed(random_seed)
    y_positions = np.ones(len(box_vals)) + np.random.normal(0, jitter_std, len(box_vals))

    # scatter colored by hue if provided
    hue = kwargs.get("hue", None)
    if hue is None:
        ax.scatter(
            box_vals,
            y_positions,
            facecolors=swarm_facecolor,
            edgecolors=swarm_edgecolor,
            linewidths=swarm_linewidth,
            s=swarm_size,
            alpha=swarm_alpha,
            zorder=3,
        )
    else:
        df = plot_data
        vals = df[kwargs.get("value_col")] if kwargs.get("value_col") else df
        groups = df.groupby(hue, observed=False)
        group_names = list(groups.groups.keys())
        hue_palette = kwargs.get("hue_palette")
        if hue_palette is None:
            s = style or DefaultStyle()
            palette_vals = list(getattr(s, "palette", {}).values())
            colors = {g: palette_vals[i % len(palette_vals)] for i, g in enumerate(group_names)}
        else:
            colors = {g: hue_palette.get(g, _default_color(style)) for g in group_names}

        handles = []
        for g in group_names:
            grp = groups.get_group(g)
            gvals = grp[kwargs.get("value_col")] if kwargs.get("value_col") else grp
            gp_y = np.ones(len(gvals)) + np.random.normal(
                0, kwargs.get("jitter_std", 0.02), len(gvals)
            )
            h = ax.scatter(
                gvals,
                gp_y,
                facecolors=colors[g],
                edgecolors=kwargs.get("swarm_edgecolor", swarm_edgecolor),
                linewidths=kwargs.get("swarm_linewidth", swarm_linewidth),
                s=kwargs.get("swarm_size", swarm_size),
                alpha=kwargs.get("swarm_alpha", swarm_alpha),
                zorder=3,
            )
            handles.append(h)
        if kwargs.get("hue_swarm_legend", True):
            # labels with medians
            labels = []
            for i, g in enumerate(group_names):
                grp = groups.get_group(g)
                gvals = grp[kwargs.get("value_col")] if kwargs.get("value_col") else grp
                gvals = gvals.dropna()
                m = float(np.nanmedian(gvals)) if len(gvals) > 0 else float("nan")
                labels.append(f"{g} (Median = {m:.2f})")
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        # draw per-group medians if requested
        if kwargs.get("show_group_medians", False):
            # compute medians explicitly to avoid pandas FutureWarning about apply
            try:
                if kwargs.get("value_col"):
                    medians = df.groupby(hue, observed=False)[kwargs.get("value_col")].median()
                else:
                    # if no value_col specified, attempt to compute median of numeric columns
                    # use numeric_only=True to avoid grouping columns being included
                    medians = df.groupby(hue, observed=False).median(numeric_only=True).iloc[:, 0]
            except Exception:
                # fallback to safer apply-on-selected column
                medians = (
                    df.groupby(hue, observed=False)[kwargs.get("value_col")].median()
                    if kwargs.get("value_col")
                    else df.groupby(hue, observed=False).median().iloc[:, 0]
                )

            for g in medians.index:
                m = medians.loc[g]
                c = colors.get(g, _default_color(style))
                ax.axvline(m, color=c, linestyle="--", linewidth=1.0, zorder=4)
                if kwargs.get("group_median_label", "legend") == "onplot":
                    ax.annotate(
                        str(m),
                        xy=(m, 0.98),
                        xycoords="axes fraction",
                        fontsize=kwargs.get("median_label_fontsize", 10),
                        color=c,
                    )

    ax.set_xlabel(xlabel or variable_name, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel or "Distribution", fontsize=ylabel_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(xtick_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(ytick_fontsize)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([1])

    if y_ticklabels is not None:
        ax.set_yticklabels(y_ticklabels)
    else:
        ax.set_yticklabels([""])

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0.5, 1.5)

    if title is not None:
        final_title = title
    elif title_template is not None:
        try:
            final_title = title_template.format(indication=indication, variable=variable_name)
        except Exception:
            final_title = f"{indication}: {variable_name} Boxplot with Swarm Overlay"
    else:
        final_title = f"{indication}: {variable_name} Boxplot with Swarm Overlay"

    if title_prefix:
        final_title = f"{title_prefix} - {final_title}"

    ax.set_title(final_title, fontsize=title_fontsize)


def generate_grouped_boxplots(
    ax,
    plot_data,
    variable_name: str,
    indication: str,
    value_col: Optional[str] = None,
    hue: Optional[str] = None,
    hue_palette: Optional[dict] = None,
    style: Optional[StyleBase] = None,
    title: Optional[str] = None,
    title_prefix: Optional[str] = None,
    title_fontsize: int = 12,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    box_alpha: Optional[float] = None,
    show_group_medians: bool = True,
    group_median_marker: str = "v",
    group_median_markersize: int = 8,
    # optional median-label controls for the grouped boxplot panel
    show_box_median_label: bool = False,
    box_median_label_location: str = "auto",
    median_label_fontsize: int = 10,
    # swarm overlay options for grouped boxes
    swarm: bool = False,
    swarm_size: int = 30,
    swarm_alpha: float = 1.0,
    swarm_facecolor: Optional[str] = None,
    swarm_edgecolor: str = "black",
    swarm_jitter: float = 0.05,
    # box shading alpha (boxes will be colored with this alpha)
    per_box_alpha: Optional[float] = None,
    group_order: Optional[list] = None,
):
    """Render horizontal grouped boxplots where each group gets its own box."""
    if hue is None or not isinstance(plot_data, pd.DataFrame):
        # fallback to single box
        generate_horizontal_boxplot_with_swarm(
            ax,
            plot_data,
            variable_name,
            indication,
            style=style,
            title=title,
            title_prefix=title_prefix,
        )
        return

    # determine group order: explicit `group_order` -> categorical dtype order -> observed order
    if group_order is not None:
        group_names = list(group_order)
    else:
        # if hue column is categorical with ordered categories, respect that
        if hue in plot_data.columns and pd.api.types.is_categorical_dtype(plot_data[hue]):
            group_names = list(plot_data[hue].cat.categories)
        else:
            groups = plot_data.groupby(hue, observed=False)
            group_names = list(groups.groups.keys())
            # ensure groups variable is available for later use
            try:
                groups = plot_data.groupby(hue, observed=False)
            except Exception:
                groups = None
    data_list = []
    # build data_list according to determined group_names
    data_list = []
    for g in group_names:
        if hue in plot_data.columns:
            grp = plot_data[plot_data[hue] == g]
        else:
            grp = plot_data
        vals = grp[value_col] if value_col else grp
        data_list.append(vals.dropna())

    # resolve colors per group
    if hue_palette is None:
        s = style or DefaultStyle()
        palette_vals = list(getattr(s, "palette", {}).values())
        colors = [palette_vals[i % len(palette_vals)] for i in range(len(group_names))]
    else:
        colors = [hue_palette.get(g, _default_color(style)) for g in group_names]

    # positions: natural ordering (1..N); we'll invert the y-axis for top-first display
    positions = list(range(1, len(data_list) + 1))
    bp = ax.boxplot(
        data_list,
        vert=False,
        patch_artist=True,
        positions=positions,
        showfliers=False,
        widths=0.6,
    )

    # color boxes
    for box, c in zip(bp.get("boxes", []), colors):
        try:
            box.set_facecolor(c)
            # apply per-box alpha if provided else fall back to box_alpha or 1.0
            box.set_alpha(
                per_box_alpha
                if per_box_alpha is not None
                else (box_alpha if box_alpha is not None else 1.0)
            )
            box.set_edgecolor("black")
        except Exception:
            pass

    # style median lines from the boxplot artists so they align with box widths
    medians = bp.get("medians", [])
    if show_group_medians and medians:
        for i, median_line in enumerate(medians):
            try:
                median_line.set_color("black")
                median_line.set_linewidth(1.5)
                median_line.set_zorder(4)
            except Exception:
                pass
    else:
        # fallback: if medians not present, place small black markers at median values
        if show_group_medians:
            for i, g in enumerate(group_names):
                vals = data_list[i]
                m = float(np.nanmedian(vals)) if len(vals) > 0 else float("nan")
                ax.scatter(
                    m,
                    positions[i],
                    marker=group_median_marker,
                    s=group_median_markersize,
                    color="black",
                    zorder=4,
                )

    # add legend entries for each group (color + median)
    try:
        handles = []
        labels = []
        medians_for_legend = [
            float(np.nanmedian(d)) if len(d) > 0 else float("nan") for d in data_list
        ]
        for i, g in enumerate(group_names):
            c = colors[i]
            handles.append(plt.Line2D([0], [0], color=c, lw=4))
            labels.append(f"{g} (Median = {medians_for_legend[i]:.2f})")
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    except Exception:
        pass

    ax.set_yticks(positions)
    ax.set_yticklabels(group_names)
    # invert y-axis so the first group in `group_names` appears at the top visually
    try:
        ax.invert_yaxis()
    except Exception:
        pass
    for tick in ax.get_xticklabels():
        tick.set_fontsize(xtick_fontsize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(ytick_fontsize)

    final_title = title or f"{indication}: {variable_name} Grouped Boxplots"
    if title_prefix:
        final_title = f"{title_prefix} - {final_title}"
    ax.set_title(final_title, fontsize=title_fontsize)

    # optional per-group swarm overlay
    if swarm and hue and isinstance(plot_data, pd.DataFrame):
        np.random.seed(42)
        groups = plot_data.groupby(hue, observed=False)
        for i, g in enumerate(group_names):
            grp = groups.get_group(g)
            gvals = grp[value_col] if value_col else grp
            gvals = gvals.dropna()
            if len(gvals) == 0:
                continue
            # jitter y positions around the group's y position
            y_base = positions[i]
            jitter = np.random.normal(0, swarm_jitter, size=len(gvals))
            y_positions = y_base + jitter
            face = swarm_facecolor if swarm_facecolor is not None else colors[i]
            ax.scatter(
                gvals,
                y_positions,
                s=swarm_size,
                alpha=swarm_alpha,
                facecolors=face,
                edgecolors=swarm_edgecolor,
                linewidths=0.5,
                zorder=6,
            )

    # optional median label for boxplot (compute combined median across groups)
    if show_box_median_label:
        try:
            if len(data_list) > 0:
                combined = pd.concat(data_list)
            else:
                combined = pd.Series([])
            median_val = float(np.nanmedian(combined)) if len(combined) > 0 else 0.0
        except Exception:
            try:
                combined = pd.to_numeric(pd.concat(data_list), errors="coerce").dropna()
                median_val = float(np.nanmedian(combined)) if len(combined) > 0 else 0.0
            except Exception:
                median_val = 0.0

        if box_median_label_location == "upper_right":
            ax.annotate(
                f"Median = {median_val:.2f}",
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                xytext=(-4, -4),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=median_label_fontsize,
                horizontalalignment="right",
                verticalalignment="top",
            )
        elif box_median_label_location == "off_right":
            ax.annotate(
                f"Median = {median_val:.2f}",
                xy=(1.01, 0.95),
                xycoords="axes fraction",
                xytext=(6, 0),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=median_label_fontsize,
                horizontalalignment="left",
                verticalalignment="top",
            )
        else:
            ax.annotate(
                f"Median = {median_val:.2f}",
                xy=(median_val, ax.get_ylim()[1] * 0.9),
                xytext=(10, -10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=median_label_fontsize,
            )


def plot_distribution(
    axes: Tuple[plt.Axes, plt.Axes],
    plot_data,
    variable_name: str,
    indication: str,
    title_prefix: Optional[str] = None,
    bins: int = 20,
    style: Optional[StyleBase] = None,
    title: Optional[str] = None,
    title_template: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    y_ticks: Optional[list] = None,
    y_ticklabels: Optional[list] = None,
    ylim: Optional[tuple] = None,
    **kwargs,
):
    """
    Convenience: render histogram on axes[0] and box+swarm on axes[1].
    """
    ax_hist, ax_box = axes
    # compute median robustly: if DataFrame with a provided value_col, use that column
    if isinstance(plot_data, pd.DataFrame) and kwargs.get("value_col"):
        median_series = plot_data[kwargs.get("value_col")].dropna()
        median_value = float(np.nanmedian(median_series)) if len(median_series) > 0 else 0.0
    else:
        try:
            median_value = float(np.nanmedian(plot_data))
        except Exception:
            # fallback: coerce to numeric (will turn non-numeric to NaN) and compute
            try:
                coerced = pd.to_numeric(plot_data, errors="coerce")
                median_value = float(np.nanmedian(coerced.dropna()))
            except Exception:
                median_value = 0.0
    generate_histogram(
        ax_hist,
        plot_data,
        variable_name,
        indication,
        median_value,
        title_prefix=title_prefix,
        bins=bins,
        title=title,
        title_template=title_template,
        xlabel=xlabel,
        ylabel=ylabel,
        style=style,
        **kwargs,
    )
    generate_horizontal_boxplot_with_swarm(
        ax_box,
        plot_data,
        variable_name,
        indication,
        title_prefix=title_prefix,
        title=title,
        title_template=title_template,
        xlabel=xlabel,
        ylabel=ylabel,
        y_ticks=y_ticks,
        y_ticklabels=y_ticklabels,
        ylim=ylim,
        style=style,
        **kwargs,
    )


class DistributionPlotter:
    """
    Stateful plotter for distribution plots.

    Usage:
        p = DistributionPlotter(data=series, config=DistributionConfig())
        fig, axes = p.plot()  # or p.plot(return_fig=True)
        p.save(path)
        p.close()
    """

    def __init__(self, data: Any = None, config: DistributionConfig | dict | None = None):
        if isinstance(config, dict):
            config = DistributionConfig(**config)
        self.df = None
        self.data = data
        self.config: DistributionConfig = config or DistributionConfig()
        self.fig: Optional[plt.Figure] = None
        self.axes: Optional[Tuple[plt.Axes, plt.Axes]] = None

    def set_data(self, data: Any) -> "DistributionPlotter":
        self.data = data
        return self

    def update_config(self, **kwargs) -> "DistributionPlotter":
        # Use pydantic's copy/update to validate incoming values
        try:
            self.config = self.config.copy(update=kwargs)
        except Exception:
            # Fallback: set attributes individually
            for k, v in kwargs.items():
                try:
                    setattr(self.config, k, v)
                except Exception:
                    continue
        return self

    def plot(self, data: Any = None, *, return_fig: Optional[bool] = None):
        """
        Render plot according to config. Returns fig (and axes) when requested.

        If `show_hist` or `show_box` are False in config, the corresponding
        panel will be omitted (if both False, raises ValueError).
        """
        if data is not None:
            self.set_data(data)

        if self.data is None:
            raise RuntimeError("No data provided to DistributionPlotter")

        cfg = self.config
        if return_fig is None:
            return_fig = cfg.return_fig

        plot_data = self.data
        # Ensure Series-like with dropna
        try:
            if isinstance(plot_data, np.ndarray):
                plot_data = pd.Series(plot_data).dropna()
            else:
                plot_data = plot_data.dropna()
        except Exception:
            # fallback attempt
            plot_data = pd.Series(plot_data).dropna()

        if not cfg.show_hist and not cfg.show_box:
            raise ValueError("At least one of show_hist or show_box must be True")

        # create subplots according to what to show
        if cfg.show_hist and cfg.show_box:
            # if hue provided, include a third panel for grouped boxplots
            if cfg.hue:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=cfg.figsize)
                try:
                    fig.subplots_adjust(right=0.78)
                except Exception:
                    pass
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cfg.figsize)
            # ensure consistent figure background and transparency
            try:
                face = getattr(cfg, "figure_facecolor", None) or "white"
                transparent = getattr(cfg, "figure_transparent", False)
                fig.patch.set_facecolor(face)
                fig.patch.set_alpha(0.0 if transparent else 1.0)
            except Exception:
                pass
            if cfg.hue:
                plot_distribution(
                    (ax1, ax2),
                    plot_data,
                    getattr(self, "variable_name", "") or "",
                    getattr(self, "indication", "") or "",
                    title_prefix=cfg.title_prefix,
                    bins=cfg.bins,
                    alpha=cfg.alpha,
                    grid_alpha=cfg.grid_alpha,
                    hist_grid=cfg.hist_grid,
                    box_grid=cfg.box_grid,
                    style=cfg.style,
                    title=cfg.title,
                    title_template=cfg.title_template,
                    xlabel=cfg.xlabel,
                    ylabel=cfg.ylabel,
                    y_ticks=cfg.y_ticks,
                    y_ticklabels=cfg.y_ticklabels,
                    ylim=cfg.ylim,
                    color=cfg.hist_color,
                    edgecolor=cfg.hist_edgecolor,
                    median_color=cfg.median_color,
                    median_linestyle=cfg.median_linestyle,
                    median_linewidth=cfg.median_linewidth,
                    median_alpha=cfg.median_alpha,
                    median_label_fmt=cfg.median_label_fmt,
                    show_median_label=cfg.show_median_label,
                    median_label_location=cfg.median_label_location,
                    median_label_fontsize=cfg.median_label_fontsize,
                    title_fontsize=cfg.title_fontsize,
                    xlabel_fontsize=cfg.xlabel_fontsize,
                    ylabel_fontsize=cfg.ylabel_fontsize,
                    xtick_fontsize=cfg.xtick_fontsize,
                    ytick_fontsize=cfg.ytick_fontsize,
                    swarm_facecolor=cfg.swarm_facecolor,
                    swarm_edgecolor=cfg.swarm_edgecolor,
                    swarm_linewidth=cfg.swarm_linewidth,
                    swarm_size=cfg.swarm_size,
                    swarm_alpha=cfg.swarm_alpha,
                    jitter_std=cfg.jitter_std,
                    random_seed=cfg.random_seed,
                    hist_alpha=cfg.hist_alpha,
                    box_alpha=cfg.box_alpha,
                    # grouping / hue forwarding
                    hue=cfg.hue,
                    value_col=cfg.value_col,
                    hue_palette=cfg.hue_palette,
                    hist_mode=cfg.hist_mode,
                    hist_hue_overlap=cfg.hist_hue_overlap,
                    hue_alpha=cfg.hue_alpha,
                    hue_swarm_legend=cfg.hue_swarm_legend,
                    show_group_medians=cfg.show_group_medians,
                    group_median_label=cfg.group_median_label,
                    group_median_fmt=cfg.group_median_fmt,
                    group_median_marker=cfg.group_median_marker,
                    group_median_markersize=cfg.group_median_markersize,
                )
                # render grouped boxplots separately in third axis
                generate_grouped_boxplots(
                    ax3,
                    plot_data,
                    getattr(self, "variable_name", "") or "",
                    getattr(self, "indication", "") or "",
                    value_col=cfg.value_col,
                    hue=cfg.hue,
                    hue_palette=cfg.hue_palette,
                    style=cfg.style,
                    title=cfg.title,
                    title_prefix=cfg.title_prefix,
                    title_fontsize=cfg.title_fontsize,
                    box_alpha=cfg.box_alpha,
                    per_box_alpha=cfg.box_alpha,
                    show_group_medians=cfg.show_group_medians,
                    group_median_marker=cfg.group_median_marker,
                    group_median_markersize=cfg.group_median_markersize,
                    # swarm options
                    swarm=getattr(cfg, "swarm", True),
                    swarm_size=cfg.swarm_size,
                    swarm_alpha=cfg.swarm_alpha,
                    swarm_facecolor=cfg.swarm_facecolor,
                    swarm_edgecolor=cfg.swarm_edgecolor,
                    swarm_jitter=cfg.jitter_std,
                    group_order=cfg.group_order,
                )
            else:
                plot_distribution(
                    (ax1, ax2),
                    plot_data,
                    getattr(self, "variable_name", "") or "",
                    getattr(self, "indication", "") or "",
                    title_prefix=cfg.title_prefix,
                    bins=cfg.bins,
                    alpha=cfg.alpha,
                    grid_alpha=cfg.grid_alpha,
                    hist_grid=cfg.hist_grid,
                    box_grid=cfg.box_grid,
                    style=cfg.style,
                    title=cfg.title,
                    title_template=cfg.title_template,
                    xlabel=cfg.xlabel,
                    ylabel=cfg.ylabel,
                    y_ticks=cfg.y_ticks,
                    y_ticklabels=cfg.y_ticklabels,
                    ylim=cfg.ylim,
                    color=cfg.hist_color,
                    edgecolor=cfg.hist_edgecolor,
                    median_color=cfg.median_color,
                    median_linestyle=cfg.median_linestyle,
                    median_linewidth=cfg.median_linewidth,
                    median_alpha=cfg.median_alpha,
                    median_label_fmt=cfg.median_label_fmt,
                    show_median_label=cfg.show_median_label,
                    median_label_location=cfg.median_label_location,
                    median_label_fontsize=cfg.median_label_fontsize,
                    title_fontsize=cfg.title_fontsize,
                    xlabel_fontsize=cfg.xlabel_fontsize,
                    ylabel_fontsize=cfg.ylabel_fontsize,
                    xtick_fontsize=cfg.xtick_fontsize,
                    ytick_fontsize=cfg.ytick_fontsize,
                    swarm_facecolor=cfg.swarm_facecolor,
                    swarm_edgecolor=cfg.swarm_edgecolor,
                    swarm_linewidth=cfg.swarm_linewidth,
                    swarm_size=cfg.swarm_size,
                    swarm_alpha=cfg.swarm_alpha,
                    jitter_std=cfg.jitter_std,
                    random_seed=cfg.random_seed,
                    hist_alpha=cfg.hist_alpha,
                    box_alpha=cfg.box_alpha,
                )
            self.fig = fig
            if cfg.hue:
                self.axes = (ax1, ax2, ax3)
            else:
                self.axes = (ax1, ax2)
        else:
            # single panel
            if cfg.show_hist:
                fig, ax = plt.subplots(figsize=cfg.figsize)
                try:
                    face = getattr(cfg, "figure_facecolor", None) or "white"
                    transparent = getattr(cfg, "figure_transparent", False)
                    fig.patch.set_facecolor(face)
                    fig.patch.set_alpha(0.0 if transparent else 1.0)
                except Exception:
                    pass
                # if DataFrame with value_col provided, compute median of that column
                if isinstance(plot_data, pd.DataFrame) and cfg.value_col:
                    median_value = float(np.nanmedian(plot_data[cfg.value_col].dropna()))
                else:
                    median_value = float(np.nanmedian(plot_data))
                generate_histogram(
                    ax,
                    plot_data,
                    getattr(self, "variable_name", "") or "",
                    getattr(self, "indication", "") or "",
                    median_value,
                    title_prefix=cfg.title_prefix,
                    bins=cfg.bins,
                    alpha=cfg.alpha,
                    grid_alpha=cfg.grid_alpha,
                    hist_grid=cfg.hist_grid,
                    style=cfg.style,
                    hue=cfg.hue,
                    value_col=cfg.value_col,
                    hue_palette=cfg.hue_palette,
                    hist_mode=cfg.hist_mode,
                    hist_hue_overlap=cfg.hist_hue_overlap,
                    hue_alpha=cfg.hue_alpha,
                    hue_swarm_legend=cfg.hue_swarm_legend,
                    color=cfg.hist_color,
                    edgecolor=cfg.hist_edgecolor,
                    median_color=cfg.median_color,
                    median_linestyle=cfg.median_linestyle,
                    median_linewidth=cfg.median_linewidth,
                    median_alpha=cfg.median_alpha,
                    median_label_fmt=cfg.median_label_fmt,
                    title=cfg.title,
                    title_template=cfg.title_template,
                    xlabel=cfg.xlabel,
                    ylabel=cfg.ylabel,
                    median_label_location=cfg.median_label_location,
                    median_label_fontsize=cfg.median_label_fontsize,
                    title_fontsize=cfg.title_fontsize,
                    xlabel_fontsize=cfg.xlabel_fontsize,
                    ylabel_fontsize=cfg.ylabel_fontsize,
                    xtick_fontsize=cfg.xtick_fontsize,
                    ytick_fontsize=cfg.ytick_fontsize,
                )
                self.fig = fig
                self.axes = (ax,)
            else:
                # If hue is present, create two stacked axes: overall box+swarm and grouped boxplots
                if cfg.hue:
                    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=cfg.figsize)
                    try:
                        face = getattr(cfg, "figure_facecolor", None) or "white"
                        transparent = getattr(cfg, "figure_transparent", False)
                        fig.patch.set_facecolor(face)
                        fig.patch.set_alpha(0.0 if transparent else 1.0)
                    except Exception:
                        pass
                    # overall box+swarm on top (neutral box background, per-group colored swarm)
                    generate_horizontal_boxplot_with_swarm(
                        ax_top,
                        plot_data,
                        getattr(self, "variable_name", "") or "",
                        getattr(self, "indication", "") or "",
                        title_prefix=cfg.title_prefix,
                        alpha=cfg.alpha,
                        grid_alpha=cfg.grid_alpha,
                        box_grid=cfg.box_grid,
                        style=cfg.style,
                        box_color=cfg.box_color,
                        median_color=cfg.median_color,
                        median_linewidth=cfg.median_linewidth,
                        swarm_facecolor=cfg.swarm_facecolor,
                        swarm_edgecolor=cfg.swarm_edgecolor,
                        swarm_linewidth=cfg.swarm_linewidth,
                        swarm_size=cfg.swarm_size,
                        swarm_alpha=cfg.swarm_alpha,
                        jitter_std=cfg.jitter_std,
                        random_seed=cfg.random_seed,
                        hue=cfg.hue,
                        value_col=cfg.value_col,
                        hue_palette=cfg.hue_palette,
                        title=cfg.title,
                        title_template=cfg.title_template,
                        xlabel=cfg.xlabel,
                        ylabel=cfg.ylabel,
                        show_box_median_label=cfg.show_box_median_label,
                        box_median_label_location=cfg.box_median_label_location,
                        median_label_fontsize=cfg.median_label_fontsize,
                        title_fontsize=cfg.title_fontsize,
                        xlabel_fontsize=cfg.xlabel_fontsize,
                        ylabel_fontsize=cfg.ylabel_fontsize,
                        xtick_fontsize=cfg.xtick_fontsize,
                        ytick_fontsize=cfg.ytick_fontsize,
                        y_ticks=cfg.y_ticks,
                        y_ticklabels=cfg.y_ticklabels,
                        ylim=cfg.ylim,
                    )
                    # grouped boxplots below
                    generate_grouped_boxplots(
                        ax_bottom,
                        plot_data,
                        getattr(self, "variable_name", "") or "",
                        getattr(self, "indication", "") or "",
                        value_col=cfg.value_col,
                        hue=cfg.hue,
                        hue_palette=cfg.hue_palette,
                        style=cfg.style,
                        title=cfg.title,
                        title_prefix=cfg.title_prefix,
                        title_fontsize=cfg.title_fontsize,
                        box_alpha=cfg.box_alpha,
                        per_box_alpha=cfg.box_alpha,
                        show_group_medians=cfg.show_group_medians,
                        group_median_marker=cfg.group_median_marker,
                        group_median_markersize=cfg.group_median_markersize,
                        swarm=getattr(cfg, "swarm", True),
                        swarm_size=cfg.swarm_size,
                        swarm_alpha=cfg.swarm_alpha,
                        swarm_facecolor=cfg.swarm_facecolor,
                        swarm_edgecolor=cfg.swarm_edgecolor,
                        swarm_jitter=cfg.jitter_std,
                        group_order=cfg.group_order,
                    )
                    self.fig = fig
                    self.axes = (ax_top, ax_bottom)
                else:
                    fig, ax = plt.subplots(figsize=cfg.figsize)
                    try:
                        face = getattr(cfg, "figure_facecolor", None) or "white"
                        transparent = getattr(cfg, "figure_transparent", False)
                        fig.patch.set_facecolor(face)
                        fig.patch.set_alpha(0.0 if transparent else 1.0)
                    except Exception:
                        pass
                    generate_horizontal_boxplot_with_swarm(
                        ax,
                        plot_data,
                        getattr(self, "variable_name", "") or "",
                        getattr(self, "indication", "") or "",
                        title_prefix=cfg.title_prefix,
                        alpha=cfg.alpha,
                        grid_alpha=cfg.grid_alpha,
                        box_grid=cfg.box_grid,
                        style=cfg.style,
                        box_color=cfg.box_color,
                        median_color=cfg.median_color,
                        median_linewidth=cfg.median_linewidth,
                        swarm_facecolor=cfg.swarm_facecolor,
                        swarm_edgecolor=cfg.swarm_edgecolor,
                        swarm_linewidth=cfg.swarm_linewidth,
                        swarm_size=cfg.swarm_size,
                        swarm_alpha=cfg.swarm_alpha,
                        jitter_std=cfg.jitter_std,
                        random_seed=cfg.random_seed,
                        hue=cfg.hue,
                        value_col=cfg.value_col,
                        hue_palette=cfg.hue_palette,
                        show_group_medians=cfg.show_group_medians,
                        group_median_label=cfg.group_median_label,
                        group_median_fmt=cfg.group_median_fmt,
                        group_median_marker=cfg.group_median_marker,
                        group_median_markersize=cfg.group_median_markersize,
                        title=cfg.title,
                        title_template=cfg.title_template,
                        xlabel=cfg.xlabel,
                        ylabel=cfg.ylabel,
                        show_box_median_label=cfg.show_box_median_label,
                        box_median_label_location=cfg.box_median_label_location,
                        median_label_fontsize=cfg.median_label_fontsize,
                        title_fontsize=cfg.title_fontsize,
                        xlabel_fontsize=cfg.xlabel_fontsize,
                        ylabel_fontsize=cfg.ylabel_fontsize,
                        xtick_fontsize=cfg.xtick_fontsize,
                        ytick_fontsize=cfg.ytick_fontsize,
                        y_ticks=cfg.y_ticks,
                        y_ticklabels=cfg.y_ticklabels,
                        ylim=cfg.ylim,
                    )
                    self.fig = fig
                    self.axes = (ax,)

        if not return_fig:
            plt.close(self.fig)

        return (self.fig, self.axes) if return_fig else True

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
            self.axes = None
