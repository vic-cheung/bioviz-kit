"""
Spider plot utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

import math

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import font_manager
from matplotlib.lines import Line2D

from bioviz.configs import (
    ScanOverlayPlotConfig,
    StyledSpiderPlotConfig,
)
from bioviz.plot_utils import adjust_legend
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

__all__ = [
    "generate_styled_spiderplot",
    "generate_styled_spiderplot_with_scan_overlay",
]


def generate_styled_spiderplot(
    df: pd.DataFrame,
    config: StyledSpiderPlotConfig,
    ax: plt.Axes | None = None,
    draw_legend: bool = True,
) -> tuple[plt.Figure, list[Line2D], list[str]]:
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    # Validate required long-format columns; adapters should perform any
    # forward-fill or reshaping before calling bioviz.
    required_cols = {config.group_col, config.x, config.y}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns for spiderplot: {missing}")

    if config.linestyle_col and config.linestyle_col not in df.columns:
        raise ValueError(f"Style column '{config.linestyle_col}' not found in DataFrame.")
    if config.markerstyle_col and config.markerstyle_col not in df.columns:
        raise ValueError(f"Style column '{config.markerstyle_col}' not found in DataFrame.")

    labels = sorted(df[config.group_col].unique())
    num_labels = len(labels)

    if config.palette is None:
        palette = sns.color_palette("Dark2", n_colors=num_labels)
        color_dict = dict(zip(labels, palette))
    elif isinstance(config.palette, dict):
        color_dict = config.palette
    else:
        if len(config.palette) < num_labels:
            raise ValueError("Palette list has fewer colors than required.")
        color_dict = dict(zip(labels, config.palette))

    for label in labels:
        subset = df[df[config.group_col] == label].dropna(subset=[config.x, config.y])

        linestyle_value = None
        if config.linestyle_col:
            non_na = subset[config.linestyle_col].dropna()
            if not non_na.empty:
                linestyle_value = non_na.mode().iloc[0]

        markerstyle_value = None
        if config.markerstyle_col:
            non_na = subset[config.markerstyle_col].dropna()
            if not non_na.empty:
                markerstyle_value = non_na.mode().iloc[0]

        linestyle = "-"
        if linestyle_value is not None and config.linestyle_dict:
            linestyle = config.linestyle_dict.get(linestyle_value, "-")

        marker = config.marker_style
        if markerstyle_value is not None and config.markerstyle_dict:
            marker = config.markerstyle_dict.get(markerstyle_value, config.marker_style)

        ax.plot(
            subset[config.x].cat.codes,
            subset[config.y],
            color=color_dict[label],
            linestyle=linestyle,
            linewidth=config.lw,
            zorder=14,
            alpha=0.5,
        )

        ax.scatter(
            x=subset[config.x].cat.codes,
            y=subset[config.y],
            color=color_dict[label],
            edgecolor="white",
            linewidth=1,
            s=config.markersize**2,
            marker=marker,
            zorder=15,
            label=label,
        )

    cat_to_pos = {cat: i for i, cat in enumerate(df[config.x].cat.categories)}
    if cat_to_pos:
        xmax = max(cat_to_pos.values())
        ax.set_xlim(0, xmax + 0.5)
        ax.set_xticks(list(cat_to_pos.values()))
        ax.set_xticklabels(list(cat_to_pos.keys()))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(config.xlabel or config.x, fontweight="bold")
    ax.set_ylabel(config.ylabel or r"$\Delta$ from First Timepoint", fontweight="bold")
    ax.set_title(
        config.title or r"$\Delta$ ddPCR Value from First Timepoint", loc="left", fontweight="bold"
    )

    handles = []
    if config.color_dict_subgroup:
        handles.append(Line2D([0], [0], color="none", label=config.subgroup_name))
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    markersize=config.markersize,
                    linewidth=config.lw,
                    label=label,
                    alpha=0.5,
                )
                for label, color in config.color_dict_subgroup.items()
            ]
        )
    else:
        handles.append(Line2D([0], [0], color="none", label=config.group_col))
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=color_dict[label],
                    marker="o",
                    markerfacecolor=color_dict[label],
                    markeredgecolor="white",
                    markeredgewidth=1,
                    markersize=config.markersize,
                    linewidth=config.lw,
                    label=label,
                    alpha=0.5,
                )
                for label in df[config.group_col].unique()
            ]
        )

    if config.linestyle_dict:
        handles.append(Line2D([0], [0], color="none", label=""))
        handles.append(Line2D([0], [0], color="none", label=config.linestyle_col))
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=None,
                    linewidth=config.lw,
                    label=label,
                    linestyle=linestyle,
                    alpha=1,
                )
                for label, linestyle in config.linestyle_dict.items()
            ]
        )

    if config.markerstyle_dict:
        handles.append(Line2D([0], [0], color="none", label=""))
        handles.append(Line2D([0], [0], color="none", label=config.markerstyle_col))
        handles.extend(
            [
                Line2D(
                    [0], [0], color="black", marker=markerstyle, linewidth=0, label=label, alpha=1
                )
                for label, markerstyle in config.markerstyle_dict.items()
            ]
        )

    labels = [h.get_label() for h in handles]

    if draw_legend:
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.25, 0.5),
            loc="center",
            frameon=False,
            prop=font_manager.FontProperties(
                family=DefaultStyle().font_family, size=14, weight="bold"
            ),
        )
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    return fig, handles, labels


def generate_styled_spiderplot_with_scan_overlay(
    df: pd.DataFrame | None,
    scan_data: pd.DataFrame | None,
    spider_config: StyledSpiderPlotConfig | None,
    scan_overlay_config: ScanOverlayPlotConfig | None,
    recist_color_dict: dict[str, str] | None = None,
) -> plt.Figure:
    has_df = df is not None and not df.empty
    has_scan = scan_data is not None and not scan_data.empty
    if not has_df and not has_scan:
        raise ValueError("At least one of df or scan_data must be provided and non-empty.")

    fig, ax = plt.subplots(
        figsize=(
            spider_config.figsize
            if spider_config is not None
            else (scan_overlay_config.figsize if scan_overlay_config is not None else (9, 6))
        )
    )
    ax2 = None

    all_timepoints = set()
    if has_df:
        x_col = spider_config.x
        df_cats = df[x_col].cat.remove_unused_categories().cat.categories
        all_timepoints |= set(df_cats)
    if has_scan:
        scan_x_col = scan_overlay_config.x
        scan_cats = scan_data[scan_x_col].cat.remove_unused_categories().cat.categories
        all_timepoints |= set(scan_cats)
    all_timepoints = sorted(all_timepoints)
    cat_to_pos = {cat: i for i, cat in enumerate(all_timepoints)}
    ax.set_xlim(0, len(all_timepoints) - 0.5)

    if has_df:
        df[x_col] = pd.Categorical(df[x_col], categories=all_timepoints, ordered=True)
        if not spider_config.title:
            ref_row = df[spider_config.col_vals_to_include_in_title].iloc[0]
            spider_config.title = " | ".join(
                ", ".join(map(str, val)) if isinstance(val, list) else str(val)
                for val in ref_row.to_dict().values()
            )

    if has_scan:
        scan_data[scan_x_col] = pd.Categorical(
            scan_data[scan_x_col], categories=all_timepoints, ordered=True
        )

    default_ylim = 105
    use_absolute_scale_main = (
        has_df and hasattr(spider_config, "use_absolute_scale") and spider_config.use_absolute_scale
    )
    y1_max = abs(df[spider_config.y].max()) if has_df else 0
    y2_max = abs(scan_data[scan_overlay_config.y].max()) if has_scan else 0
    y2_min = scan_data[scan_overlay_config.y].min() if has_scan else 0
    if use_absolute_scale_main:
        combined_max = y1_max
    else:
        scan_extreme = max(abs(y2_max), abs(y2_min)) if has_scan else 0
        combined_max = max(y1_max, scan_extreme)

    if use_absolute_scale_main:
        base_ylim = spider_config.absolute_ylim or (-5, 105)
        base_yticks = spider_config.absolute_yticks or [0, 25, 50, 75, 100]
        ylim = base_ylim
        yticks = base_yticks
        new_ylim = ylim[1]
        spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 25
        ax.set_ylim(ylim[0], ylim[1])
        ymax = combined_max
    else:
        ymax = combined_max
        if ymax > default_ylim:
            yticks = np.arange(-default_ylim + 5, ymax + 25, 25)
            spacing = yticks[1] - yticks[0] if len(yticks) >= 2 else 5
            new_ylim = math.ceil(ymax / spacing) * spacing
            ax.set_ylim(-default_ylim, new_ylim)
            yticks = ax.get_yticks()
            spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 1
            _, new_ylim = ax.get_ylim()
        else:
            new_ylim = default_ylim
            spacing = 25
            ax.set_ylim(-default_ylim, new_ylim)
            yticks = np.arange(-100, 101, spacing)
            ax.set_yticks(yticks)

    text_y = new_ylim - 0.4 * spacing

    if has_df and not has_scan:
        generate_styled_spiderplot(df=df, config=spider_config, ax=ax)
        ax.set_ylabel(spider_config.ylabel or r"$\Delta$ from First Timepoint", fontweight="bold")
        adjust_legend(ax, (1.2, 0.7), redraw=True)
        ax.set_facecolor("white")
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(right=spider_config.rhs_pdf_padding)
        xticks = ax.get_xticks()
        if len(xticks) > 8:
            ax.tick_params(axis="x", labelsize=13, rotation=45)
        else:
            ax.tick_params(axis="x", labelsize=13)
        return fig

    if has_scan and not has_df:
        recists = (
            scan_data[[scan_x_col, scan_overlay_config.recist_col]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for _, row in recists.iterrows():
            x = cat_to_pos[row[scan_x_col]]
            text = row[scan_overlay_config.recist_col]
            ax.text(
                x + 0.03,
                text_y,
                text,
                fontsize=14,
                fontweight="bold",
                color=(recist_color_dict.get(text, "black") if recist_color_dict else "black"),
                zorder=1,
            )
            ax.axvline(x=x, color="gainsboro", linestyle="--", linewidth=1, zorder=1)
        sns.lineplot(
            data=scan_data,
            x=scan_x_col,
            y=scan_overlay_config.y,
            hue=scan_overlay_config.hue_col,
            palette=scan_overlay_config.palette,
            ax=ax,
            linewidth=scan_overlay_config.lw / 2,
            linestyle=scan_overlay_config.linestyle,
            alpha=scan_overlay_config.alpha,
            zorder=2,
        )
        sns.scatterplot(
            data=scan_data,
            x=scan_x_col,
            y=scan_overlay_config.y,
            hue=scan_overlay_config.hue_col,
            palette=scan_overlay_config.palette,
            ax=ax,
            s=(scan_overlay_config.markersize / 1.5) ** 2,
            edgecolor="white",
            linewidth=1,
            legend=False,
            zorder=2,
        )
        texts, x_pos_, y_pos_ = [], [], []
        for label, group in scan_data.groupby(scan_overlay_config.hue_col):
            last = group.dropna(subset=[scan_x_col, scan_overlay_config.y]).iloc[-1]
            x = cat_to_pos[last[scan_x_col]] + np.random.normal(0, 0.01)
            y = last[scan_overlay_config.y] + 2 + np.random.normal(0, 0.01)
            x_pos_.append(x)
            y_pos_.append(y)
            texts.append(
                ax.text(
                    x + 0.05,
                    y + 0.3,
                    label,
                    fontdict={"family": DefaultStyle().font_family, "size": 10, "weight": "bold"},
                    color=scan_overlay_config.palette.get(label, "black"),
                    ha="left",
                    va="center",
                    zorder=10,
                )
            )
        plt.draw()
        adjust_text(
            texts,
            ax=ax,
            force_text=(1, 3),
            force_points=(0.5, 1),
            points=list(zip(x_pos_, y_pos_)),
            expand_text=(1.2, 1.2),
            verbose=0,
        )
        ax.set_title(scan_overlay_config.title, loc="left", fontweight="bold")
        ax.set_ylabel(
            r"Diameter %$\Delta$ from First Timepoint",
            fontdict={
                "family": DefaultStyle().font_family,
                "size": ax.yaxis.label.get_fontsize(),
                "weight": "bold",
            },
        )
        handles, labels = ax.get_legend_handles_labels()
        section_label = Line2D([0], [0], color="none", label="Location")
        for h in handles:
            if hasattr(h, "set_alpha"):
                h.set_alpha(1)
        ax.legend(
            handles=[section_label] + handles,
            labels=["Location"] + labels,
            frameon=False,
            prop=fm.FontProperties(family=DefaultStyle().font_family, size=14, weight="bold"),
        )
        adjust_legend(ax, (1.2, 0.7))
        fig.subplots_adjust(right=spider_config.rhs_pdf_padding if spider_config else 0.85)
        ax.set_facecolor("white")
        fig.patch.set_alpha(0.0)
        ax.set_xlabel(scan_overlay_config.x, fontweight="bold")
        xticks = ax.get_xticks()
        if len(xticks) > 8:
            ax.tick_params(axis="x", labelsize=13, rotation=45)
        else:
            ax.tick_params(axis="x", labelsize=13)
        return fig

    generate_styled_spiderplot(df=df, config=spider_config, ax=ax)
    ax2 = ax.twinx()
    if ax2:
        ax2.set_xlim(0, len(all_timepoints) - 0.5)
    recists = (
        scan_data[[scan_x_col, scan_overlay_config.recist_col]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    for _, row in recists.iterrows():
        x = cat_to_pos[row[scan_x_col]]
        text = row[scan_overlay_config.recist_col]
        ax.text(
            x + 0.03,
            text_y,
            text,
            fontsize=14,
            fontweight="bold",
            color=(recist_color_dict.get(text, "black") if recist_color_dict else "black"),
            zorder=1,
        )
        ax.axvline(x=x, color="gainsboro", linestyle="--", linewidth=1, zorder=1)
    sns.lineplot(
        data=scan_data,
        x=scan_x_col,
        y=scan_overlay_config.y,
        hue=scan_overlay_config.hue_col,
        palette=scan_overlay_config.palette,
        ax=ax2,
        linewidth=scan_overlay_config.lw / 2,
        linestyle=scan_overlay_config.linestyle,
        alpha=scan_overlay_config.alpha,
        zorder=2,
    )
    sns.scatterplot(
        data=scan_data,
        x=scan_x_col,
        y=scan_overlay_config.y,
        hue=scan_overlay_config.hue_col,
        palette=scan_overlay_config.palette,
        ax=ax2,
        s=(scan_overlay_config.markersize / 1.5) ** 2,
        edgecolor="white",
        linewidth=1,
        legend=False,
        zorder=2,
    )
    texts, x_pos_, y_pos_ = [], [], []
    for label, group in scan_data.groupby(scan_overlay_config.hue_col):
        last = group.dropna(subset=[scan_x_col, scan_overlay_config.y]).iloc[-1]
        x = cat_to_pos[last[scan_x_col]] + np.random.normal(0, 0.01)
        y = last[scan_overlay_config.y] + 2 + np.random.normal(0, 0.01)
        x_pos_.append(x)
        y_pos_.append(y)
        texts.append(
            ax2.text(
                x + 0.05,
                y + 0.3,
                label,
                fontdict={"family": DefaultStyle().font_family, "size": 10, "weight": "bold"},
                color=scan_overlay_config.palette.get(label, "black"),
                ha="left",
                va="center",
                zorder=10,
            )
        )
    plt.draw()
    if use_absolute_scale_main:
        if has_scan and ax2:
            scan_extreme = max(abs(y2_max), abs(y2_min)) if has_scan else 0
            base_scan_limit = 100
            if scan_extreme > base_scan_limit:
                tick_spacing = 25
                extended_limit = math.ceil(scan_extreme / tick_spacing) * tick_spacing
                scan_ylim = extended_limit
            else:
                scan_ylim = base_scan_limit
            ax2.set_ylim(-scan_ylim - 5, scan_ylim + 5)
            tick_spacing = 25
            scan_ticks = list(range(-int(scan_ylim), int(scan_ylim) + 1, tick_spacing))
            ax2.set_yticks(scan_ticks)
        elif ax2:
            ax2.set_ylim(-new_ylim, new_ylim)
    else:
        ax.set_ylim(-new_ylim, new_ylim)
        if has_scan and ax2:
            ax2.set_ylim(-new_ylim, new_ylim)
            main_ticks = ax.get_yticks()
            ax2.set_yticks(main_ticks)
        elif ax2:
            ax2.set_ylim(-new_ylim, new_ylim)
            main_ticks = ax.get_yticks()
            ax2.set_yticks(main_ticks)
    adjust_text(
        texts,
        ax=ax2,
        force_text=(1, 3),
        force_points=(0.5, 1),
        points=list(zip(x_pos_, y_pos_)),
        expand_text=(1.2, 1.2),
        verbose=0,
    )
    ax2.set_ylabel(
        r"Diameter %$\Delta$ from First Timepoint",
        fontdict={
            "family": DefaultStyle().font_family,
            "size": ax.yaxis.label.get_fontsize(),
            "weight": "bold",
        },
    )
    handles, labels = ax2.get_legend_handles_labels()
    section_label = Line2D([0], [0], color="none", label="Location")
    for h in handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
    ax2.legend(
        handles=[section_label] + handles,
        labels=["Location"] + labels,
        frameon=False,
        prop=fm.FontProperties(family=DefaultStyle().font_family, size=14, weight="bold"),
    )
    adjust_legend(ax2, (1.2, 0.3))
    adjust_legend(ax, (1.2, 0.8), redraw=True)
    fig.subplots_adjust(right=spider_config.rhs_pdf_padding)
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0)
    xticks = ax.get_xticks()
    if len(xticks) > 8:
        ax.tick_params(axis="x", labelsize=13, rotation=45)
    else:
        ax.tick_params(axis="x", labelsize=13)
    return fig
