"""Simple waterfall plot implementation."""

from typing import Optional, Tuple, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_composite_helpers import get_default_palette


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

    Args:
        df: DataFrame with values
        value_col: column name for values (numeric)
        id_col: optional id column to label bars
        color_col: optional column for colors (mapped to categorical palette)
        figsize: figure size
        ax: optional axes
        show: whether to call plt.show()

    Returns:
        (fig, ax)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.0)
        except Exception:
            pass
        created_fig = True
    else:
        fig = ax.figure

    # If aggregation requested, compute group-level values
    if group_col is not None and aggregate is not None:
        # allow aggregate to be a string like 'mean' or 'sum'
        if isinstance(aggregate, str):
            agg_func = aggregate
        else:
            agg_func = aggregate
        agg_df = (
            df.groupby(group_col)[value_col]
            .agg(agg_func)
            .reset_index()
            .rename(columns={value_col: "agg_value"})
        )
        plot_df = agg_df.sort_values(by="agg_value", ascending=False).reset_index(drop=True)
        x = np.arange(len(plot_df))
        values = plot_df["agg_value"].values
        if color_col and color_col in plot_df.columns:
            groups = pd.Categorical(plot_df[color_col])
            pal = get_default_palette(len(groups.categories))
            colors = [pal[groups.codes[i] % len(pal)] for i in range(len(groups))]
        else:
            colors = get_default_palette(1)[0]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df[group_col].astype(str), rotation=90)
        agg_label = (
            aggregate if isinstance(aggregate, str) else getattr(aggregate, "__name__", "agg")
        )
        ax.set_ylabel(f"{value_col} ({agg_label})")
        ax.set_xlabel(group_col)
        if created_fig and show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    # Facet mode: create small multiples per facet value
    if facet_by is not None and facet_by in df.columns:
        facet_vals = list(pd.Categorical(df[facet_by]).categories)
        n = len(facet_vals)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        try:
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.0)
        except Exception:
            pass
        axes = axes.flatten()
        for i, fv in enumerate(facet_vals):
            sub = df[df[facet_by] == fv].sort_values(by=value_col, ascending=False)
            x = np.arange(len(sub))
            vals = sub[value_col].values
            if color_col and color_col in sub.columns:
                groups = pd.Categorical(sub[color_col])
                pal = get_default_palette(len(groups.categories))
                colors = [pal[groups.codes[j] % len(pal)] for j in range(len(groups))]
            else:
                colors = get_default_palette(1)[0]
            ax_i = axes[i]
            ax_i.bar(x, vals, color=colors)
            ax_i.set_title(str(fv))
            ax_i.set_xticks([])
        # hide unused axes
        for j in range(len(facet_vals), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes[0]

    # Default per-sample waterfall
    df_sorted = df.sort_values(by=value_col, ascending=False).reset_index(drop=True)
    x = np.arange(len(df_sorted))
    values = df_sorted[value_col].values

    if color_col and color_col in df_sorted.columns:
        groups = pd.Categorical(df_sorted[color_col])
        pal = get_default_palette(len(groups.categories))
        colors = [pal[groups.codes[i] % len(pal)] for i in range(len(groups))]
    else:
        colors = get_default_palette(1)[0]

    ax.bar(x, values, color=colors)
    if id_col and id_col in df_sorted.columns:
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted[id_col].astype(str), rotation=90)
    else:
        ax.set_xticks([])

    ax.set_ylabel(value_col)
    ax.set_xlabel(id_col or "Index")
    if created_fig and show:
        plt.tight_layout()
        plt.show()
    return fig, ax


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
        fig.patch.set_facecolor('white')
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
