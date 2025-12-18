"""
Line plot utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

# %%
import math
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
from adjustText import adjust_text
from matplotlib import font_manager
from matplotlib.lines import Line2D

from bioviz.configs import LinePlotConfig
from bioviz.utils.plotting import adjust_legend, resolve_font_family


# Expose public function
__all__ = [
    "generate_lineplot",
    "generate_styled_lineplot",
    "generate_styled_multigroup_lineplot",
    "generate_lineplot_twinx",
]


class LinePlotter:
    """
    Stateful wrapper for line plots.

    Construct with `(df, config)` where `config` is a `LinePlotConfig` or
    dict acceptable to it. Delegates rendering to the canonical
    `generate_lineplot`/`generate_styled_lineplot` functions.
    """

    def __init__(self, df: pd.DataFrame, config: LinePlotConfig | dict):
        if isinstance(config, dict):
            config = LinePlotConfig(**config)
        self.df = df.copy()
        self.config = config
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.annotation_history: List[Dict] = []

    def set_data(self, df: pd.DataFrame) -> "LinePlotter":
        self.df = df.copy()
        return self

    def update_config(self, **kwargs) -> "LinePlotter":
        for k, v in kwargs.items():
            try:
                setattr(self.config, k, v)
            except Exception:
                continue
        return self

    def plot(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        twinx_data: Optional[pd.DataFrame] = None,
        secondary_config: Optional[LinePlotConfig | dict] = None,
        annotation_color_dict: Optional[dict] = None,
        annotation_source: str = "auto",
        draw_legend: bool = True,
    ):
        """Render the line plot and store `fig, ax` on the instance.

        This dispatcher supports:
        - Single-/multi-group plots via `generate_lineplot` (the default).
        - Twin-axis overlay plots via `generate_lineplot_twinx` when `twinx_data`
          or `secondary_config` is provided.

        Note: `generate_lineplot` already inspects the provided `LinePlotConfig`
        to choose between single-series and multi-group variants (based on
        `group_col` / `label_col`). The twin-axis case requires explicit
        `twinx_data` (or a `secondary_config` describing the overlay) and is
        dispatched here.
        """

        if df is not None:
            self.set_data(df)
        if self.df is None or self.config is None:
            raise RuntimeError(
                "Both dataframe and config are required; construct with (df, config)"
            )

        # If twin-axis data or a secondary config is provided, delegate to the
        # twinx generator which handles combined/secondary-only/primary-only modes.
        if twinx_data is not None or secondary_config is not None:
            sec_cfg = None
            if isinstance(secondary_config, dict):
                sec_cfg = LinePlotConfig(**secondary_config)
            elif isinstance(secondary_config, LinePlotConfig):
                sec_cfg = secondary_config

            self.fig = generate_lineplot_twinx(
                df=self.df,
                twinx_data=twinx_data,
                primary_config=self.config,
                secondary_config=sec_cfg,
                annotation_color_dict=annotation_color_dict,
                annotation_source=annotation_source,
            )
        else:
            # default single/multi-group dispatcher
            self.fig = generate_lineplot(self.df, self.config, ax=None, draw_legend=draw_legend)

        # ensure ax is available when the generator returns a fig
        if self.fig is None:
            self.ax = None
        else:
            try:
                self.ax = self.fig.axes[0] if self.fig.axes else None
            except Exception:
                self.ax = None
        return self.fig, self.ax

    def save(self, path: str, **save_kwargs) -> None:
        from pathlib import Path

        if self.fig is None:
            raise RuntimeError("No figure available; call .plot() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, **save_kwargs)

    def annotate(self, *args, **kwargs) -> "LinePlotter":
        # placeholder for future annotation methods; record call
        self.annotation_history.append({"args": args, "kwargs": kwargs})
        return self

    def close(self) -> None:
        try:
            if self.fig is not None:
                plt.close(self.fig)
        finally:
            self.fig = None
            self.ax = None


def generate_lineplot(
    df: pd.DataFrame,
    config: LinePlotConfig,
    ax: plt.Axes | None = None,
    draw_legend: bool = True,
) -> plt.Figure | None:
    """Unified entry point; dispatches to single- or multi-group plots based on config.

    - If `group_col` is set, renders multi-group trajectories.
    - Else if `label_col` is set, renders a single-series plot with hue.
    - If both are set, multi-group takes precedence.
    """

    has_group = bool(getattr(config, "group_col", None))
    has_label = bool(getattr(config, "label_col", None))

    if has_group:
        fig, *_ = generate_styled_multigroup_lineplot(
            df=df, config=config, ax=ax, draw_legend=draw_legend
        )
        try:
            face = getattr(config, 'figure_facecolor', None) or 'white'
            transparent = getattr(config, 'figure_transparent', False)
            fig.patch.set_facecolor(face)
            fig.patch.set_alpha(0.0 if transparent else 1.0)
        except Exception:
            pass
        return fig

    if has_label:
        fig = generate_styled_lineplot(df=df, config=config, ax=ax)
        try:
            face = getattr(config, 'figure_facecolor', None) or 'white'
            transparent = getattr(config, 'figure_transparent', False)
            if fig is not None:
                fig.patch.set_facecolor(face)
                fig.patch.set_alpha(0.0 if transparent else 1.0)
        except Exception:
            pass
        return fig

    raise ValueError("LinePlotConfig must set either group_col or label_col for line plotting.")


def generate_styled_lineplot(
    df: pd.DataFrame,
    config: LinePlotConfig,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if not config.label_col:
        raise ValueError("line plot requires label_col (hue) to be set in LinePlotConfig.")
    if not config.x or not config.y:
        raise ValueError("line plot requires x and y to be set in LinePlotConfig.")
    entity_id = getattr(config, "entity_id", None) or getattr(config, "patient_id", None)
    if df.empty:
        print(f"No data for id '{entity_id}': DataFrame is empty.")
        return None

    if config.label_col not in df:
        print(f"No data for id '{entity_id}': column '{config.label_col}' does not exist.")
        return None

    if df[config.label_col].dropna().empty:
        print(
            f"No data for id '{entity_id}': column '{config.label_col}' contains only missing values."
        )
        return None

    if config.y not in df:
        print(f"No data for id '{entity_id}': column '{config.y}' does not exist.")
        return None

    if df[config.y].dropna().empty:
        print(f"No data for id '{entity_id}': column '{config.y}' contains only missing values.")
        return None

    # Ensure long-format required columns exist; bioviz expects callers to
    # supply long-format data. Forward-fill (if desired) should be done by
    # adapters (e.g. tm_toolbox) before calling bioviz.
    required_cols = [
        c for c in (config.x, config.y, config.label_col, config.secondary_group_col) if c
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns for long-format plotting: {missing}"
        )

    df = df.dropna(subset=[config.y]).copy()

    # Ensure x is categorical; if not, coerce using appearance order to keep caller intent.
    if config.x in df and not pd.api.types.is_categorical_dtype(df[config.x]):
        x_dtype = CategoricalDtype(categories=list(pd.unique(df[config.x])), ordered=True)
        df[config.x] = df[config.x].astype(x_dtype)
    elif config.x in df and hasattr(df[config.x].dtype, "categories"):
        try:
            df[config.x] = df[config.x].cat.remove_unused_categories()
        except Exception:
            pass

    # Map categorical Timepoint to numeric position for x-axis plotting
    cat_to_pos = (
        {cat: i for i, cat in enumerate(df[config.x].cat.categories)}
        if config.x in df and hasattr(df[config.x].dtype, "categories")
        else {}
    )

    if not config.title:
        existing_cols = [col for col in config.col_vals_to_include_in_title if col in df.columns]
        if existing_cols:
            records = df.loc[:, existing_cols].to_dict(orient="records")
            ref = records[0] if records else {}
        else:
            ref = {}
        title = " | ".join(
            ", ".join(map(str, v)) if isinstance(v, list) else str(v) for v in ref.values()
        )
    else:
        title = config.title

    labels = sorted(df[config.label_col].unique())
    num_labels = len(labels)

    palette = config.palette
    if palette is None:
        palette = sns.color_palette("Dark2", n_colors=num_labels)

    if isinstance(palette, dict):
        color_dict = palette
    else:
        if len(palette) < num_labels:
            raise ValueError("Palette has fewer colors than the number of unique labels.")
        color_dict = dict(zip(labels, palette))

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
        try:
            face = getattr(config, 'figure_facecolor', None) or 'white'
            transparent = getattr(config, 'figure_transparent', False)
            fig.patch.set_facecolor(face)
            fig.patch.set_alpha(0.0 if transparent else 1.0)
        except Exception:
            pass
    else:
        fig = ax.figure

    sns.lineplot(
        data=df,
        x=config.x,
        y=config.y,
        hue=config.label_col,
        marker=None,
        lw=config.lw,
        alpha=0.5,
        palette=color_dict,
        ax=ax,
        zorder=5,
    )

    for label in df[config.label_col].unique():
        subset = df[df[config.label_col] == label].dropna(subset=[config.x, config.y])
        if config.threshold:
            above_thresh = subset[subset[config.y] > config.threshold]
            below_thresh = subset[subset[config.y] <= config.threshold]
            ax.scatter(
                x=above_thresh[config.x].cat.codes,
                y=above_thresh[config.y],
                color=color_dict[label],
                edgecolor="white",
                linewidth=1,
                s=(config.markersize * config.filled_marker_scale) ** 2,
                zorder=6,
            )
            ax.scatter(
                x=below_thresh[config.x].cat.codes,
                y=below_thresh[config.y],
                facecolors="white",
                edgecolors=color_dict[label],
                linewidth=1.5,
                s=config.markersize**2,
                zorder=7,
            )
        else:
            ax.scatter(
                x=subset[config.x].cat.codes,
                y=subset[config.y],
                color=color_dict[label],
                edgecolor="white",
                linewidth=1,
                s=(config.markersize * config.filled_marker_scale) ** 2,
                zorder=6,
            )

    # Y-limits and title/labels
    ymin_data = df[config.y].min()
    ymax_data = df[config.y].max()
    yrange = ymax_data - ymin_data if ymax_data > ymin_data else 1.0
    ymin_padded = ymin_data - config.ymin_padding_fraction * yrange
    if config.add_extra_tick_to_ylim:
        yticks = ax.get_yticks()
        tick_spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 1.0
        ymax_padded = ymax_data + tick_spacing
    else:
        ymax_padded = ymax_data

    ylim = (ymin_padded, ymax_padded)

    # Honor user-specified ylim if provided; use computed value when bound is None
    if getattr(config, "ylim", None) is not None:
        y0, y1 = config.ylim
        if y0 is not None or y1 is not None:
            ymin = y0 if y0 is not None else ymin_padded
            ymax = y1 if y1 is not None else ymax_padded
            ylim = (ymin, ymax)

    if cat_to_pos:
        xmax = max(cat_to_pos.values())
    else:
        xmax = 0

    if getattr(config, "align_first_tick_to_origin", False):
        xlim_default = (0, xmax + config.xlim_padding)
    else:
        xlim_default = (-1 * config.xlim_padding, xmax + config.xlim_padding)

    # Honor user-specified xlim if provided; otherwise use padding-based default.
    xlim = xlim_default
    if getattr(config, "xlim", None) is not None:
        xl0, xl1 = config.xlim
        if xl0 is not None or xl1 is not None:
            left = xl0 if xl0 is not None else xlim_default[0]
            right = xl1 if xl1 is not None else xlim_default[1]
            xlim = (left, right)

    if config.threshold is not None:
        thresh_kwargs = dict(
            y=config.threshold,
            color=getattr(config, "threshold_color", "#C0C0C0"),
            linestyle=getattr(config, "threshold_style", "--"),
            linewidth=getattr(config, "threshold_width", 1.0),
            alpha=getattr(config, "threshold_alpha", 1.0),
            zorder=0,
        )
        tdashes = getattr(config, "threshold_dashes", (5, 5))
        if tdashes is not None:
            thresh_kwargs["dashes"] = tdashes
        ax.axhline(**thresh_kwargs)
        yticks = ax.get_yticks()
        spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 1
        label_text = getattr(config, "threshold_label", None)
        if label_text:
            text_y = config.threshold + 0.1 * spacing
            # Place near the right edge of the plotting range for better stability with custom xlim
            x_span = xlim[1] - xlim[0]
            text_x = xlim[1] - 0.02 * x_span
            label_color = getattr(config, "threshold_label_color", None) or getattr(
                config, "threshold_color", "#C0C0C0"
            )
            ax.text(
                text_x,
                text_y,
                label_text,
                fontsize=getattr(config, "threshold_label_fontsize", 14),
                fontweight="normal",
                color=label_color,
                alpha=getattr(config, "threshold_label_alpha", 1.0),
                zorder=0,
                ha="right",
            )

    xlabel = config.xlabel if config.xlabel is not None else config.x
    ylabel = config.ylabel if config.ylabel is not None else config.y
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

    # Override fontsizes if provided
    if getattr(config, "xlabel_fontsize", None) is not None:
        ax.xaxis.label.set_size(config.xlabel_fontsize)
    if getattr(config, "ylabel_fontsize", None) is not None:
        ax.yaxis.label.set_size(config.ylabel_fontsize)
    tick_kwargs = {}
    if getattr(config, "xtick_fontsize", None) is not None:
        tick_kwargs["labelsize"] = config.xtick_fontsize
    if tick_kwargs:
        ax.tick_params(axis="x", **tick_kwargs)
    tick_kwargs = {}
    if getattr(config, "ytick_fontsize", None) is not None:
        tick_kwargs["labelsize"] = config.ytick_fontsize
    if tick_kwargs:
        ax.tick_params(axis="y", **tick_kwargs)
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        fontsize=getattr(config, "title_fontsize", 20),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if config.label_points:
        df = df.dropna(subset=[config.x, config.y])
        first_points = (
            df.groupby(config.label_col)
            .apply(lambda df: df.sort_values(config.x).head(1))
            .reset_index(drop=True)
        )
        texts = []
        x_pos_ = []
        y_pos_ = []
        for _, row in first_points.iterrows():
            x_pos = cat_to_pos[row[config.x]]
            y_pos = row[config.y]
            jitter_x = np.random.normal(loc=0, scale=0.01)
            jitter_y = np.random.normal(loc=0, scale=0.01)
            x_pos += jitter_x
            y_pos += jitter_y
            x_pos_.append(x_pos)
            y_pos_.append(y_pos)
            texts.append(
                ax.text(
                    x_pos - 0.08,
                    y_pos,
                    row[config.label_col],
                    fontsize=14,
                    fontweight="bold",
                    color=color_dict[row[config.label_col]],
                    ha="right",
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

    handles = [
        Line2D(
            [0],
            [0],
            linewidth=0,
            label=(config.label_col or "Value"),
            color="black",
        )
    ] + [
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
        for label in df[config.label_col].unique()
    ]

    detection_handles = []
    if config.threshold:
        detection_handles.extend(
            [
                Line2D([0], [0], linewidth=0, label="", color="black"),
                Line2D(
                    [0],
                    [0],
                    linewidth=0,
                    label=config.threshold_legend_title or "Threshold",
                    color="black",
                    markerfacecolor="black",
                    markeredgecolor="black",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="white",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=config.markersize,
                    linewidth=0,
                    label=config.threshold_below_label or "Below Threshold",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="white",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=config.markersize,
                    linewidth=0,
                    label=config.threshold_above_label or "Above Threshold",
                ),
            ]
        )

    # Use caller/applied rcParams for legend font; fall back to matplotlib default
    legend_family = resolve_font_family()

    lgd = ax.legend(
        handles=handles + detection_handles,
        bbox_to_anchor=(1.25, 0.5),
        loc="center",
        frameon=False,
        prop=font_manager.FontProperties(family=legend_family, size=14, weight="bold"),
    )

    if config.match_legend_text_color:
        for text in lgd.get_texts():
            label = text.get_text()
            if label in color_dict:
                text.set_color(color_dict[label])
                text.set_fontweight("bold")
            elif label in {
                config.label_col,
                config.threshold_legend_title or "Threshold",
            }:
                text.set_color("black")
                text.set_fontweight("bold")
                text.set_fontsize(getattr(config, "legend_fontsize", 16))

    plt.subplots_adjust(right=config.rhs_pdf_padding)
    ax.set_facecolor("white")

    # Figure background is configurable; default to opaque white for exports. Use getattr
    # for backward compatibility with configs created before the new fields existed.
    face_cfg = getattr(config, "figure_facecolor", None)
    transparent_cfg = getattr(config, "figure_transparent", False)
    face = face_cfg if face_cfg is not None else "white"
    fig.patch.set_facecolor(face)
    fig.patch.set_alpha(0.0 if transparent_cfg else 1.0)

    return fig


def generate_styled_multigroup_lineplot(
    df: pd.DataFrame,
    config: LinePlotConfig,
    ax: plt.Axes | None = None,
    draw_legend: bool = True,
) -> tuple[plt.Figure, list[Line2D], list[str]]:
    """
    Render multiple group trajectories (one line per group) with consistent styling.

    Args:
       df: Long-format DataFrame containing group, x, and y columns.
       config: `LinePlotConfig` controlling styling, titles, markers, and legend behavior.
       ax: Optional matplotlib Axes to draw into; if omitted a new figure/axes are created.
       draw_legend: Whether to draw the plot legend.

    Returns:
       Tuple of `(figure, handles, labels)` where `handles` are legend handles and
       `labels` are their corresponding labels.
    """

    if not config.group_col:
        raise ValueError("multigroup line plot requires group_col to be set in LinePlotConfig.")
    if not config.x or not config.y:
        raise ValueError("multigroup line plot requires x and y to be set in LinePlotConfig.")
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    if config.x in df and not pd.api.types.is_categorical_dtype(df[config.x]):
        x_dtype = CategoricalDtype(categories=list(pd.unique(df[config.x])), ordered=True)
        df[config.x] = df[config.x].astype(x_dtype)
    elif config.x in df and hasattr(df[config.x].dtype, "categories"):
        try:
            df[config.x] = df[config.x].cat.remove_unused_categories()
        except Exception:
            pass

    required_cols = [config.group_col, config.x, config.y]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns for multigroup line plot: {missing}"
        )

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
    xpad = getattr(config, "xlim_padding", 0.8)
    align_origin = getattr(config, "align_first_tick_to_origin", False)
    if cat_to_pos:
        xmax = max(cat_to_pos.values())
        x_start = 0 if align_origin else -1 * xpad
        x_end = xmax + xpad
        ax.set_xlim(x_start, x_end)
        ax.set_xticks(list(cat_to_pos.values()))
        ax.set_xticklabels(list(cat_to_pos.keys()))
    else:
        ax.set_xlim(-1 * xpad, xpad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(config.xlabel or config.x, fontweight="bold")
    ax.set_ylabel(config.ylabel or str(config.y), fontweight="bold")
    ax.set_title(
        config.title or "Change over Time",
        loc="left",
        fontweight="bold",
        fontsize=getattr(config, "title_fontsize", 20),
    )

    if getattr(config, "xlabel_fontsize", None) is not None:
        ax.xaxis.label.set_size(config.xlabel_fontsize)
    if getattr(config, "ylabel_fontsize", None) is not None:
        ax.yaxis.label.set_size(config.ylabel_fontsize)
    tick_kwargs = {}
    if getattr(config, "xtick_fontsize", None) is not None:
        tick_kwargs["labelsize"] = config.xtick_fontsize
    if tick_kwargs:
        ax.tick_params(axis="x", **tick_kwargs)
    tick_kwargs = {}
    if getattr(config, "ytick_fontsize", None) is not None:
        tick_kwargs["labelsize"] = config.ytick_fontsize
    if tick_kwargs:
        ax.tick_params(axis="y", **tick_kwargs)

    ref_val = getattr(config, "reference", None)
    if ref_val is not None:
        reference_kwargs = dict(
            y=ref_val,
            color=getattr(config, "reference_color", "#C0C0C0"),
            linestyle=getattr(config, "reference_style", "--"),
            linewidth=getattr(config, "reference_width", 1.0),
            alpha=getattr(config, "reference_alpha", 1.0),
            zorder=-1,
        )
        rdashes = getattr(config, "reference_dashes", (5, 5))
        if rdashes is not None:
            reference_kwargs["dashes"] = rdashes
        ax.axhline(**reference_kwargs)

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
                    [0],
                    [0],
                    color="black",
                    marker=markerstyle,
                    linewidth=0,
                    label=label,
                    alpha=1,
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
            prop=font_manager.FontProperties(family=resolve_font_family(), size=14, weight="bold"),
        )
    else:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    face_cfg = getattr(config, "figure_facecolor", None)
    transparent_cfg = getattr(config, "figure_transparent", False)
    face = face_cfg if face_cfg is not None else "white"
    fig.patch.set_facecolor(face)
    fig.patch.set_alpha(0.0 if transparent_cfg else 1.0)

    return fig, handles, labels


def generate_lineplot_twinx(
    df: pd.DataFrame | None,
    twinx_data: pd.DataFrame | None,
    primary_config: LinePlotConfig | None = None,
    secondary_config: LinePlotConfig | None = None,
    annotation_color_dict: dict[str, str] | None = None,
    annotation_source: str = "auto",
) -> plt.Figure:
    """
    Generate a plot with an optional secondary (twinx) series overlaid.

    This function supports three modes:
    - Primary-only: `df` provided, `twinx_data` omitted — renders a standard multi-group plot.
    - Twinx-only: `twinx_data` provided, `df` omitted — renders the secondary series with annotations.
    - Combined: both `df` and `twinx_data` provided — renders both series on shared x-axis.

    Args:
        df: Primary (main) DataFrame for left axis plotting; may be None when only secondary is desired.
        twinx_data: DataFrame for secondary axis plotting (overlay); may be None.
        primary_config: `LinePlotConfig` for the primary axis (required when `df` is provided).
        secondary_config: `LinePlotConfig` or overlay config for the secondary axis.
        annotation_color_dict: Optional mapping of annotation value -> color for overlay text.
        annotation_source: 'auto'|'primary'|'secondary' to select which config supplies overlay fields.

    Returns:
        A matplotlib `Figure` containing the rendered plot.
    """

    ann_cfg = secondary_config or primary_config

    source_preference = (annotation_source or "auto").lower()
    # Backward-compat aliases: primary/left -> primary axis (main), secondary/right -> secondary axis (twin).
    if source_preference in {"primary", "left"}:
        source_preference = "primary"
    elif source_preference in {"secondary", "right"}:
        source_preference = "secondary"
    if source_preference not in {"auto", "primary", "secondary"}:
        source_preference = "auto"

    has_df = df is not None and not df.empty
    has_twinx = twinx_data is not None and not twinx_data.empty

    # Resolve secondary fields from standard axes/hue styling on the secondary config.
    secondary_y = getattr(ann_cfg, "y", None) if ann_cfg else None
    secondary_hue = None
    if ann_cfg:
        secondary_hue = getattr(ann_cfg, "label_col", None) or getattr(ann_cfg, "group_col", None)
    overlay_palette = getattr(ann_cfg, "palette", None) if ann_cfg else None
    secondary_linestyle = getattr(ann_cfg, "linestyle", None) if ann_cfg else None
    if secondary_linestyle is None:
        secondary_linestyle = ":"

    # Resolve secondary annotation field once so both branches (primary-only or twinx-only)
    # can reuse it without hitting UnboundLocalError when df is missing.
    primary_ann_col = getattr(primary_config, "overlay_col", None) if primary_config else None
    secondary_ann_col = getattr(ann_cfg, "overlay_col", None)
    if primary_ann_col and secondary_ann_col and primary_ann_col != secondary_ann_col:
        print(
            f"overlay_col differs between configs ({primary_ann_col} vs {secondary_ann_col}); using primary_config value."
        )
    annotation_field = primary_ann_col or secondary_ann_col

    # Allow using the same DataFrame for twin-axis secondary by reusing df when no separate
    # twinx_data is provided but twin fields exist on the secondary_config.
    if not has_twinx and has_df and secondary_config is not None:
        if (
            secondary_config.x
            and secondary_config.y
            and (secondary_config.label_col or secondary_config.group_col)
        ):
            twinx_data = df
            has_twinx = True
    if not has_df and not has_twinx:
        raise ValueError("At least one of df or twinx_data must be provided and non-empty.")

    if has_df and primary_config is None:
        raise ValueError("primary_config is required when df is provided.")
    if has_twinx and ann_cfg is None:
        raise ValueError(
            "Provide a LinePlotConfig with overlay fields when twinx_data is provided."
        )
    if has_twinx:
        if not ann_cfg.x or not secondary_y or not secondary_hue:
            raise ValueError(
                "Secondary config must define x, y, and a hue (label_col or group_col)."
            )

    fig, ax = plt.subplots(
        figsize=(
            primary_config.figsize
            if primary_config is not None
            else (ann_cfg.figsize if ann_cfg is not None else (9, 6))
        )
    )
    ax2 = None

    face = None
    transparent = False
    if primary_config is not None:
        face = getattr(primary_config, "figure_facecolor", None)
        transparent = getattr(primary_config, "figure_transparent", False)
    elif ann_cfg is not None:
        face = getattr(ann_cfg, "figure_facecolor", None)
        transparent = getattr(ann_cfg, "figure_transparent", False)
    fig.patch.set_facecolor(face if face is not None else "white")
    fig.patch.set_alpha(0.0 if transparent else 1.0)

    def _categories_in_order(series: pd.Series) -> list:
        """
        Return category ordering for a series, preserving categorical dtype order when present.
        """
        if pd.api.types.is_categorical_dtype(series):
            return list(series.cat.categories)
        return list(pd.unique(series))

    def _combine_categories(base: list, new_vals: list) -> list:
        """
        Append new_vals to base preserving order and avoiding duplicates.
        """
        seen = set(base)
        combined = list(base)
        for val in new_vals:
            if val not in seen:
                combined.append(val)
                seen.add(val)
        return combined

    def _palette_dict(labels: list, palette_cfg) -> dict[str, str]:
        """
        Construct a mapping of label -> color from various palette configurations.
        """
        if not labels:
            return {}
        if palette_cfg is None:
            palette_cfg = sns.color_palette("Dark2", n_colors=len(labels))
        if isinstance(palette_cfg, str):
            return {label: palette_cfg for label in labels}
        if isinstance(palette_cfg, dict):
            fallback_palette = sns.color_palette("Dark2", n_colors=len(labels))
            return {
                label: palette_cfg.get(label, fallback_palette[i % len(fallback_palette)])
                for i, label in enumerate(labels)
            }
        # palette_cfg is a list-like
        if len(palette_cfg) < len(labels):
            palette_cfg = sns.color_palette("Dark2", n_colors=len(labels))
        return {label: color for label, color in zip(labels, palette_cfg)}

    all_x_levels: list = []
    if has_df:
        x_col = primary_config.x
        if not pd.api.types.is_categorical_dtype(df[x_col]):
            x_dtype = CategoricalDtype(categories=list(pd.unique(df[x_col])), ordered=True)
            df[x_col] = df[x_col].astype(x_dtype)
        else:
            try:
                df[x_col] = df[x_col].cat.remove_unused_categories()
            except Exception:
                pass
        df_cats = _categories_in_order(df[x_col])
        all_x_levels = _combine_categories(all_x_levels, df_cats)
    if has_twinx:
        twinx_x_col = ann_cfg.x
        if not pd.api.types.is_categorical_dtype(twinx_data[twinx_x_col]):
            twinx_dtype = CategoricalDtype(
                categories=list(pd.unique(twinx_data[twinx_x_col])), ordered=True
            )
            twinx_data[twinx_x_col] = twinx_data[twinx_x_col].astype(twinx_dtype)
        else:
            try:
                twinx_data[twinx_x_col] = twinx_data[twinx_x_col].cat.remove_unused_categories()
            except Exception:
                pass
        twinx_cats = _categories_in_order(twinx_data[twinx_x_col])
        all_x_levels = _combine_categories(all_x_levels, twinx_cats)

    if not all_x_levels:
        all_x_levels = []
    cat_to_pos = {cat: i for i, cat in enumerate(all_x_levels)}
    xpad = getattr(primary_config or ann_cfg, "xlim_padding", 0.8)
    x_end = max(len(all_x_levels) - 0.5, 0)
    align_origin = getattr(primary_config or ann_cfg, "align_first_tick_to_origin", False)
    x_start = 0 if align_origin else -1 * xpad
    ax.set_xlim(x_start, x_end + xpad)

    if has_df:
        if not pd.api.types.is_categorical_dtype(df[x_col]) or list(
            df[x_col].cat.categories
        ) != list(all_x_levels):
            all_tp_dtype = CategoricalDtype(categories=all_x_levels, ordered=True)
            df[x_col] = df[x_col].astype(all_tp_dtype)
        if not primary_config.title:
            ref_row = df[primary_config.col_vals_to_include_in_title].iloc[0]
            primary_config.title = " | ".join(
                ", ".join(map(str, val)) if isinstance(val, list) else str(val)
                for val in ref_row.to_dict().values()
            )

    if has_twinx:
        if not pd.api.types.is_categorical_dtype(twinx_data[twinx_x_col]) or list(
            twinx_data[twinx_x_col].cat.categories
        ) != list(all_x_levels):
            all_tp_dtype = CategoricalDtype(categories=all_x_levels, ordered=True)
            twinx_data[twinx_x_col] = twinx_data[twinx_x_col].astype(all_tp_dtype)

    overlay_color_dict: dict[str, str] = {}
    if has_twinx:
        hue_col = secondary_hue
        secondary_labels = sorted(twinx_data[hue_col].dropna().unique())
        overlay_color_dict = _palette_dict(secondary_labels, overlay_palette)

    default_ylim = 105
    use_absolute_scale_main = (
        has_df
        and hasattr(primary_config, "use_absolute_scale")
        and primary_config.use_absolute_scale
    )
    # Allow the secondary/twin config to override symmetric scaling when provided.
    if ann_cfg is not None and hasattr(ann_cfg, "symmetric_ylim"):
        symmetric_ylim = ann_cfg.symmetric_ylim
    elif primary_config is not None and hasattr(primary_config, "symmetric_ylim"):
        symmetric_ylim = primary_config.symmetric_ylim
    else:
        symmetric_ylim = True

    y1_max = df[primary_config.y].max() if has_df else 0
    y1_min = df[primary_config.y].min() if has_df else 0
    y2_max = twinx_data[secondary_y].max() if has_twinx else 0
    y2_min = twinx_data[secondary_y].min() if has_twinx else 0

    if use_absolute_scale_main:
        combined_max = abs(y1_max)
    else:
        scan_extreme = max(abs(y2_max), abs(y2_min)) if has_twinx else 0
        combined_max = max(abs(y1_max), abs(y1_min), scan_extreme)

    if use_absolute_scale_main:
        base_ylim = primary_config.absolute_ylim or (-5, 105)
        absolute_yticks = getattr(primary_config, "absolute_yticks", None)
        ymax = combined_max
        upper_bound = base_ylim[1]
        if ymax > upper_bound:
            upper_bound = math.ceil(ymax / 50.0) * 50
        ylim = (base_ylim[0], upper_bound)
        if absolute_yticks:
            yticks = np.asarray(absolute_yticks)
            spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 25
        else:
            yticks = np.arange(base_ylim[0], upper_bound + 1, 25)
            spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 25
        ax.set_ylim(*ylim)
        new_ylim = ylim[1]
    else:
        if symmetric_ylim:
            ymax = combined_max
            extent = default_ylim
            if ymax > extent:
                extent = math.ceil(ymax / 50.0) * 50
            new_ylim = extent
            spacing = 25
            ax.set_ylim(-extent, extent)
            yticks = np.arange(-extent, extent + 1, spacing)
            ax.set_yticks(yticks)
        else:
            if has_df:
                ymin = y1_min
                ymax = y1_max
            elif has_twinx:
                ymin = y2_min
                ymax = y2_max
            else:
                ymin, ymax = 0, 1
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            yrange = ymax - ymin
            padding = 0.05 * yrange if yrange > 0 else 1
            ax.set_ylim(ymin - padding, ymax + padding)
            yticks = ax.get_yticks()
            spacing = yticks[1] - yticks[0] if len(yticks) > 1 else yrange or 1
            new_ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))

    text_y = new_ylim - 0.4 * spacing
    overlay_use_axes = getattr(ann_cfg, "overlay_in_axes_coords", False)
    overlay_ypos_axes = getattr(ann_cfg, "overlay_ypos_axes", 0.98)

    if has_df and not has_twinx:
        generate_styled_multigroup_lineplot(df=df, config=primary_config, ax=ax)
        ax.set_ylabel(primary_config.ylabel or r"$\Delta$ from First Timepoint", fontweight="bold")
        # Allow callers to override legend anchor via config fields.
        if ann_cfg is not None:
            _adj_x = getattr(ann_cfg, "adjust_legend_x", 1.2)
            _adj_y = getattr(ann_cfg, "adjust_legend_y", 0.7)
        else:
            _adj_x = getattr(primary_config, "adjust_legend_x", 1.2)
            _adj_y = getattr(primary_config, "adjust_legend_y", 0.7)
        adjust_legend(ax, (_adj_x, _adj_y), redraw=True)
        ax.set_facecolor("white")
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(right=primary_config.rhs_pdf_padding)
        xticks = ax.get_xticks()
        if len(xticks) > 8:
            ax.tick_params(axis="x", labelsize=13, rotation=45)
        else:
            ax.tick_params(axis="x", labelsize=13)
        return fig

    if has_twinx and not has_df:
        reference_val = getattr(ann_cfg, "reference", None) if ann_cfg else None
        if reference_val is not None:
            reference_kwargs = dict(
                y=reference_val,
                color=getattr(ann_cfg, "reference_color", "#C0C0C0"),
                linestyle=getattr(ann_cfg, "reference_style", "--"),
                linewidth=getattr(ann_cfg, "reference_width", 1.0),
                alpha=getattr(ann_cfg, "reference_alpha", 1.0),
                zorder=-1,
            )
            rdashes = getattr(ann_cfg, "reference_dashes", (5, 5))
            if rdashes is not None:
                reference_kwargs["dashes"] = rdashes
            ax.axhline(**reference_kwargs)
        # Decide which config and DataFrame supplies annotations.
        # annotation_field already resolved above
        if annotation_field and annotation_field in twinx_data.columns:
            ann_df = twinx_data
        else:
            ann_df = twinx_data  # fallback to twinx_data even if missing to avoid breakage
        annotations = (
            ann_df[[twinx_x_col, annotation_field]]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
            if annotation_field and annotation_field in ann_df.columns
            else pd.DataFrame(columns=[twinx_x_col, "_annotation"])
        )
        annotation_labels = (
            sorted(annotations[annotation_field].unique()) if not annotations.empty else []
        )
        overlay_palette_cfg = getattr(ann_cfg, "overlay_palette", None) or getattr(
            ann_cfg, "palette", None
        )
        overlay_palette = _palette_dict(annotation_labels, overlay_palette_cfg)
        overlay_fontweight = getattr(ann_cfg, "overlay_fontweight", "bold")
        overlay_fontsize = getattr(ann_cfg, "overlay_fontsize", None) or 14
        overlay_vline_color = getattr(ann_cfg, "overlay_vline_color", "gainsboro")
        overlay_vline_style = getattr(ann_cfg, "overlay_vline_style", "--")
        overlay_vline_width = getattr(ann_cfg, "overlay_vline_width", 1.0)
        overlay_vline_dashes = getattr(ann_cfg, "overlay_vline_dashes", (5, 5))
        for _, row in annotations.iterrows():
            x = cat_to_pos[row[twinx_x_col]]
            text = row[annotation_field] if annotation_field in row else None
            if text is None:
                continue
            if overlay_use_axes:
                ax.annotate(
                    text,
                    xy=(x + 0.03, overlay_ypos_axes),
                    xycoords=("data", "axes fraction"),
                    fontsize=overlay_fontsize,
                    fontweight=overlay_fontweight,
                    color=(
                        annotation_color_dict.get(text, "black")
                        if annotation_color_dict
                        else overlay_palette.get(text, "black")
                    ),
                    ha="left",
                    va="center",
                    zorder=1,
                )
            else:
                ax.text(
                    x + 0.03,
                    text_y,
                    text,
                    fontsize=overlay_fontsize,
                    fontweight=overlay_fontweight,
                    color=(
                        annotation_color_dict.get(text, "black")
                        if annotation_color_dict
                        else overlay_palette.get(text, "black")
                    ),
                    zorder=1,
                )
            overlay_vline_alpha = getattr(ann_cfg, "overlay_vline_alpha", 1.0)
            vline_kwargs = dict(
                x=x,
                color=overlay_vline_color,
                linestyle=overlay_vline_style,
                linewidth=overlay_vline_width,
                alpha=overlay_vline_alpha,
                zorder=-1,
            )
            if overlay_vline_dashes is not None:
                vline_kwargs["dashes"] = overlay_vline_dashes
            ax.axvline(**vline_kwargs)
        sns.lineplot(
            data=twinx_data,
            x=twinx_x_col,
            y=secondary_y,
            hue=secondary_hue,
            palette=overlay_color_dict or overlay_palette,
            ax=ax,
            linewidth=ann_cfg.lw / 2,
            linestyle=secondary_linestyle,
            alpha=ann_cfg.twin_alpha,
            zorder=2,
        )
        sns.scatterplot(
            data=twinx_data,
            x=twinx_x_col,
            y=secondary_y,
            hue=secondary_hue,
            palette=overlay_color_dict or overlay_palette,
            ax=ax,
            s=(ann_cfg.markersize / 1.5) ** 2,
            edgecolor=getattr(ann_cfg, "edgecolor", "white"),
            linewidth=1,
            legend=False,
            zorder=2,
        )
        texts, x_pos_, y_pos_ = [], [], []
        palette_lookup = overlay_color_dict
        for label, group in twinx_data.groupby(secondary_hue):
            last = group.dropna(subset=[twinx_x_col, secondary_y]).iloc[-1]
            x = cat_to_pos[last[twinx_x_col]] + np.random.normal(0, 0.01)
            y = last[secondary_y] + 2 + np.random.normal(0, 0.01)
            x_pos_.append(x)
            y_pos_.append(y)
            texts.append(
                ax.text(
                    x + 0.05,
                    y + 0.3,
                    label,
                    fontdict={
                        "family": resolve_font_family(),
                        "size": 10,
                        "weight": overlay_fontweight,
                    },
                    color=palette_lookup.get(label, "black"),
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
        ax.set_title(ann_cfg.title, loc="left", fontweight="bold")
        ax.set_ylabel(
            r"Diameter %$\Delta$ from First Timepoint",
            fontdict={
                "family": resolve_font_family(),
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
            prop=font_manager.FontProperties(family=resolve_font_family(), size=14, weight="bold"),
        )
        _adj_x = getattr(ann_cfg, "adjust_legend_x", 1.2) if ann_cfg else 1.2
        _adj_y = getattr(ann_cfg, "adjust_legend_y", 0.7) if ann_cfg else 0.7
        adjust_legend(ax, (_adj_x, _adj_y))
        fig.subplots_adjust(right=primary_config.rhs_pdf_padding if primary_config else 0.85)
        ax.set_facecolor("white")
        fig.patch.set_alpha(0.0)
        ax.set_xlabel(ann_cfg.x, fontweight="bold")
        xtick_size = getattr(primary_config or ann_cfg, "xtick_fontsize", None)
        xticks = ax.get_xticks()
        if xtick_size is not None:
            ax.tick_params(axis="x", labelsize=xtick_size)
        else:
            if len(xticks) > 8:
                ax.tick_params(axis="x", labelsize=13, rotation=45)
            else:
                ax.tick_params(axis="x", labelsize=13)
        return fig

    generate_styled_multigroup_lineplot(df=df, config=primary_config, ax=ax)
    ax2 = ax.twinx()
    if ax2:
        ax2.set_xlim(x_start, x_end + xpad)
    # annotation_field already resolved above

    reference_val = getattr(ann_cfg, "reference", None) if ann_cfg else None
    if reference_val is not None:
        target_ax = ax2 if ax2 is not None else ax
        reference_kwargs = dict(
            y=reference_val,
            color=getattr(ann_cfg, "reference_color", "#C0C0C0"),
            linestyle=getattr(ann_cfg, "reference_style", "--"),
            linewidth=getattr(ann_cfg, "reference_width", 1.0),
            alpha=getattr(ann_cfg, "reference_alpha", 1.0),
            zorder=-1,
        )
        rdashes = getattr(ann_cfg, "reference_dashes", (5, 5))
        if rdashes is not None:
            reference_kwargs["dashes"] = rdashes
        target_ax.axhline(**reference_kwargs)

    def _annotation_sources() -> list[tuple[str, pd.DataFrame | None]]:
        # Default: prefer primary (main axis) annotations, then fall back to secondary (twin axis).
        if source_preference == "primary":
            return [("primary", df), ("secondary", twinx_data)]
        if source_preference == "secondary":
            return [("secondary", twinx_data), ("primary", df)]
        return [("primary", df), ("secondary", twinx_data)]

    ann_df = twinx_data
    ann_source_used = "secondary"
    for source_name, candidate in _annotation_sources():
        if candidate is None or candidate.empty or annotation_field not in candidate.columns:
            continue
        ann_df = candidate
        ann_source_used = source_name
        break

    if ann_source_used == "primary" and primary_config:
        twinx_x_col = primary_config.x
        cat_to_pos = {cat: i for i, cat in enumerate(all_x_levels)}
    annotations = (
        ann_df[[twinx_x_col, annotation_field]].dropna().drop_duplicates().reset_index(drop=True)
        if annotation_field and annotation_field in ann_df.columns
        else pd.DataFrame(columns=[twinx_x_col, "_annotation"])
    )

    # Build annotation color mapping based on the chosen source, allowing palettes on either config.
    annotation_labels = (
        sorted(annotations[annotation_field].unique()) if not annotations.empty else []
    )
    primary_palette_cfg = None
    if primary_config is not None:
        primary_palette_cfg = getattr(primary_config, "overlay_palette", None) or getattr(
            primary_config, "palette", None
        )
    overlay_palette_cfg = getattr(ann_cfg, "overlay_palette", None) or getattr(
        ann_cfg, "palette", None
    )
    overlay_palette = _palette_dict(
        annotation_labels,
        primary_palette_cfg if ann_source_used == "primary" else overlay_palette_cfg,
    )
    # Default to black when caller does not pass a dict/palette
    if not annotation_color_dict and not overlay_palette_cfg and not primary_palette_cfg:
        overlay_palette = {label: "black" for label in annotation_labels}
    overlay_fontweight = (
        getattr(primary_config, "overlay_fontweight", None)
        if ann_source_used == "primary"
        else getattr(ann_cfg, "overlay_fontweight", None)
    ) or "bold"
    overlay_fontsize = (
        getattr(primary_config, "overlay_fontsize", None)
        if ann_source_used == "primary"
        else getattr(ann_cfg, "overlay_fontsize", None)
    ) or 14
    overlay_vline_color = getattr(ann_cfg, "overlay_vline_color", "gainsboro")
    overlay_vline_style = getattr(ann_cfg, "overlay_vline_style", "--")
    overlay_vline_width = getattr(ann_cfg, "overlay_vline_width", 1.0)
    overlay_vline_dashes = getattr(ann_cfg, "overlay_vline_dashes", (5, 5))
    overlay_vline_alpha = getattr(ann_cfg, "overlay_vline_alpha", 1.0)
    for _, row in annotations.iterrows():
        x = cat_to_pos[row[twinx_x_col]]
        text = row[annotation_field] if annotation_field in row else None
        if text is None:
            continue
        if overlay_use_axes:
            ax.annotate(
                text,
                xy=(x + 0.03, overlay_ypos_axes),
                xycoords=("data", "axes fraction"),
                fontsize=overlay_fontsize,
                fontweight=overlay_fontweight,
                color=(
                    annotation_color_dict.get(text, "black")
                    if annotation_color_dict
                    else overlay_palette.get(text, "black")
                ),
                ha="left",
                va="center",
                zorder=1,
            )
        else:
            ax.text(
                x + 0.03,
                text_y,
                text,
                fontsize=overlay_fontsize,
                fontweight=overlay_fontweight,
                color=(
                    annotation_color_dict.get(text, "black")
                    if annotation_color_dict
                    else overlay_palette.get(text, "black")
                ),
                zorder=1,
            )
        vline_kwargs = dict(
            x=x,
            color=overlay_vline_color,
            linestyle=overlay_vline_style,
            linewidth=overlay_vline_width,
            alpha=overlay_vline_alpha,
            zorder=1,
        )
        if overlay_vline_dashes is not None:
            vline_kwargs["dashes"] = overlay_vline_dashes
        ax.axvline(**vline_kwargs)
    sns.lineplot(
        data=twinx_data,
        x=twinx_x_col,
        y=secondary_y,
        hue=secondary_hue,
        palette=overlay_color_dict or overlay_palette,
        ax=ax2,
        linewidth=ann_cfg.lw / 2,
        linestyle=secondary_linestyle,
        alpha=ann_cfg.twin_alpha,
        zorder=2,
    )
    sns.scatterplot(
        data=twinx_data,
        x=twinx_x_col,
        y=secondary_y,
        hue=secondary_hue,
        palette=overlay_color_dict or overlay_palette,
        ax=ax2,
        s=(ann_cfg.markersize / 1.5) ** 2,
        edgecolor="white",
        linewidth=1,
        legend=False,
        zorder=2,
    )
    texts, x_pos_, y_pos_ = [], [], []
    palette_lookup = overlay_color_dict
    for label, group in twinx_data.groupby(secondary_hue):
        last = group.dropna(subset=[twinx_x_col, secondary_y]).iloc[-1]
        x = cat_to_pos[last[twinx_x_col]] + np.random.normal(0, 0.01)
        y = last[secondary_y] + 2 + np.random.normal(0, 0.01)
        x_pos_.append(x)
        y_pos_.append(y)
        texts.append(
            ax2.text(
                x + 0.05,
                y + 0.3,
                label,
                fontdict={
                    "family": resolve_font_family(),
                    "size": 10,
                    "weight": overlay_fontweight,
                },
                color=palette_lookup.get(label, "black"),
                ha="left",
                va="center",
                zorder=10,
            )
        )
    plt.draw()
    # Respect user-provided ylims when supplied.
    main_ylim_override = getattr(primary_config, "ylim", None) if primary_config else None
    if main_ylim_override is not None:
        y0, y1 = main_ylim_override
        lower = y0 if y0 is not None else ax.get_ylim()[0]
        upper = y1 if y1 is not None else ax.get_ylim()[1]
        ax.set_ylim(lower, upper)

    main_ylim = ax.get_ylim()

    if use_absolute_scale_main:
        if has_twinx and ax2:
            scan_extreme = max(abs(y2_max), abs(y2_min)) if has_twinx else 0
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
            ax2.set_ylim(main_ylim)
    else:
        if ax2:
            twin_ylim_override = getattr(ann_cfg, "ylim", None) if ann_cfg else None
            if twin_ylim_override is not None:
                y0, y1 = twin_ylim_override
                lower = y0 if y0 is not None else main_ylim[0]
                upper = y1 if y1 is not None else main_ylim[1]
                ax2.set_ylim(lower, upper)
            elif not symmetric_ylim and has_twinx:
                # Auto-scale twin axis from its own data when asymmetric scaling is requested.
                tymin = twinx_data[secondary_y].min()
                tymax = twinx_data[secondary_y].max()
                if tymin == tymax:
                    tymin -= 1
                    tymax += 1
                tyrange = tymax - tymin
                padding = 0.05 * tyrange if tyrange > 0 else 1
                ax2.set_ylim(tymin - padding, tymax + padding)
            else:
                ax2.set_ylim(main_ylim)
            if symmetric_ylim:
                ax2.set_yticks(ax.get_yticks())
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
            "family": resolve_font_family(),
            "size": ax.yaxis.label.get_fontsize(),
            "weight": "bold",
        },
    )
    # Propagate configured font sizes to the twin y-axis when provided.
    ylabel_size = getattr(ann_cfg, "ylabel_fontsize", None) or getattr(
        primary_config, "ylabel_fontsize", None
    )
    if ylabel_size is not None:
        ax2.yaxis.label.set_size(ylabel_size)
    tick_kwargs = {}
    ytick_size = getattr(ann_cfg, "ytick_fontsize", None) or getattr(
        primary_config, "ytick_fontsize", None
    )
    if ytick_size is not None:
        tick_kwargs["labelsize"] = ytick_size
    if tick_kwargs:
        ax2.tick_params(axis="y", **tick_kwargs)
    handles, labels = ax2.get_legend_handles_labels()
    section_label = Line2D([0], [0], color="none", label="Location")
    for h in handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)
    ax2.legend(
        handles=[section_label] + handles,
        labels=["Location"] + labels,
        frameon=False,
        prop=font_manager.FontProperties(family=resolve_font_family(), size=14, weight="bold"),
    )
    _adj_x_ax2 = getattr(ann_cfg, "adjust_legend_x", 1.2) if ann_cfg else 1.2
    _adj_y_ax2 = getattr(ann_cfg, "adjust_legend_y", 0.3) if ann_cfg else 0.3
    adjust_legend(ax2, (_adj_x_ax2, _adj_y_ax2))
    _adj_x_ax = getattr(primary_config, "adjust_legend_x", 1.2) if primary_config else 1.2
    _adj_y_ax = getattr(primary_config, "adjust_legend_y", 0.8) if primary_config else 0.8
    adjust_legend(ax, (_adj_x_ax, _adj_y_ax), redraw=True)
    fig.subplots_adjust(right=primary_config.rhs_pdf_padding)
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0)
    xtick_size = getattr(primary_config or ann_cfg, "xtick_fontsize", None)
    xticks = ax.get_xticks()
    if xtick_size is not None:
        ax.tick_params(axis="x", labelsize=xtick_size)
        if ax2:
            ax2.tick_params(axis="x", labelsize=xtick_size)
    else:
        if len(xticks) > 8:
            ax.tick_params(axis="x", labelsize=13, rotation=45)
        else:
            ax.tick_params(axis="x", labelsize=13)
    return fig
