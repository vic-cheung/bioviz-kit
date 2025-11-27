"""
Line plot utilities (bioviz)

Ported and adapted from tm_toolbox. Uses neutral `DefaultStyle`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import font_manager
from matplotlib.lines import Line2D

from bioviz.configs import StyledLinePlotConfig
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

# Expose public function
__all__ = ["generate_styled_lineplot"]


def generate_styled_lineplot(
    df: pd.DataFrame,
    config: StyledLinePlotConfig,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if df.empty:
        print(f"No data for patient {config.patient_id}: DataFrame is empty.")
        return None

    if config.label_col not in df:
        print(
            f"No data for patient {config.patient_id}: Column '{config.label_col}' does not exist."
        )
        return None

    if df[config.label_col].dropna().empty:
        print(
            f"No data for patient {config.patient_id}: Column '{config.label_col}' contains only missing values."
        )
        return None

    if config.y not in df:
        print(f"No data for patient {config.patient_id}: Column '{config.y}' does not exist.")
        return None

    if df[config.y].dropna().empty:
        print(
            (
                f"No data for patient {config.patient_id}: Column '{config.y}' contains only missing values."
            )
        )
        return None

    # Ensure long-format required columns exist; bioviz expects callers to
    # supply long-format data. Forward-fill (if desired) should be done by
    # adapters (e.g. tm_toolbox) before calling bioviz.
    required_cols = {config.x, config.y, config.label_col, config.secondary_group_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns for long-format plotting: {missing}"
        )

    df = df.dropna(subset=[config.y]).copy()

    if config.x in df and hasattr(df[config.x].dtype, "categories"):
        try:
            cleaned = df[config.x].cat.remove_unused_categories()
            df[config.x] = cleaned
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

    if cat_to_pos:
        xmax = max(cat_to_pos.values()) + config.xlim_padding
    else:
        xmax = 1

    if config.threshold:
        ax.axhline(
            y=config.threshold,
            color="#C0C0C0",
            linestyle="--",
            dashes=(5, 5),
            linewidth=1,
            zorder=0,
        )
        yticks = ax.get_yticks()
        spacing = yticks[1] - yticks[0] if len(yticks) > 1 else 1
        text_y = config.threshold + 0.1 * spacing
        ax.text(
            xmax - 0.25,
            text_y,
            r"$LoD_{95}$",
            fontsize=14,
            fontweight="normal",
            color="#C0C0C0",
            zorder=0,
        )

    xlabel = config.xlabel if config.xlabel is not None else config.x
    ylabel = config.ylabel if config.ylabel is not None else config.y
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(-1 * config.xlim_padding, xmax), ylim=ylim)
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold")
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

    handles = [Line2D([0], [0], linewidth=0, label="Variant", color="black")] + [
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
                Line2D([0], [0], linewidth=0, label="Detection Status", color="black"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="white",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=config.markersize,
                    linewidth=0,
                    label="Not detected",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="white",
                    markerfacecolor="black",
                    markeredgecolor="white",
                    markersize=config.markersize,
                    linewidth=0,
                    label="Detected",
                ),
            ]
        )

    lgd = ax.legend(
        handles=handles + detection_handles,
        bbox_to_anchor=(1.25, 0.5),
        loc="center",
        frameon=False,
        prop=font_manager.FontProperties(family=DefaultStyle().font_family, size=14, weight="bold"),
    )

    if config.match_legend_text_color:
        for text in lgd.get_texts():
            label = text.get_text()
            if label in color_dict:
                text.set_color(color_dict[label])
                text.set_fontweight("bold")
            elif label in {"Variant", "Detection Status"}:
                text.set_color("black")
                text.set_fontweight("bold")
                text.set_fontsize(16)

    plt.subplots_adjust(right=config.rhs_pdf_padding)
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0)

    return fig
