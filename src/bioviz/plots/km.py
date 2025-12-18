"""
Kaplan-Meier plotting helpers ported to bioviz.

These functions are plotting-only: they accept lifelines KaplanMeierFitter
objects and matplotlib axes and render curves, risk tables, legends, and
annotations. They intentionally do not perform data fitting or statistical
tests — those remain in tm_modeling.
"""

from typing import Mapping, Optional

import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _wrap_label(text: str | None, width: int | None, max_lines: int | None) -> str | None:
    if text is None or width is None:
        return text
    lines = textwrap.wrap(str(text), width=int(width))
    if max_lines is not None and len(lines) > int(max_lines):
        lines = lines[: int(max_lines)]
        if not lines[-1].endswith("…"):
            lines[-1] = lines[-1].rstrip() + "…"
    return "\n".join(lines)


def _wrap_labels(labels: list[str | None] | None, width: int | None, max_lines: int | None):
    if labels is None or width is None:
        return labels
    return [_wrap_label(lab, width, max_lines) for lab in labels]


def plot_custom_km(
    kmf,
    ax,
    color="#1f77b4",
    label=None,
    show_ci=True,
    ci_alpha=0.25,
    ci_style="fill",
    linewidth=1.5,
    linestyle="-",
    censor_marker="+",
    censor_markersize=12,
    censor_markeredgewidth=1,
    force_show_censors=True,
    conf_type="log_log",
    per_patient_censor_markers=False,
):
    x_vals = kmf.survival_function_.index.to_numpy()
    y_vals = kmf.survival_function_.iloc[:, 0].to_numpy()
    plot_label = label if label is not None else getattr(kmf, "_label", None)

    if x_vals.size <= 1:
        xmin, xmax = ax.get_xlim()
        if xmax <= xmin:
            xmin, xmax = 0.0, 1.0
        line = ax.hlines(
            y=y_vals[0] if y_vals.size else 1.0,
            xmin=xmin,
            xmax=xmax,
            colors=color,
            linewidth=linewidth,
            linestyles=linestyle,
            label=plot_label,
        )
    else:
        line = ax.step(
            x_vals,
            y_vals,
            where="post",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=plot_label,
        )

    if show_ci and hasattr(kmf, "confidence_interval_"):
        ci_cols = kmf.confidence_interval_.columns
        lower_col = None
        upper_col = None
        if conf_type == "linear" and "linear_lower_0.95" in ci_cols:
            lower_col = "linear_lower_0.95"
            upper_col = "linear_upper_0.95"
        elif len(ci_cols) >= 2:
            lower_col = [col for col in ci_cols if "lower" in col.lower()][0]
            upper_col = [col for col in ci_cols if "upper" in col.lower()][0]

        if lower_col and upper_col:
            lower = kmf.confidence_interval_[lower_col]
            upper = kmf.confidence_interval_[upper_col]
            ci_x = kmf.confidence_interval_.index.to_numpy()
            if ci_style == "fill":
                ax.fill_between(
                    ci_x,
                    lower.to_numpy(),
                    upper.to_numpy(),
                    alpha=ci_alpha,
                    step="post",
                    color=color,
                    linewidth=0,
                )
            elif ci_style == "lines":
                ax.step(
                    ci_x,
                    lower.to_numpy(),
                    where="post",
                    color=color,
                    linewidth=linewidth / 2,
                    alpha=0.7,
                    linestyle="--",
                )
                ax.step(
                    ci_x,
                    upper.to_numpy(),
                    where="post",
                    color=color,
                    linewidth=linewidth / 2,
                    alpha=0.7,
                    linestyle="--",
                )

    # Censor markers
    try:
        et = kmf.event_table
        cens_col = et["censored"] if "censored" in et.columns else None
    except Exception:
        et, cens_col = None, None

    plotted_censors = False
    if (
        per_patient_censor_markers
        and hasattr(kmf, "original_durations")
        and hasattr(kmf, "original_events")
    ):
        try:
            pp_times = [
                t for t, e in zip(kmf.original_durations, kmf.original_events) if int(e) == 0
            ]
        except Exception:
            pp_times = []
        if len(pp_times) > 0:
            surv_series = kmf.predict(np.array(pp_times))
            surv_at_times = (
                surv_series.values if hasattr(surv_series, "values") else np.asarray(surv_series)
            )
            ax.scatter(
                pp_times,
                surv_at_times,
                marker=censor_marker,
                s=(censor_markersize**2),
                linewidths=censor_markeredgewidth,
                color=color,
                zorder=11,
            )
            plotted_censors = True
    if not plotted_censors and et is not None and cens_col is not None:
        cens_mask = cens_col > 0
        if np.any(cens_mask.values if hasattr(cens_mask, "values") else cens_mask):
            cens_times = et.index.values[cens_mask]
            surv_series = kmf.predict(cens_times)
            surv_at_times = (
                surv_series.values if hasattr(surv_series, "values") else np.asarray(surv_series)
            )
            ax.scatter(
                cens_times,
                surv_at_times,
                marker=censor_marker,
                s=(censor_markersize**2),
                linewidths=censor_markeredgewidth,
                color=color,
                zorder=11,
            )
            plotted_censors = True
    if not plotted_censors and force_show_censors:
        if len(kmf.timeline) > 2:
            median_idx = len(kmf.timeline) // 2
            t = kmf.timeline[median_idx]
            prob = kmf.survival_function_.iloc[median_idx].values[0]
            ax.scatter(
                [t],
                [prob],
                marker=censor_marker,
                s=(censor_markersize**2),
                linewidths=censor_markeredgewidth,
                color=color,
                zorder=11,
            )

    return line


def format_pvalue(p_value, significance_cutoffs=None):
    if significance_cutoffs is None:
        significance_cutoffs = {
            0.001: "p < 0.001",
            0.01: "p = {:.3f}",
            0.05: "p = {:.3f}",
            1.0: "p = {:.2f}",
        }
    for cutoff, format_str in sorted(significance_cutoffs.items()):
        if p_value < cutoff:
            if "< " in format_str:
                return format_str
            else:
                return format_str.format(p_value)
    return f"p = {p_value:.2f}"


def add_pvalue_annotation(
    ax, p_value, loc="bottom_right", box=True, fontsize=12, alpha=0.8, format_p=True
):
    if p_value is None:
        return None
    if format_p:
        p_text = format_pvalue(p_value)
    else:
        p_text = f"p = {p_value:.4f}"
    position_map = {
        "top_left": (0.05, 0.95),
        "top_right": (0.95, 0.95),
        "bottom_left": (0.05, 0.05),
        "bottom_right": (0.95, 0.05),
        "center_right": (0.95, 0.5),
    }
    xy = position_map.get(loc, (0.95, 0.05))
    ha = "right" if xy[0] > 0.5 else "left"
    va = "top" if xy[1] > 0.5 else "bottom"
    bbox_props = None
    if box:
        bbox_props = dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=alpha, edgecolor="lightgray"
        )
    text = ax.annotate(
        p_text, xy=xy, xycoords="axes fraction", ha=ha, va=va, fontsize=fontsize, bbox=bbox_props
    )
    return text


def position_legend(
    ax,
    loc="bottom",
    ncol=None,
    fontsize=10,
    frameon=True,
    title=None,
    title_fontsize=None,
    title_fontweight=None,
    label_fontweight=None,
    label_wrap_chars=None,
    label_max_lines=2,
    markerscale: float = 1.0,
    linewidth_scale: float | None = None,
):
    if title_fontsize is None:
        title_fontsize = fontsize + 2
    handles, labels = ax.get_legend_handles_labels()
    if label_wrap_chars is not None and labels:
        labels = _wrap_labels(labels, label_wrap_chars, label_max_lines)
    if not handles:
        return None
    if ncol is None:
        if loc == "bottom":
            ncol = max(1, min(len(handles), 3))
        else:
            ncol = 1
    if loc == "bottom":
        legend = ax.legend(
            handles,
            labels,
            fontsize=fontsize,
            frameon=frameon,
            title=title,
            title_fontsize=title_fontsize,
            ncol=ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            markerscale=markerscale,
        )
    elif loc == "right":
        legend = ax.legend(
            handles,
            labels,
            fontsize=fontsize,
            frameon=frameon,
            title=title,
            title_fontsize=title_fontsize,
            ncol=ncol,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
            markerscale=markerscale,
        )
    else:
        legend = ax.legend(
            handles,
            labels,
            fontsize=fontsize,
            frameon=frameon,
            title=title,
            title_fontsize=title_fontsize,
            ncol=ncol,
            loc=loc,
            markerscale=markerscale,
        )
    if frameon and legend:
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor("lightgray")
    if legend is not None:
        if title_fontweight is not None:
            try:
                legend.get_title().set_fontweight(title_fontweight)
            except Exception:
                pass
        if label_fontweight is not None:
            for txt in legend.get_texts():
                try:
                    txt.set_fontweight(label_fontweight)
                except Exception:
                    pass
        try:
            if linewidth_scale is not None and float(linewidth_scale) > 0:
                for legline in legend.get_lines():
                    try:
                        lw = legline.get_linewidth()
                        legline.set_linewidth(float(lw) * float(linewidth_scale))
                    except Exception:
                        continue
        except Exception:
            pass
    return legend


def expand_figure_to_fit_legend(fig, legend, pad_in: float = 0.25) -> None:
    if legend is None or fig is None:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    try:
        leg_px = legend.get_window_extent(renderer=renderer)
        fig_px = fig.bbox
        overhang_px = max(0.0, float(leg_px.x1 - fig_px.x1))
        if overhang_px > 0:
            extra_in = overhang_px / float(fig.dpi) + float(pad_in)
            w, h = fig.get_size_inches()
            fig.set_size_inches(w + extra_in, h, forward=True)
    except Exception:
        return


def expand_canvas(
    fig, left_in: float = 0.0, right_in: float = 0.0, top_in: float = 0.0, bottom_in: float = 0.0
) -> None:
    if fig is None:
        return
    add_w = max(0.0, float(left_in) + float(right_in))
    add_h = max(0.0, float(top_in) + float(bottom_in))
    if add_w == 0 and add_h == 0:
        return
    w, h = fig.get_size_inches()
    new_w = w + float(left_in) + float(right_in)
    new_h = h + float(top_in) + float(bottom_in)
    for ax in list(fig.axes):
        bbox = ax.get_position()
        x0_in = bbox.x0 * w
        y0_in = bbox.y0 * h
        width_in = bbox.width * w
        height_in = bbox.height * h
        x0_in_new = x0_in + float(left_in)
        y0_in_new = y0_in + float(bottom_in)
        x0_new = x0_in_new / new_w
        y0_new = y0_in_new / new_h
        width_new = width_in / new_w
        height_new = height_in / new_h
        ax.set_position([x0_new, y0_new, width_new, height_new])
    fig.set_size_inches(new_w, new_h, forward=True)


def expand_figure_to_fit_artists(
    fig,
    artists,
    pad_left_in: float = 0.25,
    pad_right_in: float = 0.25,
    pad_top_in: float = 0.0,
    pad_bottom_in: float = 0.0,
) -> None:
    if fig is None or not artists:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    try:
        fig_px = fig.bbox
        left_over_px = 0.0
        right_over_px = 0.0
        top_over_px = 0.0
        bottom_over_px = 0.0
        for art in artists:
            if art is None:
                continue
            try:
                bb = art.get_window_extent(renderer=renderer)
            except Exception:
                continue
            left_over_px = max(left_over_px, float(fig_px.x0 - bb.x0))
            right_over_px = max(right_over_px, float(bb.x1 - fig_px.x1))
            bottom_over_px = max(bottom_over_px, float(fig_px.y0 - bb.y0))
            top_over_px = max(top_over_px, float(bb.y1 - fig_px.y1))
        dpi = float(fig.dpi)
        add_left = (left_over_px / dpi + pad_left_in) if left_over_px > 0 else 0.0
        add_right = (right_over_px / dpi + pad_right_in) if right_over_px > 0 else 0.0
        add_bottom = (bottom_over_px / dpi + pad_bottom_in) if bottom_over_px > 0 else 0.0
        add_top = (top_over_px / dpi + pad_top_in) if top_over_px > 0 else 0.0
        if any(v > 0 for v in (add_left, add_right, add_top, add_bottom)):
            expand_canvas(
                fig, left_in=add_left, right_in=add_right, top_in=add_top, bottom_in=add_bottom
            )
    except Exception:
        return


def adjust_plot_margins(fig, ax, legend_loc="bottom", risk_table=False, risk_table_height=1.0):
    legend = ax.get_legend()
    if legend_loc == "bottom" and legend:
        legend_height = 0.15
        if risk_table:
            bottom_margin = legend_height + risk_table_height / fig.get_figheight()
        else:
            bottom_margin = legend_height
        fig.subplots_adjust(bottom=bottom_margin)
    elif legend_loc == "right" and legend:
        fig.subplots_adjust(right=0.8)
    elif risk_table:
        fig.subplots_adjust(bottom=risk_table_height / fig.get_figheight() + 0.05)


def _get_count_at_time(kmf, t, count_type):
    if t == 0:
        if count_type == "At risk":
            return kmf.event_table.at_risk.iloc[0]
        else:
            return 0
    event_times = kmf.event_table.index.values
    idx = np.searchsorted(event_times, t)
    if idx >= len(event_times):
        return 0
    if idx > 0 and (idx == len(event_times) or t < event_times[idx]):
        idx = idx - 1
    if count_type == "At risk":
        at_risk = kmf.event_table.at_risk.iloc[idx]
        removed = kmf.event_table.removed.iloc[idx]
        return at_risk - removed
    elif count_type == "Censored":
        return kmf.event_table.censored.iloc[: idx + 1].sum()
    elif count_type == "Events":
        return kmf.event_table.observed.iloc[: idx + 1].sum()
    else:
        return 0


def add_custom_risktable(
    ax,
    kmfs,
    labels=None,
    colors=None,
    fontsize=20,
    title_fontsize=None,
    rows_to_show=None,
    table_height=0.25,
    table_gap=0.08,
    xticks=None,
    color_counts=False,
    use_timeline_for_risktable=True,
    table_ax=None,
    row_spacing=1.8,
    title_gap_factor=0.6,
    label_wrap_chars=None,
    label_max_lines=2,
    adaptive_title_gap=True,
    small_n_title_gap_factor=0.3,
    per_row_in_inch=None,
    title_pad_in_inch=None,
    min_display_rows=None,
):
    if rows_to_show is None:
        rows_to_show = ["At risk"]
    fig = ax.figure
    xmin, xmax = ax.get_xlim()
    if xticks is not None:
        time_points = np.array(xticks)
    elif use_timeline_for_risktable and any(
        hasattr(kmf, "timeline_for_risktable_") for kmf in kmfs
    ):
        for kmf in kmfs:
            if hasattr(kmf, "timeline_for_risktable_"):
                time_points = np.array(kmf.timeline_for_risktable_)
                break
    else:
        time_points = np.linspace(0, xmax, min(6, int(xmax) + 1))
        time_points = np.round(time_points, 1)
    if labels is None:
        labels = []
        for kmf in kmfs:
            raw_label = getattr(kmf, "_label", None)
            if raw_label is not None:
                label = raw_label.split(" (n=")[0]
                labels.append(label)
            else:
                labels.append(f"Group {len(labels) + 1}")
    if colors is None:
        colors = []
        for line in ax.lines:
            if line.get_label() not in ("_nolegend_", ""):
                colors.append(line.get_color())
    if len(colors) < len(kmfs):
        colors = colors + ["#1f77b4"] * (len(kmfs) - len(colors))
    if table_ax is not None:
        table_ax.cla()
        table_ax.set_xlim(xmin, xmax)
        if xticks is not None:
            time_points = np.array(xticks, dtype=float)
        else:
            km_ticks = ax.get_xticks()
            time_points = np.array(
                [t for t in km_ticks if xmin - 1e-9 <= t <= xmax + 1e-9], dtype=float
            )
        if labels is None:
            labels = []
            for kmf in kmfs:
                raw_label = getattr(kmf, "_label", None)
                labels.append(
                    raw_label.split(" (n=")[0] if raw_label else f"Group {len(labels) + 1}"
                )
        if colors is None or len(colors) < len(kmfs):
            default = plt.cm.tab10.colors
            new_colors = []
            for i in range(len(kmfs)):
                if colors and i < len(colors):
                    new_colors.append(colors[i])
                else:
                    new_colors.append(default[i % len(default)])
            colors = new_colors
        if time_points.size > 1:
            bin_width = float(time_points[1] - time_points[0])
        else:
            bin_width = max((xmax - xmin) * 0.1, 1e-6)
        half_bin = 0.5 * bin_width
        n_groups = len(kmfs)
        try:
            min_slots = int(min_display_rows) if min_display_rows is not None else n_groups
        except (TypeError, ValueError):
            min_slots = n_groups
        min_slots = max(min_slots, n_groups, 1)
        spacing_factor = row_spacing
        slot_positions = (np.arange(min_slots)[::-1]) * spacing_factor
        y_positions = slot_positions[:n_groups]
        table_ax.set_yticks(y_positions)
        table_ax.set_yticklabels([])
        if per_row_in_inch is not None and title_pad_in_inch is not None:
            pad_axis = (float(title_pad_in_inch) * float(spacing_factor)) / float(per_row_in_inch)
            if adaptive_title_gap and n_groups <= 2:
                cap_axis = float(spacing_factor) * float(small_n_title_gap_factor)
                pad_axis = min(pad_axis, cap_axis)
            title_pad = pad_axis
        else:
            title_pad = spacing_factor * float(title_gap_factor)
            if adaptive_title_gap and n_groups <= 2:
                cap = spacing_factor * float(small_n_title_gap_factor)
                title_pad = min(title_pad, cap)
        y_min = -0.5
        y_max = (min_slots - 1) * spacing_factor + title_pad
        table_ax.set_ylim(y_min, y_max)
        table_ax.tick_params(axis="x", labelsize=max(10, fontsize - 6))
        for spine in table_ax.spines.values():
            spine.set_visible(False)
        table_ax.tick_params(axis="y", length=0)
        _title_fs = fontsize if title_fontsize is None else int(title_fontsize)
        table_ax.set_title(
            "Number at Risk",
            loc="left",
            fontsize=_title_fs,
            fontweight="bold",
            pad=max(8, int(_title_fs * 0.6)),
        )
        for g_idx, kmf in enumerate(kmfs):
            counts = [_get_count_at_time(kmf, t, "At risk") for t in time_points]
            for t, count in zip(time_points, counts):
                count_color = colors[g_idx] if color_counts else "black"
                table_ax.text(
                    t,
                    y_positions[g_idx],
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=count_color,
                )
            lab = _wrap_label(labels[g_idx], label_wrap_chars, label_max_lines)
            label_color = (
                colors[g_idx] if (colors and g_idx < len(colors) and colors[g_idx]) else "black"
            )
            table_ax.text(
                xmin - half_bin,
                y_positions[g_idx],
                lab,
                transform=table_ax.transData,
                ha="right",
                va="center",
                fontsize=fontsize,
                color=label_color,
                fontweight="bold",
                clip_on=False,
            )
        return [table_ax], table_height + table_gap
    bbox = ax.get_position()
    n_groups = len(kmfs)
    try:
        min_slots = int(min_display_rows) if min_display_rows is not None else n_groups
    except (TypeError, ValueError):
        min_slots = n_groups
    min_slots = max(min_slots, n_groups, 1)
    n_rows = len(rows_to_show)
    total_height = table_height * n_rows
    table_axes = []
    ax.set_aspect("auto")
    gap = table_gap * 2.5
    for row_idx, row_type in enumerate(rows_to_show):
        table_pos = [
            bbox.x0,
            bbox.y0 - total_height - gap + (n_rows - row_idx - 1) * table_height,
            bbox.width,
            table_height,
        ]
        table_ax = fig.add_axes(table_pos)
        table_ax.set_xlim(xmin, xmax)
        table_ax.set_aspect("auto")
        table_ax.set_xticks([])
        table_ax.set_xticklabels([])
        table_ax.set_xlim(xmin, xmax)
        for spine in table_ax.spines.values():
            spine.set_visible(False)
        table_ax.set_ylabel("", fontsize=fontsize)
        table_ax.yaxis.set_label_position("left")
        table_ax.yaxis.set_label_coords(-0.12, 0.5)
        spacing_factor = row_spacing
        slot_positions = (np.arange(min_slots)[::-1]) * spacing_factor
        y_positions = slot_positions[:n_groups]
        table_ax.set_yticks(y_positions)
        y_min = -0.5
        title_pad = spacing_factor * float(title_gap_factor)
        if adaptive_title_gap and n_groups <= 2:
            cap = spacing_factor * float(small_n_title_gap_factor)
            title_pad = min(title_pad, cap)
        y_max = (min_slots - 1) * spacing_factor + title_pad
        table_ax.set_ylim(y_min, y_max)
        table_ax.tick_params(axis="y", which="both", left=False, labelleft=False, pad=0)
        if row_idx == 0:
            _title_fs = fontsize if title_fontsize is None else int(title_fontsize)
            table_ax.set_title(
                "Number at Risk",
                loc="left",
                fontsize=_title_fs,
                fontweight="bold",
                pad=max(8, int(_title_fs * 0.6)),
            )
        for g_idx, (kmf, label) in enumerate(zip(kmfs, labels)):
            y_pos = y_positions[g_idx]
            if time_points.size > 1:
                risk_table_label_distance = (time_points[1] - time_points[0]) / 2 + (
                    time_points[1] - time_points[0]
                ) / 4
            else:
                risk_table_label_distance = (xmax - xmin) * 0.05
            label_color = (
                colors[g_idx] if (colors and g_idx < len(colors) and colors[g_idx]) else "black"
            )
            table_ax.text(
                xmin - risk_table_label_distance,
                y_pos,
                label,
                ha="right",
                va="center",
                fontsize=fontsize,
                color=label_color,
                fontweight="bold",
                transform=table_ax.transData,
            )
            counts = [_get_count_at_time(kmf, t, row_type) for t in time_points]
            for t_idx, (t, count) in enumerate(zip(time_points, counts)):
                count_color = colors[g_idx] if color_counts else "black"
                table_ax.text(
                    t,
                    y_pos,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=count_color,
                )
        table_ax.grid(False)
        table_axes.append(table_ax)
    return table_axes, total_height + table_gap
