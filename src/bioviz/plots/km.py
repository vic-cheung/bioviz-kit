"""Kaplan-Meier survival plotter for bioviz-kit.

Provides a high-level interface for generating KM survival curves with optional
risk tables using lifelines. Font sizes default to None to inherit from rcParams.

Example
-------
>>> from bioviz.configs import KMPlotConfig
>>> from bioviz.plots import KMPlotter
>>> cfg = KMPlotConfig(time_col="PFS_M", event_col="EVENT", group_col="ARM")
>>> plotter = KMPlotter(df, cfg)
>>> fig, ax, pvalue = plotter.plot()
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from ..configs.km_cfg import KMPlotConfig

__all__ = [
    "KMPlotter",
    "format_pvalue",
    "add_pvalue_annotation",
    "expand_canvas",
    "expand_figure_to_fit_artists",
]


# =============================================================================
# Canvas Expansion Utilities
# =============================================================================
def expand_canvas(
    fig,
    left_in: float = 0.0,
    right_in: float = 0.0,
    top_in: float = 0.0,
    bottom_in: float = 0.0,
) -> None:
    """Expand figure canvas by the given inches on each side without squeezing axes.

    Axes retain their physical size; we shift them to account for new margins.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to resize.
    left_in, right_in, top_in, bottom_in : float
        Inches to add on each side.
    """
    if fig is None:
        return
    add_w = max(0.0, float(left_in) + float(right_in))
    add_h = max(0.0, float(top_in) + float(bottom_in))
    if add_w == 0 and add_h == 0:
        return

    # Current and new sizes (inches)
    w, h = fig.get_size_inches()
    new_w = w + float(left_in) + float(right_in)
    new_h = h + float(top_in) + float(bottom_in)

    # Update axes positions to preserve physical sizes and shift by new margins
    for ax in list(fig.axes):
        bbox = ax.get_position()
        # Convert to inches
        x0_in = bbox.x0 * w
        y0_in = bbox.y0 * h
        width_in = bbox.width * w
        height_in = bbox.height * h
        # Shift by added margins
        x0_in_new = x0_in + float(left_in)
        y0_in_new = y0_in + float(bottom_in)
        # Convert back to figure coords of new canvas
        x0_new = x0_in_new / new_w
        y0_new = y0_in_new / new_h
        width_new = width_in / new_w
        height_new = height_in / new_h
        ax.set_position([x0_new, y0_new, width_new, height_new])

    # Finally, grow the figure
    fig.set_size_inches(new_w, new_h, forward=True)


def expand_figure_to_fit_artists(
    fig,
    artists,
    pad_left_in: float = 0.25,
    pad_right_in: float = 0.25,
    pad_top_in: float = 0.0,
    pad_bottom_in: float = 0.0,
) -> None:
    """Expand canvas to ensure a set of artists fit within the figure bounds.

    Computes overhang on each side for provided artists and expands canvas just enough
    to bring them into view, adding specified padding.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    artists : list[Artist]
        Artists to fit (e.g., legend, text labels).
    pad_left_in, pad_right_in, pad_top_in, pad_bottom_in : float
        Extra padding (inches) to add on respective sides once the artist fits.
    """
    if fig is None or not artists:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return

    try:
        fig_px = fig.bbox
        dpi = fig.dpi
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
            # Overhangs: positive if beyond figure on that side
            left_over_px = max(left_over_px, float(fig_px.x0 - bb.x0))
            right_over_px = max(right_over_px, float(bb.x1 - fig_px.x1))
            bottom_over_px = max(bottom_over_px, float(fig_px.y0 - bb.y0))
            top_over_px = max(top_over_px, float(bb.y1 - fig_px.y1))

        # Convert pixels to inches and add padding
        left_in = max(0.0, left_over_px / dpi) + pad_left_in
        right_in = max(0.0, right_over_px / dpi) + pad_right_in
        top_in = max(0.0, top_over_px / dpi) + pad_top_in
        bottom_in = max(0.0, bottom_over_px / dpi) + pad_bottom_in

        # Only expand if needed
        if left_in > 0 or right_in > 0 or top_in > 0 or bottom_in > 0:
            expand_canvas(
                fig, left_in=left_in, right_in=right_in, top_in=top_in, bottom_in=bottom_in
            )
    except Exception:
        pass


# =============================================================================
# Helpers
# =============================================================================
def _resolve_fontsize(config_value: int | None, rcparam_key: str) -> int | float:
    """Return config value if set, else fall back to rcParams."""
    if config_value is not None:
        return config_value
    return plt.rcParams.get(rcparam_key, 12)


def _wrap_label(label: str, wrap_chars: int | None, max_lines: int = 2) -> str:
    """Wrap label at `wrap_chars` and truncate to `max_lines`."""
    if wrap_chars is None or wrap_chars <= 0:
        return label
    lines = textwrap.wrap(label, width=wrap_chars)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "…"
    return "\n".join(lines)


def _wrap_labels(labels: List[str], wrap_chars: int | None, max_lines: int = 2) -> List[str]:
    """Apply label wrapping to a list."""
    return [_wrap_label(lbl, wrap_chars, max_lines) for lbl in labels]


def format_pvalue(p_value: float, significance_cutoffs: Dict[float, str] | None = None) -> str:
    """Format p-value with appropriate precision and notation.

    Parameters
    ----------
    p_value : float
        The p-value to format.
    significance_cutoffs : dict, optional
        Mapping of cutoffs to format strings.

    Returns
    -------
    str
        Formatted p-value string.
    """
    if significance_cutoffs is None:
        significance_cutoffs = {
            0.001: "p < 0.001",
            0.01: "p = {:.3f}",
            0.05: "p = {:.3f}",
            1.0: "p = {:.2f}",
        }
    for cutoff, fmt in sorted(significance_cutoffs.items()):
        if p_value < cutoff:
            return fmt if "< " in fmt else fmt.format(p_value)
    return f"p = {p_value:.2f}"


def add_pvalue_annotation(
    ax,
    p_value: float,
    loc: str = "bottom_right",
    box: bool = True,
    fontsize: int | float = 12,
    alpha: float = 0.8,
    format_p: bool = True,
):
    """Add p-value annotation to an axes.

    Parameters
    ----------
    ax : Axes
        The axes to annotate.
    p_value : float
        The p-value to display.
    loc : str
        Location: 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center_right'.
    box : bool
        Whether to draw a box around the annotation.
    fontsize : int | float
        Font size for annotation.
    alpha : float
        Background transparency.
    format_p : bool
        Whether to format using format_pvalue().

    Returns
    -------
    Text or None
    """
    if p_value is None:
        return None
    p_text = format_pvalue(p_value) if format_p else f"p = {p_value:.4f}"
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
    bbox_props = (
        dict(boxstyle="round,pad=0.5", facecolor="white", alpha=alpha, edgecolor="lightgray")
        if box
        else None
    )
    return ax.annotate(
        p_text, xy=xy, xycoords="axes fraction", ha=ha, va=va, fontsize=fontsize, bbox=bbox_props
    )


# =============================================================================
# KMPlotter
# =============================================================================
class KMPlotter:
    """Generate Kaplan-Meier survival plots with optional risk tables.

    Parameters
    ----------
    data : pd.DataFrame
        Survival data with time, event, and group columns.
    config : KMPlotConfig
        Configuration object specifying plot options.

    Attributes
    ----------
    data : pd.DataFrame
    config : KMPlotConfig
    """

    def __init__(self, data: pd.DataFrame, config: KMPlotConfig) -> None:
        self.data = data
        self.config = config

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_groups(self) -> List[Any]:
        """Return ordered list of groups from data.

        Priority:
        1. config.group_order if explicitly provided
        2. pd.Categorical categories if column is categorical
        3. Unique values in data order
        """
        col = self.data[self.config.group_col]
        observed = set(col.dropna().unique())

        # 1. Explicit group_order from config
        if self.config.group_order is not None:
            return [g for g in self.config.group_order if g in observed]

        # 2. pd.Categorical order
        if isinstance(col.dtype, pd.CategoricalDtype):
            all_cats = col.cat.categories
            return [cat for cat in all_cats if cat in observed]

        # 3. Data order
        return list(col.unique())

    def _fit_kmf(
        self,
        durations: List,
        events: List,
        label: str,
        timeline: Iterable[float] | None = None,
    ) -> KaplanMeierFitter:
        """Fit a KaplanMeierFitter with optional linear CI transformation."""
        kmf = KaplanMeierFitter()
        kmf.fit(durations, events, label=label)
        # Store original data for per-patient censor markers
        kmf.original_durations = durations
        kmf.original_events = events
        if timeline is not None:
            kmf.timeline_for_risktable_ = list(timeline)
        # Optionally compute linear CIs
        if self.config.conf_type == "linear":
            self._add_linear_ci(kmf)
        return kmf

    def _add_linear_ci(self, kmf: KaplanMeierFitter) -> None:
        """Compute symmetric linear confidence intervals on the KMF."""
        se = kmf.confidence_interval_.iloc[:, 1] - kmf.survival_function_.iloc[:, 0]
        se /= 1.96
        lower = (kmf.survival_function_.iloc[:, 0] - 1.96 * se).clip(0, 1)
        upper = (kmf.survival_function_.iloc[:, 0] + 1.96 * se).clip(0, 1)
        kmf.confidence_interval_ = pd.DataFrame(
            {"linear_lower_0.95": lower.values, "linear_upper_0.95": upper.values},
            index=kmf.timeline,
        )

    def _plot_single_km(
        self,
        kmf: KaplanMeierFitter,
        ax,
        color: str,
    ) -> None:
        """Plot a single KM curve with CI and censor markers."""
        cfg = self.config
        x_vals = kmf.survival_function_.index.to_numpy()
        y_vals = kmf.survival_function_.iloc[:, 0].to_numpy()
        label = getattr(kmf, "_label", None)

        # Plot survival curve
        if x_vals.size <= 1:
            xmin, xmax = ax.get_xlim()
            if xmax <= xmin:
                xmin, xmax = 0.0, 1.0
            ax.hlines(
                y=y_vals[0] if y_vals.size else 1.0,
                xmin=xmin,
                xmax=xmax,
                colors=color,
                linewidth=cfg.linewidth,
                linestyles=cfg.linestyle,
                label=label,
            )
        else:
            ax.step(
                x_vals,
                y_vals,
                where="post",
                color=color,
                linewidth=cfg.linewidth,
                linestyle=cfg.linestyle,
                label=label,
            )

        # Confidence intervals
        if cfg.show_ci and hasattr(kmf, "confidence_interval_"):
            self._plot_ci(kmf, ax, color)

        # Censor markers
        self._plot_censors(kmf, ax, color)

    def _plot_ci(self, kmf: KaplanMeierFitter, ax, color: str) -> None:
        """Plot confidence intervals for a KM curve."""
        cfg = self.config
        ci_cols = kmf.confidence_interval_.columns
        lower_col = upper_col = None
        if cfg.conf_type == "linear" and "linear_lower_0.95" in ci_cols:
            lower_col, upper_col = "linear_lower_0.95", "linear_upper_0.95"
        elif len(ci_cols) >= 2:
            lower_col = next(c for c in ci_cols if "lower" in c.lower())
            upper_col = next(c for c in ci_cols if "upper" in c.lower())

        if lower_col and upper_col:
            lower = kmf.confidence_interval_[lower_col].to_numpy()
            upper = kmf.confidence_interval_[upper_col].to_numpy()
            ci_x = kmf.confidence_interval_.index.to_numpy()
            if cfg.ci_style == "fill":
                ax.fill_between(
                    ci_x, lower, upper, alpha=cfg.ci_alpha, step="post", color=color, linewidth=0
                )
            elif cfg.ci_style == "lines":
                ax.step(
                    ci_x,
                    lower,
                    where="post",
                    color=color,
                    linewidth=cfg.linewidth / 2,
                    alpha=0.7,
                    linestyle="--",
                )
                ax.step(
                    ci_x,
                    upper,
                    where="post",
                    color=color,
                    linewidth=cfg.linewidth / 2,
                    alpha=0.7,
                    linestyle="--",
                )

    def _plot_censors(self, kmf: KaplanMeierFitter, ax, color: str) -> None:
        """Add censor markers for a KM curve."""
        cfg = self.config
        plotted = False

        # Per-patient markers
        if cfg.per_patient_censor_markers and hasattr(kmf, "original_durations"):
            try:
                pp_times = [
                    t for t, e in zip(kmf.original_durations, kmf.original_events) if int(e) == 0
                ]
            except Exception:
                pp_times = []
            if pp_times:
                surv = kmf.predict(np.array(pp_times))
                surv_vals = surv.values if hasattr(surv, "values") else np.asarray(surv)
                ax.scatter(
                    pp_times,
                    surv_vals,
                    marker=cfg.censor_marker,
                    s=cfg.censor_markersize**2,
                    linewidths=cfg.censor_markeredgewidth,
                    color=color,
                    zorder=11,
                )
                plotted = True

        # Fallback: unique censor times from event table
        if not plotted:
            try:
                et = kmf.event_table
                cens_col = et.get("censored")
                if cens_col is not None:
                    mask = cens_col > 0
                    if np.any(mask):
                        cens_times = et.index.values[mask]
                        surv = kmf.predict(cens_times)
                        surv_vals = surv.values if hasattr(surv, "values") else np.asarray(surv)
                        ax.scatter(
                            cens_times,
                            surv_vals,
                            marker=cfg.censor_marker,
                            s=cfg.censor_markersize**2,
                            linewidths=cfg.censor_markeredgewidth,
                            color=color,
                            zorder=11,
                        )
                        plotted = True
            except Exception:
                pass

        # Force single reference marker
        if not plotted and cfg.force_show_censors and len(kmf.timeline) > 2:
            idx = len(kmf.timeline) // 2
            t = kmf.timeline[idx]
            prob = kmf.survival_function_.iloc[idx].values[0]
            ax.scatter(
                [t],
                [prob],
                marker=cfg.censor_marker,
                s=cfg.censor_markersize**2,
                linewidths=cfg.censor_markeredgewidth,
                color=color,
                zorder=11,
            )

    def _compute_xticks(self, ax) -> List[float] | None:
        """Determine x-tick positions from config or data."""
        cfg = self.config
        if cfg.xticks is not None:
            return list(cfg.xticks)
        xmin, xmax = ax.get_xlim()
        if cfg.xtick_interval_months is not None:
            interval = cfg.xtick_interval_months
            snapped = interval * np.ceil(xmax / interval)
            ax.set_xlim(xmin, snapped)
            return list(np.arange(0.0, snapped + 1e-9, interval))
        return None

    def _position_legend(self, ax, fontsize: int | float) -> Optional[Any]:
        """Create and position the legend."""
        cfg = self.config
        handles, labels = ax.get_legend_handles_labels()
        if cfg.legend_label_wrap_chars:
            labels = _wrap_labels(labels, cfg.legend_label_wrap_chars, cfg.legend_label_max_lines)
        if not handles:
            return None
        loc = cfg.legend_loc
        ncol = max(1, min(len(handles), 3)) if loc == "bottom" else 1
        kwargs = dict(
            fontsize=fontsize,
            frameon=cfg.legend_frameon,
            title=cfg.legend_title,
            title_fontsize=fontsize + 2,
            ncol=ncol,
            markerscale=cfg.legend_markerscale,
        )
        if loc == "bottom":
            kwargs.update(loc="upper center", bbox_to_anchor=(0.5, -0.3))
        elif loc == "right":
            kwargs.update(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0)
        else:
            kwargs.update(loc=loc)
        legend = ax.legend(handles, labels, **kwargs)
        if legend and cfg.legend_title_fontweight:
            try:
                legend.get_title().set_fontweight(cfg.legend_title_fontweight)
            except Exception:
                pass
        if legend and cfg.legend_linewidth_scale:
            try:
                for ln in legend.get_lines():
                    ln.set_linewidth(ln.get_linewidth() * cfg.legend_linewidth_scale)
            except Exception:
                pass
        return legend

    def _add_risktable(
        self,
        ax,
        table_ax,
        kmfs: List[KaplanMeierFitter],
        labels: List[str],
        colors: List[str],
        xticks: List[float] | None,
    ) -> None:
        """Populate a risk table axes with counts at each time point."""
        cfg = self.config
        fontsize = _resolve_fontsize(cfg.risktable_fontsize, "font.size")
        title_fontsize = (
            cfg.risktable_title_fontsize if cfg.risktable_title_fontsize else int(fontsize) + 2
        )

        # Clear table axes
        table_ax.cla()

        xmin, xmax = ax.get_xlim()
        table_ax.set_xlim(xmin, xmax)

        # Determine time points
        if xticks is not None:
            time_points = np.array(xticks, dtype=float)
        else:
            ticks = ax.get_xticks()
            time_points = np.array([t for t in ticks if xmin - 1e-9 <= t <= xmax + 1e-9])

        n_groups = len(kmfs)
        spacing = cfg.risktable_row_spacing
        slot_positions = (np.arange(n_groups)[::-1]) * spacing
        y_positions = slot_positions[:n_groups]
        table_ax.set_yticks(y_positions)
        table_ax.set_yticklabels([])

        title_pad = spacing * cfg.risktable_title_gap_factor
        y_min, y_max = -0.5, (n_groups - 1) * spacing + title_pad
        table_ax.set_ylim(y_min, y_max)

        # Hide spines and y ticks
        for spine in table_ax.spines.values():
            spine.set_visible(False)
        table_ax.tick_params(axis="y", length=0)
        table_ax.tick_params(axis="x", labelsize=max(10, fontsize - 6))
        table_ax.set_title(
            "Number at Risk",
            loc="left",
            fontsize=title_fontsize,
            fontweight="bold",
            pad=max(8, int(title_fontsize * 0.6)),
        )

        # Compute bin width for label offset
        if len(time_points) > 1:
            bin_width = float(time_points[1] - time_points[0])
        else:
            bin_width = max((xmax - xmin) * 0.1, 1e-6)
        half_bin = 0.5 * bin_width

        for g_idx, kmf in enumerate(kmfs):
            counts = [self._get_count_at_time(kmf, t) for t in time_points]
            for t, count in zip(time_points, counts):
                color = colors[g_idx] if cfg.color_risktable_counts else "black"
                table_ax.text(
                    t,
                    y_positions[g_idx],
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=color,
                )
            # Group label
            lab = _wrap_label(
                labels[g_idx], cfg.risktable_label_wrap_chars, cfg.risktable_label_max_lines
            )
            label_color = colors[g_idx] if colors else "black"
            table_ax.text(
                xmin - half_bin,
                y_positions[g_idx],
                lab,
                ha="right",
                va="center",
                fontsize=fontsize,
                color=label_color,
                fontweight="bold",
                clip_on=False,
            )

    def _get_count_at_time(self, kmf: KaplanMeierFitter, t: float) -> int:
        """Return 'at risk' count at time t."""
        if t == 0:
            return int(kmf.event_table.at_risk.iloc[0])
        et = kmf.event_table
        times = et.index.values
        idx = np.searchsorted(times, t)
        if idx >= len(times):
            return 0
        if idx > 0 and (idx == len(times) or t < times[idx]):
            idx -= 1
        at_risk = et.at_risk.iloc[idx]
        removed = et.removed.iloc[idx]
        return int(at_risk - removed)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def plot(
        self,
        ax=None,
        fig=None,
        output_path: str | None = None,
    ) -> Tuple[Any, Any, float | None]:
        """Generate the Kaplan-Meier plot.

        Parameters
        ----------
        ax : Axes, optional
            Existing axes; if None, a new figure/axes is created.
        fig : Figure, optional
            Existing figure.
        output_path : str, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
        ax : Axes
        p_value : float or None
            Log-rank p-value if computed.
        """
        cfg = self.config
        data = self.data
        groups = self._get_groups()

        # Font sizes
        label_fs = _resolve_fontsize(cfg.label_fontsize, "axes.labelsize")
        title_fs = _resolve_fontsize(cfg.title_fontsize, "axes.titlesize")
        legend_fs = _resolve_fontsize(cfg.legend_fontsize, "legend.fontsize")
        pval_fs = _resolve_fontsize(cfg.pvalue_fontsize, "font.size")
        risktable_fs = _resolve_fontsize(cfg.risktable_fontsize, "font.size")

        # Compute layout dimensions
        n_groups = max(1, len(groups))
        risktable_min = max(cfg.risktable_min_rows, n_groups)
        per_row_in = max(0.26, (risktable_fs / 72.0) * cfg.risktable_row_spacing)
        title_pad_in = max(0.25, (risktable_fs / 72.0) * cfg.risktable_title_gap_factor * 0.7)

        table_ax = None
        if ax is None or fig is None:
            fig_w, fig_h = cfg.get_figsize()
            if cfg.show_risktable:
                rt_height = max(1.1, risktable_min * per_row_in + title_pad_in)
                # Minimum gap to fit xlabel; allow user override via risktable_hspace
                min_gap_in = (label_fs / 72.0) * 1.1
                # Use user-specified hspace if provided (0 means minimal gap)
                # Only enforce 0.5" minimum if neither is specified
                if cfg.risktable_hspace is not None and cfg.risktable_hspace >= 0:
                    gap_in = max(min_gap_in, cfg.risktable_hspace)
                else:
                    gap_in = max(0.5, min_gap_in)
                total_h = fig_h + gap_in + rt_height

                height_ratios = [fig_h, gap_in, rt_height]
                fig = plt.figure(figsize=(fig_w, total_h))
                subfigs = fig.subfigures(3, 1, height_ratios=height_ratios, hspace=0.0)
                ax = subfigs[0].add_subplot(111)
                # Spacer subfig - make it invisible
                try:
                    spacer_ax = subfigs[1].add_subplot(111)
                    spacer_ax.set_visible(False)
                    for spine in spacer_ax.spines.values():
                        spine.set_visible(False)
                    spacer_ax.set_xticks([])
                    spacer_ax.set_yticks([])
                except Exception:
                    pass
                # Risk table with sharex for alignment
                table_ax = subfigs[2].add_subplot(111, sharex=ax)
            else:
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Colors
        custom_colors = dict(cfg.color_dict) if cfg.color_dict else {}
        default_colors = plt.cm.tab10.colors
        for i, g in enumerate(groups):
            if g not in custom_colors:
                custom_colors[g] = default_colors[i % len(default_colors)]

        # Fit and plot each group
        kmfs: List[KaplanMeierFitter] = []
        risktable_labels: List[str] = []
        legend_overrides = dict(cfg.legend_label_overrides or {})
        risktable_overrides = dict(cfg.risktable_label_overrides or {})

        for g in groups:
            mask = data[cfg.group_col] == g
            if mask.sum() == 0:
                continue
            durations = data.loc[mask, cfg.time_col].tolist()
            events = data.loc[mask, cfg.event_col].tolist()
            raw_label = str(g)
            legend_label = f"{raw_label} (n={mask.sum()})" if cfg.legend_show_n else raw_label
            for cand in (g, raw_label):
                if cand in legend_overrides:
                    legend_label = legend_overrides[cand]
                    break
            risk_label = raw_label
            for cand in (g, raw_label):
                if cand in risktable_overrides:
                    risk_label = risktable_overrides[cand]
                    break

            kmf = self._fit_kmf(durations, events, legend_label, cfg.timeline)
            self._plot_single_km(kmf, ax, custom_colors[g])
            kmfs.append(kmf)
            risktable_labels.append(risk_label)

        if not kmfs:
            return fig, ax, None

        # P-value
        p_value = None
        pval_text = None
        if len(groups) >= 2 and cfg.show_pvalue:
            try:
                g0, g1 = groups[0], groups[1]
                m0 = data[cfg.group_col] == g0
                m1 = data[cfg.group_col] == g1
                res = logrank_test(
                    data.loc[m0, cfg.time_col],
                    data.loc[m1, cfg.time_col],
                    data.loc[m0, cfg.event_col],
                    data.loc[m1, cfg.event_col],
                )
                p_value = res.p_value
                pval_text = add_pvalue_annotation(
                    ax, p_value, loc=cfg.pval_loc, box=cfg.pvalue_box, fontsize=pval_fs
                )
            except Exception:
                pass

        # Axis labels/title
        # Keep xlabel on KM plot, clear it on risk table
        ax.set_xlabel(cfg.get_xlabel(), fontsize=label_fs, fontweight="bold")
        ax.set_ylabel(cfg.get_ylabel(), fontsize=label_fs, fontweight="bold")
        if cfg.title:
            ax.set_title(cfg.title, fontsize=title_fs, fontweight=cfg.title_fontweight, loc="left")

        # Limits
        if cfg.xlim:
            ax.set_xlim(cfg.xlim)
        elif cfg.timeline is not None:
            max_t = max(cfg.timeline)
            if cfg.xtick_interval_months:
                snapped = cfg.xtick_interval_months * np.ceil(max_t / cfg.xtick_interval_months)
                ax.set_xlim(0.0, snapped)
            else:
                ax.set_xlim(0.0, max_t)
        else:
            max_t = data[cfg.time_col].max()
            ax.set_xlim(0, max_t + 0.05 * max_t)
        ax.set_ylim(cfg.ylim)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # X-ticks
        xticks_final = self._compute_xticks(ax)
        if xticks_final:
            ax.set_xticks(xticks_final)
            ax.set_xticklabels(
                [str(int(t)) if float(t).is_integer() else str(t) for t in xticks_final]
            )

        # Legend
        legend = self._position_legend(ax, legend_fs)

        # Risk table
        if cfg.show_risktable and table_ax is not None and kmfs:
            self._add_risktable(
                ax,
                table_ax,
                kmfs,
                risktable_labels,
                [custom_colors[g] for g in groups if g in custom_colors],
                xticks_final,
            )
            # Ensure KM plot x-tick labels stay visible
            ax.tick_params(axis="x", which="both", labelbottom=True, bottom=True)
            for lab in ax.get_xticklabels():
                lab.set_visible(True)
            # Hide all tick labels on table_ax
            if xticks_final is not None:
                table_ax.set_xticks(xticks_final)
            table_ax.tick_params(
                axis="x",
                which="both",
                labelbottom=False,
                bottom=False,
                length=0,
            )
            for lab in table_ax.get_xticklabels():
                lab.set_visible(False)
            table_ax.set_xlabel("")  # Clear xlabel on risk table

            # Expand canvas to fit all artists (legend, p-value, risk table labels)
            artists = []
            if legend is not None:
                artists.append(legend)
            if pval_text is not None:
                artists.append(pval_text)
            # Include y-axis label and ticks
            try:
                artists.append(ax.yaxis.get_label())
                for ytick in ax.get_yticklabels():
                    artists.append(ytick)
            except Exception:
                pass
            # Include risk table text elements (group labels and counts)
            try:
                for child in table_ax.get_children():
                    if hasattr(child, "get_text") and child.get_text():
                        artists.append(child)
            except Exception:
                pass
            if artists:
                expand_figure_to_fit_artists(
                    fig,
                    artists,
                    pad_left_in=0.3,
                    pad_right_in=0.3,
                    pad_top_in=0.2,
                    pad_bottom_in=0.2,
                )
            # Always expand canvas for risk table plots
            try:
                fig.canvas.draw()
                expand_canvas(fig, left_in=0.3, bottom_in=0.2, right_in=0.3, top_in=0.2)
            except Exception:
                pass

        # Save
        if output_path and fig:
            try:
                fig.canvas.draw()
            except Exception:
                pass
            save_kwargs = {"dpi": 300}
            # Only use bbox_inches="tight" when NOT showing risk table
            # (tight layout can cause artifacts with subfigures)
            if cfg.save_bbox_inches and not cfg.show_risktable:
                save_kwargs["bbox_inches"] = cfg.save_bbox_inches
            fig.savefig(output_path, **save_kwargs)

        return fig, ax, p_value
