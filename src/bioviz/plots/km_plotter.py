from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from bioviz.configs.km_cfg import KMPlotConfig
from bioviz.plots import km as bk


class KMPlotter:
    """Driver for Kaplan-Meier plotting.

    Supports two usage patterns:
    - Pure-plot mode: create with a list of pre-fitted `KaplanMeierFitter` and call
      `plot_from_kmfs(config)`.
    - Convenience mode: use `from_dataframe(data, config)` which fits KMFs and
      returns an instance ready to `plot()`.
    """

    def __init__(self, kmfs: Iterable[KaplanMeierFitter], labels: Optional[Iterable[str]] = None):
        self.kmfs = list(kmfs)
        self.labels = (
            list(labels) if labels is not None else [getattr(k, "_label", None) for k in self.kmfs]
        )

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, config: KMPlotConfig):
        """Compatibility constructor that fits KMFs from `data` using `config`.

        Returns a KMPlotter instance containing fitted KMFs.
        """
        kwargs = config.to_kwargs()
        time_col = kwargs["time_col"]
        event_col = kwargs["event_col"]
        group_col = kwargs["group_col"]

        if isinstance(data[group_col].dtype, pd.CategoricalDtype):
            all_cats = data[group_col].cat.categories
            groups = [cat for cat in all_cats if cat in data[group_col].values]
        else:
            groups = list(pd.Series(data[group_col].unique()))

        kmfs = []
        labels = []
        legend_show_n = kwargs.get("legend_show_n", False)
        legend_label_overrides_raw = kwargs.get("legend_label_overrides") or {}
        legend_label_overrides = (
            dict(legend_label_overrides_raw) if isinstance(legend_label_overrides_raw, dict) else {}
        )

        for g in groups:
            mask = data[group_col] == g
            if mask.sum() == 0:
                continue
            kmf = KaplanMeierFitter()
            raw_label = str(g)
            label_val = f"{raw_label} (n={int(mask.sum())})" if legend_show_n else raw_label
            for candidate in (g, raw_label):
                if candidate in legend_label_overrides:
                    label_val = legend_label_overrides[candidate]
                    break
            durations = data.loc[mask, time_col].values.tolist()
            events = data.loc[mask, event_col].values.tolist()
            kmf.fit(durations, events, label=label_val)
            kmf.original_durations = durations
            kmf.original_events = events
            if kwargs.get("timeline") is not None:
                kmf.timeline_for_risktable_ = kwargs.get("timeline")
            kmfs.append(kmf)
            labels.append(raw_label)

        return cls(kmfs=kmfs, labels=labels)

    def plot_from_kmfs(
        self,
        config: KMPlotConfig,
        output_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes, Optional[float]]:
        """Render a Kaplan-Meier figure from pre-fitted `KaplanMeierFitter` objects.

        Uses plotting helpers in `bioviz.plots.km` and preserves legacy config options.
        """
        kwargs = config.to_kwargs()

        # extract plotting options (keep legacy defaults where possible)
        title = kwargs.get("title")
        color_dict = kwargs.get("color_dict") or {}
        fig_width = kwargs.get("fig_width", 10)
        fig_height = kwargs.get("fig_height", 6)
        xlim = kwargs.get("xlim")
        ylim = kwargs.get("ylim", (0, 1.0))
        xlab = kwargs.get("xlab", "Time (Months)")
        ylab = kwargs.get("ylab", "Survival Probability")
        legend_loc = kwargs.get("legend_loc", "bottom")
        legend_title = kwargs.get("legend_title")
        legend_title_fontweight = kwargs.get("legend_title_fontweight")
        legend_fontsize = kwargs.get("legend_fontsize", 16)
        legend_frameon = kwargs.get("legend_frameon", False)
        legend_markerscale = kwargs.get("legend_markerscale", 1.0)
        legend_linewidth_scale = kwargs.get("legend_linewidth_scale")
        pval_loc = kwargs.get("pval_loc", "top_right")
        pvalue_fontsize = kwargs.get("pvalue_fontsize", 18)
        pvalue_box = kwargs.get("pvalue_box", False)
        show_risktable = kwargs.get("show_risktable", True)
        show_pvalue = kwargs.get("show_pvalue", True)
        ci_style = kwargs.get("ci_style", "fill")
        ci_alpha = kwargs.get("ci_alpha", 0.25)
        show_ci = kwargs.get("show_ci", True)
        linewidth = kwargs.get("linewidth", 3.0)
        linestyle = kwargs.get("linestyle", "-")
        censor_marker = kwargs.get("censor_marker", "+")
        censor_markersize = kwargs.get("censor_markersize", 12.0)
        censor_markeredgewidth = kwargs.get("censor_markeredgewidth", 2.5)
        force_show_censors = kwargs.get("force_show_censors", True)
        xticks = kwargs.get("xticks")
        timeline = kwargs.get("timeline")
        risktable_fontsize = kwargs.get("risktable_fontsize", 18)
        risktable_row_spacing = kwargs.get("risktable_row_spacing", 1.8)
        risktable_title_gap_factor = kwargs.get("risktable_title_gap_factor", 0.6)
        legend_label_wrap_chars = kwargs.get("legend_label_wrap_chars")
        legend_label_max_lines = kwargs.get("legend_label_max_lines", 2)
        risktable_label_wrap_chars = kwargs.get("risktable_label_wrap_chars")
        risktable_label_max_lines = kwargs.get("risktable_label_max_lines", 2)
        label_fontsize = kwargs.get("label_fontsize", 18)
        title_fontsize = kwargs.get("title_fontsize", 20)
        risktable_title_fontsize = kwargs.get("risktable_title_fontsize", 20)
        color_risktable_counts = kwargs.get("color_risktable_counts", False)
        conf_type = kwargs.get("conf_type", "log_log")
        per_patient_censor_markers = kwargs.get("per_patient_censor_markers", True)
        auto_expand_for_legend = kwargs.get("auto_expand_for_legend", False)
        save_bbox_inches = kwargs.get("save_bbox_inches", "tight")
        xtick_interval_months = kwargs.get("xtick_interval_months", 3.0)
        risktable_hspace = kwargs.get("risktable_hspace", 0.5)
        legend_show_n = kwargs.get("legend_show_n", False)
        legend_label_overrides_raw = kwargs.get("legend_label_overrides") or {}
        risktable_label_overrides_raw = kwargs.get("risktable_label_overrides") or {}
        legend_label_overrides = (
            dict(legend_label_overrides_raw) if isinstance(legend_label_overrides_raw, dict) else {}
        )
        risktable_label_overrides = (
            dict(risktable_label_overrides_raw)
            if isinstance(risktable_label_overrides_raw, dict)
            else {}
        )
        raw_risktable_min_rows = kwargs.get("risktable_min_rows", 4)
        try:
            risktable_min_rows = int(raw_risktable_min_rows)
        except (TypeError, ValueError):
            risktable_min_rows = 1
        if risktable_min_rows < 1:
            risktable_min_rows = 1

        # prepare figure and axes
        table_ax = None
        if show_risktable:
            effective_rows = max(risktable_min_rows, max(1, len(self.kmfs)))
            per_row_in = max(
                0.26, (float(risktable_fontsize) / 72.0) * float(risktable_row_spacing)
            )
            title_pad_in = max(
                0.25, (float(risktable_fontsize) / 72.0) * float(risktable_title_gap_factor) * 0.7
            )
            rt_height_in = max(1.1, effective_rows * per_row_in + title_pad_in)
            min_gap_in = (float(label_fontsize) / 72.0) * 1.1
            gap_in = max(0.5, float(risktable_hspace), min_gap_in)
            fig_h_total = float(fig_height) + float(gap_in) + float(rt_height_in)
            height_ratios = [float(fig_height), float(gap_in), float(rt_height_in)]

            fig = plt.figure(figsize=(float(fig_width), fig_h_total))
            subfigs = fig.subfigures(3, 1, height_ratios=height_ratios, hspace=0.0)
            km_subfig = subfigs[0]
            spacer_subfig = subfigs[1]
            rt_subfig = subfigs[2]
            ax = km_subfig.add_subplot(111)
            try:
                spacer_ax = spacer_subfig.add_subplot(111)
                spacer_ax.set_visible(False)
                for spine in spacer_ax.spines.values():
                    spine.set_visible(False)
                spacer_ax.set_xticks([])
                spacer_ax.set_yticks([])
            except Exception:
                pass
            table_ax = rt_subfig.add_subplot(111, sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))

        # determine colors
        custom_colors = color_dict or {}
        if len(custom_colors) < len(self.kmfs):
            default_colors = plt.cm.tab10.colors
            for i, lab in enumerate(self.labels):
                if lab not in custom_colors:
                    custom_colors[lab] = default_colors[i % len(default_colors)]

        # draw each KMF
        table_labels = []
        for kmf, raw_label in zip(self.kmfs, self.labels):
            bk.plot_custom_km(
                kmf,
                ax,
                color=custom_colors.get(raw_label),
                show_ci=show_ci,
                ci_alpha=ci_alpha,
                ci_style=ci_style,
                linewidth=linewidth,
                linestyle=linestyle,
                censor_marker=censor_marker,
                censor_markersize=censor_markersize,
                censor_markeredgewidth=censor_markeredgewidth,
                force_show_censors=force_show_censors,
                conf_type=conf_type,
                per_patient_censor_markers=per_patient_censor_markers,
            )
            table_labels.append(raw_label)

        # p-value (best-effort; requires original data or a two-group fallback)
        p_value = None
        pval_text = None
        if show_pvalue:
            try:
                # attempt tm_modeling helper if available
                from ..survival.modules.survival_analysis import compute_logrank_pvalue

                # compute_logrank_pvalue expects data; skip here unless caller used from_dataframe
            except Exception:
                # fallback: if exactly two groups and kmfs contain original data, use logrank_test
                if len(self.kmfs) == 2 and hasattr(self.kmfs[0], "original_durations"):
                    a = self.kmfs[0]
                    b = self.kmfs[1]
                    try:
                        results = logrank_test(
                            a.original_durations,
                            b.original_durations,
                            a.original_events,
                            b.original_events,
                        )
                        p_value = results.p_value
                        if p_value is not None:
                            pval_text = bk.add_pvalue_annotation(
                                ax,
                                p_value,
                                loc=pval_loc,
                                box=bool(pvalue_box),
                                fontsize=int(pvalue_fontsize),
                            )
                    except Exception:
                        p_value = None

        # labels, titles, limits
        try:
            ax.set_xlabel(xlab, fontsize=label_fontsize, fontweight="bold")
            if show_risktable and table_ax is not None:
                table_ax.set_xlabel("")
        except Exception:
            pass
        ax.set_ylabel(ylab, fontsize=label_fontsize, fontweight="bold")
        if title:
            ax.set_title(title, fontsize=title_fontsize, fontweight="bold", loc="left")

        if xlim:
            ax.set_xlim(xlim)
        elif timeline is not None:
            try:
                max_time = float(timeline[-1])
            except Exception:
                max_time = float(np.max(timeline)) if len(list(timeline)) > 0 else 0.0
            interval = float(xtick_interval_months) if xtick_interval_months is not None else None
            if interval and interval > 0:
                snapped_max = float(interval) * float(np.ceil(max_time / float(interval)))
                ax.set_xlim(0.0, snapped_max)
            else:
                ax.set_xlim(0.0, max_time)
        else:
            # derive max time from KMF timelines
            max_time = 0
            for kmf in self.kmfs:
                try:
                    times = float(kmf.timeline.max()) if len(kmf.timeline) > 0 else 0.0
                except Exception:
                    times = 0.0
                max_time = max(max_time, times)
            if max_time == 0:
                ax.set_xlim(0, 1.0)
            else:
                padding = max(0.05 * max_time, 0.1)
                ax.set_xlim(0, max_time + padding)

        if ylim:
            ax.set_ylim(ylim)
        try:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        except Exception:
            pass

        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        _legend_loc = legend_loc
        _legend_title = legend_title if legend_title is not None else ""
        legend = bk.position_legend(
            ax,
            loc=_legend_loc,
            title=_legend_title,
            title_fontweight=legend_title_fontweight or "bold",
            fontsize=legend_fontsize,
            frameon=legend_frameon,
            label_wrap_chars=legend_label_wrap_chars,
            label_max_lines=legend_label_max_lines,
            markerscale=float(legend_markerscale),
            linewidth_scale=legend_linewidth_scale,
        )
        if legend is not None and (auto_expand_for_legend or _legend_loc == "right"):
            try:
                bk.expand_figure_to_fit_legend(fig, legend, pad_in=0.3)
            except Exception:
                pass

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmin > 0:
            xmin = 0
            ax.set_xlim(xmin, xmax)
        if ymin > 0:
            ymin = 0
            ax.set_ylim(ymin, ymax)

        # xticks
        if xticks is not None:
            xticks_final = list(xticks)
        elif timeline is not None:
            try:
                _tl = [float(t) for t in list(timeline)]
            except Exception:
                _tl = []
            if xtick_interval_months is not None and len(_tl) > 0:
                interval = float(xtick_interval_months)
                xmin_cur, xmax_cur = ax.get_xlim()
                snapped_max = interval * np.ceil(float(xmax_cur) / interval)
                ax.set_xlim(xmin_cur, snapped_max)
                xticks_final = list(np.arange(0.0, snapped_max + 1e-9, interval))
            else:
                if len(_tl) >= 2 and (max(_tl) - min(_tl)) > 0:
                    xticks_final = _tl
                else:
                    xticks_final = None
        else:
            xmin_cur, xmax_cur = ax.get_xlim()
            if xtick_interval_months is not None:
                interval = float(xtick_interval_months)
                snapped_max = interval * np.ceil(float(xmax_cur) / interval)
                ax.set_xlim(xmin_cur, snapped_max)
                xticks_final = list(np.arange(0.0, snapped_max + 1e-9, interval))
            else:
                xticks_final = None

        if xticks_final is not None:
            ax.set_xticks(xticks_final)
            ax.set_xticklabels(
                [str(int(t)) if float(t).is_integer() else str(t) for t in xticks_final]
            )
            try:
                ax.tick_params(
                    axis="x", which="both", labelbottom=True, bottom=True, direction="out"
                )
                for lab in ax.get_xticklabels():
                    lab.set_visible(True)
            except Exception:
                pass

        ax.tick_params(axis="both", which="major", pad=8, width=1.5)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        try:
            ax.margins(x=0.0, y=0.0)
            if hasattr(ax, "set_xmargin"):
                ax.set_xmargin(0.0)
            if hasattr(ax, "set_ymargin"):
                ax.set_ymargin(0.0)
        except Exception:
            pass

        # Risk table
        risk_table_height = 0
        table_axes = None
        if show_risktable and self.kmfs:
            xticks_for_rt = None
            try:
                if xticks_final is not None:
                    xticks_for_rt = list(xticks_final)
                else:
                    xmin_rt, xmax_rt = ax.get_xlim()
                    ticks = ax.get_xticks()
                    xticks_for_rt = [
                        float(t) for t in ticks if (xmin_rt - 1e-9) <= float(t) <= (xmax_rt + 1e-9)
                    ]
            except Exception:
                xticks_for_rt = None

            _rt_title_fs = (
                int(risktable_title_fontsize)
                if risktable_title_fontsize is not None
                else int(risktable_fontsize) + 2
            )
            table_axes, risk_table_height = bk.add_custom_risktable(
                ax,
                self.kmfs,
                labels=table_labels,
                colors=[custom_colors.get(lab) for lab in table_labels],
                rows_to_show=["At risk"],
                fontsize=risktable_fontsize,
                title_fontsize=_rt_title_fs,
                xticks=xticks_for_rt,
                color_counts=color_risktable_counts,
                use_timeline_for_risktable=True,
                row_spacing=risktable_row_spacing,
                title_gap_factor=risktable_title_gap_factor,
                label_wrap_chars=risktable_label_wrap_chars,
                label_max_lines=risktable_label_max_lines,
                table_ax=table_ax,
                per_row_in_inch=per_row_in,
                title_pad_in_inch=title_pad_in,
                adaptive_title_gap=False,
                min_display_rows=effective_rows,
            )

            if not color_risktable_counts and table_axes:
                try:
                    for ta in table_axes if isinstance(table_axes, (list, tuple)) else [table_axes]:
                        for txt in getattr(ta, "texts", []):
                            try:
                                if str(getattr(txt, "get_ha", lambda: "")()).lower() == "center":
                                    txt.set_color("black")
                            except Exception:
                                pass
                except Exception:
                    pass

            try:
                ax.tick_params(axis="x", which="both", labelbottom=True, bottom=True)
                for lab in ax.get_xticklabels():
                    lab.set_visible(True)
                if table_ax is not None:
                    if xticks_for_rt is not None:
                        table_ax.set_xticks(xticks_for_rt)
                    table_ax.tick_params(
                        axis="x", which="both", labelbottom=False, bottom=False, length=0
                    )
                    for lab in table_ax.get_xticklabels():
                        lab.set_visible(False)
            except Exception:
                pass

            try:
                artists = []
                if legend is not None:
                    artists.append(legend)
                if pval_text is not None:
                    artists.append(pval_text)
                if table_axes:
                    artists.extend(table_axes)
                    try:
                        for ta in table_axes:
                            for txt in getattr(ta, "texts", []):
                                artists.append(txt)
                    except Exception:
                        pass
                try:
                    artists.append(ax.yaxis.get_label())
                    for ytick in ax.get_yticklabels():
                        artists.append(ytick)
                except Exception:
                    pass
                if artists:
                    bk.expand_figure_to_fit_artists(
                        fig,
                        artists,
                        pad_left_in=0.6,
                        pad_right_in=0.8,
                        pad_top_in=0.4,
                        pad_bottom_in=1.0,
                    )
            except Exception:
                pass

        # Ensure KM axis physical size
        try:
            desired_km_height_in = float(fig_height)
            fig_height_in = float(fig.get_figheight()) if fig is not None else 0.0
            if desired_km_height_in > 0.0 and fig_height_in > 0.0 and ax is not None:
                km_bbox = ax.get_position()
                current_km_height_in = float(km_bbox.height) * fig_height_in
                if (
                    current_km_height_in > 0.0
                    and abs(current_km_height_in - desired_km_height_in) > 1e-2
                ):
                    scale = desired_km_height_in / current_km_height_in
                    new_fig_height_in = fig_height_in * scale
                    fig.set_size_inches(fig.get_figwidth(), new_fig_height_in, forward=True)
        except Exception:
            pass

        if ax is not None and fig is not None and not show_risktable:
            bk.adjust_plot_margins(
                fig,
                ax,
                legend_loc=legend_loc,
                risk_table=False,
                risk_table_height=risk_table_height,
            )

        if output_path and fig is not None:
            try:
                fig.canvas.draw()
            except Exception:
                pass
            save_kwargs = {"dpi": 300}
            _use_tight = (
                save_bbox_inches is not None
                and save_bbox_inches == "tight"
                and not auto_expand_for_legend
                and legend_loc not in ("right",)
                and not show_risktable
            )
            if _use_tight:
                save_kwargs["bbox_inches"] = "tight"
            try:
                bk.expand_canvas(fig, left_in=0.5, bottom_in=1.0, right_in=0.8, top_in=0.4)
            except Exception:
                pass
            fig.savefig(output_path, **save_kwargs)

        return fig, ax, p_value

    # backward-compatible convenience
    def plot(self, config: KMPlotConfig, output_path: Optional[str] = None):
        return self.plot_from_kmfs(config=config, output_path=output_path)
