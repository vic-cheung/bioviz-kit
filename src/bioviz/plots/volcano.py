"""Volcano plotting with a pydantic `VolcanoConfig`.

This file provides a single, explicit configuration object `VolcanoConfig`
and a `plot_volcano(cfg)` function that uses it. No wrapper/shims for legacy
APIs are provided — this is the canonical, hard refactor you requested.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Mapping, Dict

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from bioviz.configs.volcano_cfg import VolcanoConfig

try:
    from adjustText import adjust_text
except Exception:

    def adjust_text(texts, *args, **kwargs):  # type: ignore
        return None


def _internal_resolve_values(df: pd.DataFrame, cfg: VolcanoConfig) -> List[str]:
    # If caller provided exact values to label, honor that
    if cfg.values_to_label:
        base = list(cfg.values_to_label)
        # Warn about any requested labels that are not present in the DataFrame
        try:
            if cfg.label_col and cfg.label_col in df.columns:
                available = set(df[cfg.label_col].astype(str).tolist())
            else:
                available = set(df.index.astype(str).tolist())
            missing = [v for v in base if v not in available]
            if missing:
                warnings.warn(f"Requested labels not found in DataFrame: {missing}", UserWarning)
            if cfg.additional_values_to_label:
                # Warn about missing additions, but only append the valid ones
                add = list(cfg.additional_values_to_label)
                missing_add = [v for v in add if v not in available]
                if missing_add:
                    warnings.warn(
                        f"Additional requested labels not found in DataFrame: {missing_add}",
                        UserWarning,
                    )
                valid_add = [v for v in add if v in available]
                base = list(dict.fromkeys(base + valid_add))
        except Exception:
            pass
        return base

    # Determine label source (explicit column or index fallback)
    if cfg.label_col and cfg.label_col in df.columns:
        labels_series = df[cfg.label_col].astype(str)
    else:
        # fallback: create a Series from the DataFrame index so callers
        # can use `.loc` consistently later on
        labels_series = pd.Series(df.index.astype(str), index=df.index)

    # Warn the user if no explicit label column or explicit labels were
    # provided — the plotting code will fall back to using the DataFrame
    # index as labels which is often unintentional.
    if not (cfg.label_col and cfg.label_col in df.columns) and not cfg.values_to_label:
        try:
            warnings.warn(
                "No `label_col` found and no `values_to_label` provided; "
                "labels will be taken from the DataFrame index. "
                "If you intended to label using a column, set `cfg.label_col` or "
                "provide `values_to_label`.",
                UserWarning,
            )
        except Exception:
            pass

    # Build a significance mask using the configured `y_col` and `y_col_thresh`.
    sig_mask = pd.Series(False, index=df.index)
    try:
        y_thresh = getattr(cfg, "y_col_thresh", None)
        if y_thresh is not None and cfg.y_col and cfg.y_col in df.columns:
            sig_mask = df[cfg.y_col].astype(float).fillna(1.0) <= y_thresh
    except Exception:
        sig_mask = pd.Series(False, index=df.index)

    eff_mask = (
        df[cfg.x_col].abs() >= cfg.abs_x_thresh
        if cfg.x_col in df.columns
        else pd.Series(False, index=df.index)
    )

    # label_mode controls which points are chosen when `values_to_label` is not provided
    mode = getattr(cfg, "label_mode", "auto")
    if mode == "all":
        base = labels_series.tolist()
    elif mode == "sig":
        base = labels_series.loc[sig_mask].tolist()
    elif mode == "thresh":
        base = labels_series.loc[eff_mask].tolist()
    elif mode == "sig_and_thresh":
        base = labels_series.loc[sig_mask & eff_mask].tolist()
    elif mode == "sig_or_thresh":
        base = labels_series.loc[sig_mask | eff_mask].tolist()
    else:
        # 'auto' (and any unknown value) defaults to the intersection
        # of significance and magnitude — label points that meet both.
        mask = sig_mask & eff_mask
        base = labels_series.loc[mask].tolist()

    if cfg.additional_values_to_label:
        available = set(labels_series.tolist())
        valid_add = [g for g in cfg.additional_values_to_label if g in available]
        base = list(dict.fromkeys(base + valid_add))

    return base


def resolve_labels(df: pd.DataFrame, cfg: VolcanoConfig) -> List[str]:
    """Return the final list of labels `plot_volcano` will use.

    This helper mirrors the internal selection logic, including:
    - honoring `values_to_label` and `additional_values_to_label`,
    - applying `label_mode` when `values_to_label` isn't provided,
    - excluding explicit placements when `explicit_label_replace` is True
      (so callers can see the de-duplicated final set used for auto-labeling).
    Useful for debugging or UI workflows where you want to preview labels
    before re-rendering.
    """
    # Start from the internal resolved list (this handles values_to_label/additional)
    base = _internal_resolve_values(df, cfg)

    # If explicit_label_positions are present and explicit_label_replace=True,
    # those explicit labels are removed from the auto-label set inside
    # plot_volcano; reflect that behavior here so the returned list matches
    # what will actually be auto-labeled.
    explicit_map = {}
    if getattr(cfg, "explicit_label_positions", None) is not None:
        try:
            elp = cfg.explicit_label_positions
            if isinstance(elp, dict):
                explicit_map = {str(k): v for k, v in elp.items()}
            elif hasattr(elp, "columns"):
                cols = [c.lower() for c in elp.columns]
                if "label" in cols and ("x" in cols and "y" in cols):
                    for _, r in elp.iterrows():
                        explicit_map[str(r["label"])] = (float(r["x"]), float(r["y"]))
                else:
                    labcol = cfg.label_col
                    xcol = cfg.x_col
                    ycol = cfg.y_col
                    for _, r in elp.iterrows():
                        try:
                            explicit_map[str(r[labcol])] = (float(r[xcol]), float(r[ycol]))
                        except Exception:
                            continue
            else:
                for it in elp:
                    try:
                        explicit_map[str(it[0])] = (float(it[1][0]), float(it[1][1]))
                    except Exception:
                        continue
        except Exception:
            explicit_map = {}

    if explicit_map and getattr(cfg, "explicit_label_replace", True):
        base = [v for v in base if v not in explicit_map]

    return base


def plot_volcano(cfg: VolcanoConfig, df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a volcano using the provided `VolcanoConfig`.

    This function is intentionally strict: it requires a `VolcanoConfig` and
    the dataframe to plot. It uses the config for everything and performs no
    backward-compatibility shims.
    """
    df = df.copy()

    # Resolve labels
    values_to_label_resolved = _internal_resolve_values(df, cfg)

    # y values (allow transformation of p-values to -log10 if requested)
    # Start with raw values; we may replace with -log10(p) below.
    y_vals = (
        df[cfg.y_col]
        if (cfg.y_col and cfg.y_col in df.columns)
        else pd.Series(np.nan, index=df.index)
    )
    transformed_y = False
    # Decide whether to transform the y-column to -log10:
    # - `cfg.log_transform_ycol` True -> perform transform
    # - False -> do not transform
    do_transform = bool(getattr(cfg, "log_transform_ycol", False))

    # Perform the -log10 transform only when explicitly requested.
    if do_transform and cfg.y_col and cfg.y_col in df.columns:
        try:
            y_vals = -np.log10(
                pd.to_numeric(df[cfg.y_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            )
            transformed_y = True
        except Exception:
            y_vals = df[cfg.y_col]

    # Figure / axis
    if cfg.ax is None:
        fig, ax = plt.subplots(figsize=cfg.figsize)
        # Make figure background transparent while keeping axes face white
        try:
            fig.patch.set_alpha(0.0)
        except Exception:
            pass
        try:
            ax.set_facecolor("white")
        except Exception:
            pass
    else:
        ax = cfg.ax
        fig = ax.figure

    # Helper: compute a point on the marker edge (in data coords) in the
    # direction toward `target_disp` so connectors attach at the marker edge
    # rather than the marker center. This uses `cfg.marker_size` (the scatter
    # `s` parameter) to estimate a display-space radius.
    def _marker_edge_data_point(xd: float, yd: float, target_disp: Tuple[float, float]):
        try:
            # center in display coords
            center_disp = ax.transData.transform((xd, yd))
            dx = target_disp[0] - center_disp[0]
            dy = target_disp[1] - center_disp[1]
            norm = math.hypot(dx, dy)
            if norm <= 1e-8:
                return xd, yd
            ux, uy = dx / norm, dy / norm
            # estimate marker radius in display pixels. `cfg.marker_size` is
            # passed to scatter as `s` (points^2); approximate radius in
            # points as sqrt(s)/2, then convert to pixels: pixels = points * dpi/72.
            r_points = math.sqrt(max(cfg.marker_size, 1.0)) / 2.0
            r_pixels = r_points * fig.dpi / 72.0
            edge_disp = (center_disp[0] + ux * r_pixels, center_disp[1] + uy * r_pixels)
            edge_data = ax.transData.inverted().transform(edge_disp)
            return float(edge_data[0]), float(edge_data[1])
        except Exception:
            return xd, yd

    def _select_connector_color(is_sig: bool, ox: float):
        """Return connector color using hierarchical precedence:
        most-specific (sign+side) -> side -> sign -> nonsig -> generic.
        """
        try:
            # Most specific: sign + side
            if is_sig:
                if ox < 0 and getattr(cfg, "connector_color_sig_left", None):
                    return cfg.connector_color_sig_left
                if ox >= 0 and getattr(cfg, "connector_color_sig_right", None):
                    return cfg.connector_color_sig_right
            else:
                if ox < 0 and getattr(cfg, "connector_color_nonsig_left", None):
                    return cfg.connector_color_nonsig_left
                if ox >= 0 and getattr(cfg, "connector_color_nonsig_right", None):
                    return cfg.connector_color_nonsig_right

            # Per-side override
            side_color = cfg.connector_color_left if ox < 0 else cfg.connector_color_right
            if side_color:
                return side_color

            # Per-significance override
            if is_sig and getattr(cfg, "connector_color_sig", None):
                return cfg.connector_color_sig
            if (not is_sig) and getattr(cfg, "connector_color_nonsig", None):
                return cfg.connector_color_nonsig

            # Final fallback
            return cfg.connector_color
        except Exception:
            return cfg.connector_color

    def _nudge_label_if_overlapping(text_obj, marker_x, marker_y, marker_radius_pixels=None):
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text_obj.get_window_extent(renderer=renderer)
            if marker_radius_pixels is None:
                r_points = math.sqrt(max(cfg.marker_size, 1.0)) / 2.0
                marker_radius_pixels = r_points * fig.dpi / 72.0
            # Use bbox center to detect overlap, but shift the text anchor
            # (respecting its horizontal/vertical alignment) in display
            # coordinates so the anchor moves consistently with the visual
            # position of the text.
            # Consider ALL markers: find the closest marker whose display
            # position is within the nudge padding and push the text away
            # from that marker to avoid landing on top of other markers.
            cols = [c for c in ax.collections if hasattr(c, "get_offsets")]
            marker_disp_positions = []
            for c in cols:
                try:
                    offs = c.get_offsets()
                    for off in offs:
                        marker_disp_positions.append(
                            tuple(ax.transData.transform((off[0], off[1])))
                        )
                except Exception:
                    continue

            bbox_center = (bbox.x0 + bbox.width / 2.0, bbox.y0 + bbox.height / 2.0)
            padding = getattr(cfg, "nudge_padding_pixels", 6.0)
            closest = None
            closest_dist = float("inf")
            for md in marker_disp_positions:
                d = math.hypot(bbox_center[0] - md[0], bbox_center[1] - md[1])
                if d < closest_dist:
                    closest_dist = d
                    closest = md

            if closest is not None and closest_dist < (marker_radius_pixels + padding):
                # Prefer to nudge horizontally toward the side with more free space
                ax_left, ax_right = ax.bbox.x0, ax.bbox.x1
                space_left = bbox_center[0] - ax_left
                space_right = ax_right - bbox_center[0]
                horiz_dir = 1.0 if space_right >= space_left else -1.0

                # compute directional vector away from the closest marker
                ux = (bbox_center[0] - closest[0]) / (closest_dist + 1e-8)
                uy = (bbox_center[1] - closest[1]) / (closest_dist + 1e-8)
                # bias horizontal movement to preferred side
                ux = horiz_dir
                shift_pixels = marker_radius_pixels + padding - closest_dist
                # limit vertical movement so labels remain horizontally aligned
                uy = uy * 0.25
                text_anchor_data = text_obj.get_position()
                text_anchor_disp = ax.transData.transform(
                    (text_anchor_data[0], text_anchor_data[1])
                )
                new_anchor_disp = (
                    text_anchor_disp[0] + ux * shift_pixels,
                    text_anchor_disp[1] + uy * shift_pixels,
                )
                new_anchor_data = ax.transData.inverted().transform(new_anchor_disp)
                text_obj.set_position((new_anchor_data[0], new_anchor_data[1]))
                fig.canvas.draw()
        except Exception:
            return

    # significance mask (use appropriate threshold when y was transformed)
    sig_mask = pd.Series(False, index=df.index)
    y_col_thresh = getattr(cfg, "y_col_thresh", None)
    try:
        # build y-based mask depending on whether we transformed the y values
        if y_col_thresh is not None and cfg.y_col in df.columns:
            if transformed_y:
                try:
                    thr = -np.log10(y_col_thresh)
                except Exception:
                    thr = None
                if thr is not None:
                    y_mask = y_vals.fillna(0.0) >= thr
                else:
                    y_mask = pd.Series(False, index=df.index)
            else:
                y_mask = df[cfg.y_col].fillna(1.0) <= y_col_thresh

            # Build y-based significance mask (x-threshold is applied later
            # when label_mode or color_mode requests intersection semantics).
            sig_mask = y_mask
    except Exception:
        sig_mask = pd.Series(False, index=df.index)

    # magnitude-based mask (points whose absolute x value exceeds the
    # configured `abs_x_thresh`). Define here so color/label selection
    # logic below can reference it regardless of control flow.
    try:
        abs_x_thresh = getattr(cfg, "abs_x_thresh", None)
        if abs_x_thresh is not None and cfg.x_col in df.columns:
            eff_mask = df[cfg.x_col].abs() >= abs_x_thresh
        else:
            eff_mask = pd.Series(False, index=df.index)
    except Exception:
        eff_mask = pd.Series(False, index=df.index)

    # color selection helpers
    def _choose_direction_color(val):
        try:
            s = str(val).lower()
        except Exception:
            s = ""
        if any(tok in s for tok in ("down", "decrease", "loss", "neg", "-")):
            return cfg.palette.get("sig_down")
        if any(tok in s for tok in ("up", "increase", "gain", "pos", "+")):
            return cfg.palette.get("sig_up")
        return cfg.palette.get("sig_up")

    # Determine which points are considered "colored" according to the
    # requested `cfg.color_mode` and then map colors accordingly. This
    # separates selection logic from label selection so callers can choose
    # independent behaviors for coloring vs labeling.
    color_mode = getattr(cfg, "color_mode", "sig")
    # If the user requested 'sig' coloring but no thresholds are available
    # (neither y_col_thresh nor a positive abs_x_thresh present in the
    # dataframe), interpret that as a request to color all points so the
    # plot isn't entirely nonsignificant by default.
    try:
        has_y_thresh = getattr(cfg, "y_col_thresh", None) is not None and cfg.y_col in df.columns
        has_x_thresh = (
            getattr(cfg, "abs_x_thresh", None) is not None
            and cfg.abs_x_thresh > 0
            and cfg.x_col in df.columns
        )
        if (color_mode == "sig") and (not has_y_thresh) and (not has_x_thresh):
            color_mode = "all"
    except Exception:
        pass
    if color_mode == "all":
        color_mask = pd.Series(True, index=df.index)
    elif color_mode == "sig":
        color_mask = sig_mask.copy()
    elif color_mode == "thresh":
        color_mask = eff_mask.copy()
    elif color_mode == "sig_and_thresh":
        color_mask = sig_mask & eff_mask
    elif color_mode == "sig_or_thresh":
        color_mask = sig_mask | eff_mask
    else:
        color_mask = sig_mask.copy()

    colors = []
    if cfg.direction_col and cfg.direction_col in df.columns and cfg.direction_colors:
        for i in df.index:
            if not color_mask.loc[i]:
                colors.append(cfg.palette.get("nonsig"))
            else:
                colors.append(
                    cfg.direction_colors.get(
                        df.loc[i, cfg.direction_col], cfg.palette.get("sig_up")
                    )
                )
    else:
        for i in df.index:
            if not color_mask.loc[i]:
                colors.append(cfg.palette.get("nonsig"))
                continue
            if cfg.direction_col and cfg.direction_col in df.columns:
                color = _choose_direction_color(df.loc[i, cfg.direction_col])
            else:
                try:
                    xv = float(df.loc[i, cfg.x_col])
                except Exception:
                    xv = 0.0
                color = cfg.palette.get("sig_up") if xv >= 0 else cfg.palette.get("sig_down")
            colors.append(color)

    # axis limits: compute sensible defaults but allow caller overrides via cfg.xlim/cfg.ylim
    x_data_min, x_data_max = df[cfg.x_col].min(), df[cfg.x_col].max()
    y_data_max = y_vals.max()
    x_limit = max(4, abs(x_data_min), abs(x_data_max))
    y_limit = max(8, y_data_max)
    # If caller provided explicit limits, use them. Otherwise use computed defaults.
    if getattr(cfg, "xlim", None) is not None:
        try:
            ax.set_xlim(tuple(cfg.xlim))
        except Exception:
            ax.set_xlim(-x_limit, x_limit)
    else:
        ax.set_xlim(-x_limit, x_limit)

    if getattr(cfg, "ylim", None) is not None:
        try:
            ax.set_ylim(tuple(cfg.ylim))
        except Exception:
            ax.set_ylim(bottom=-0.5, top=y_limit)
    else:
        ax.set_ylim(bottom=-0.5, top=y_limit)

    # draw threshold lines
    ax.axvline(x=0.0, color="#000000", linestyle="-", linewidth=0.8, zorder=1)
    if cfg.x_thresh:
        for xt in cfg.x_thresh:
            ax.axvline(
                x=xt,
                color=(cfg.x_thresh_line_color or cfg.thresh_line_color),
                linestyle=(cfg.x_thresh_line_style or cfg.thresh_line_style),
                linewidth=(cfg.x_thresh_line_width or cfg.thresh_line_width),
                zorder=1,
            )
    else:
        # If caller didn't provide explicit x_thresholds, draw lines
        # at ±abs_x_thresh when it's set to a finite positive value.
        try:
            if cfg.abs_x_thresh is not None and cfg.abs_x_thresh > 0:
                ax.axvline(
                    x=cfg.abs_x_thresh,
                    color=(cfg.x_thresh_line_color or cfg.thresh_line_color),
                    linestyle=(cfg.x_thresh_line_style or cfg.thresh_line_style),
                    linewidth=(cfg.x_thresh_line_width or cfg.thresh_line_width),
                    zorder=1,
                )
                ax.axvline(
                    x=-cfg.abs_x_thresh,
                    color=(cfg.x_thresh_line_color or cfg.thresh_line_color),
                    linestyle=(cfg.x_thresh_line_style or cfg.thresh_line_style),
                    linewidth=(cfg.x_thresh_line_width or cfg.thresh_line_width),
                    zorder=1,
                )
        except Exception:
            pass
    if cfg.y_thresh is not None:
        thr_y = cfg.y_thresh
    elif transformed_y and getattr(cfg, "y_col_thresh", None) is not None:
        try:
            thr_y = -np.log10(getattr(cfg, "y_col_thresh", None))
        except Exception:
            thr_y = None
    else:
        thr_y = None

    if thr_y is not None:
        ax.axhline(
            y=thr_y,
            color=(cfg.y_thresh_line_color or cfg.thresh_line_color),
            linestyle=(cfg.y_thresh_line_style or cfg.thresh_line_style),
            linewidth=(cfg.y_thresh_line_width or cfg.thresh_line_width),
            zorder=1,
        )

    # scatter (use explicit cfg.marker_size)
    sc = ax.scatter(
        df[cfg.x_col],
        y_vals,
        c=colors,
        edgecolor="black",
        linewidths=0.5,
        s=cfg.marker_size,
        zorder=3,
    )
    try:
        # avoid clipping markers at the axes boundary
        sc.set_clip_on(False)
    except Exception:
        pass

    # build labels aggregated by coordinates
    all_texts = []
    forced_texts = []
    forced_points = []
    adjustable_texts = []
    adjustable_points = []
    adjustable_point_sigs = []
    coord_to_labels = {}
    for i, row in df.iterrows():
        try:
            coord = (float(row[cfg.x_col]), float(y_vals.loc[i]))
        except Exception:
            continue
        dir_val = (
            row[cfg.direction_col]
            if (cfg.direction_col and cfg.direction_col in df.columns)
            else None
        )
        # Resolve label value (use label_col if present, else use index)
        try:
            if cfg.label_col and cfg.label_col in df.columns:
                labval = str(row[cfg.label_col])
            else:
                labval = str(i)
        except Exception:
            labval = str(i)
        coord_to_labels.setdefault(coord, []).append((i, labval, bool(sig_mask.loc[i]), dir_val))

    # Parse explicit label placements if provided. Support dict, iterable of
    # (label, (x,y)) tuples, or a DataFrame with label/x/y columns.
    explicit_map = {}
    if getattr(cfg, "explicit_label_positions", None) is not None:
        try:
            elp = cfg.explicit_label_positions
            # dict-like
            if isinstance(elp, dict):
                for k, v in elp.items():
                    try:
                        explicit_map[str(k)] = (float(v[0]), float(v[1]))
                    except Exception:
                        continue
            # DataFrame-like
            elif hasattr(elp, "columns"):
                # prefer explicit 'label','x','y' columns
                cols = [c.lower() for c in elp.columns]
                if "label" in cols and ("x" in cols and "y" in cols):
                    for _, r in elp.iterrows():
                        try:
                            explicit_map[str(r["label"])] = (float(r["x"]), float(r["y"]))
                        except Exception:
                            continue
                else:
                    # try using label_col and x_col/y_col names
                    labcol = cfg.label_col
                    xcol = cfg.x_col
                    ycol = cfg.y_col
                    for _, r in elp.iterrows():
                        try:
                            explicit_map[str(r[labcol])] = (float(r[xcol]), float(r[ycol]))
                        except Exception:
                            continue
            else:
                # iterable of (label,(x,y))
                for it in elp:
                    try:
                        lab = str(it[0])
                        xy = it[1]
                        explicit_map[lab] = (float(xy[0]), float(xy[1]))
                    except Exception:
                        continue
        except Exception:
            explicit_map = {}
    # Warn if explicit labels reference names not present in the DataFrame
    try:
        if explicit_map:
            if cfg.label_col and cfg.label_col in df.columns:
                available_labels = set(df[cfg.label_col].astype(str).tolist())
            else:
                available_labels = set(df.index.astype(str).tolist())
            missing_explicit = [k for k in explicit_map.keys() if k not in available_labels]
            if missing_explicit:
                warnings.warn(
                    f"Explicit label positions reference labels not in DataFrame: {missing_explicit}",
                    UserWarning,
                )
    except Exception:
        pass

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    # Place any explicit labels requested by the caller.
    if explicit_map:
        # Optionally remove explicit labels from the auto-label set so they
        # aren't duplicated. Default behavior is to replace automatic labels.
        if getattr(cfg, "explicit_label_replace", True):
            try:
                values_to_label_resolved = [
                    v for v in values_to_label_resolved if v not in explicit_map
                ]
            except Exception:
                pass
        for lab, (lx, ly) in explicit_map.items():
            # skip if outside axes
            if not (x_min <= lx <= x_max and y_min <= ly <= y_max):
                continue
            # find if this label corresponds to a data point to inherit sig status
            matched_idx = None
            try:
                # match by label column or index
                if cfg.label_col and cfg.label_col in df.columns:
                    matches = df.index[df[cfg.label_col].astype(str) == lab].tolist()
                else:
                    matches = df.index[df.index.astype(str) == lab].tolist()
                matched_idx = matches[0] if matches else None
            except Exception:
                matched_idx = None

            is_sig = False
            point_color = None
            dir_val = None
            if matched_idx is not None:
                try:
                    is_sig = bool(sig_mask.loc[matched_idx])
                    pos = list(df.index).index(matched_idx)
                    point_color = colors[pos]
                    dir_val = (
                        df.loc[matched_idx, cfg.direction_col]
                        if (cfg.direction_col and cfg.direction_col in df.columns)
                        else None
                    )
                except Exception:
                    is_sig = False

            # choose annotation color
            if is_sig:
                ann_color = cfg.annotation_sig_color or point_color or cfg.palette.get("sig_up")
                weight = getattr(cfg, "annotation_fontweight_sig", "bold")
                fontsize = cfg.fontsize_sig
            else:
                ann_color = getattr(cfg, "annotation_nonsig_color", "#7f7f7f")
                weight = getattr(cfg, "annotation_fontweight_nonsig", "normal")
                fontsize = cfg.fontsize_nonsig

            t = ax.text(
                lx,
                ly,
                lab,
                fontsize=fontsize,
                fontweight=weight,
                color=ann_color,
                zorder=4,
                clip_on=False,
            )
            _nudge_label_if_overlapping(t, lx, ly)
            # Optionally include explicit labels in the adjust_text flow
            if getattr(cfg, "explicit_label_adjustable", False):
                adjustable_texts.append(t)
                try:
                    if matched_idx is not None:
                        adjustable_points.append(
                            (float(df.loc[matched_idx, cfg.x_col]), float(y_vals.loc[matched_idx]))
                        )
                    else:
                        adjustable_points.append((lx, ly))
                except Exception:
                    adjustable_points.append((lx, ly))
                adjustable_point_sigs.append(is_sig)
            else:
                # Draw connector from marker (if we matched a point) or skip
                if matched_idx is not None:
                    try:
                        ox = float(df.loc[matched_idx, cfg.x_col])
                        oy = float(y_vals.loc[matched_idx])
                    except Exception:
                        ox, oy = None, None
                    if ox is not None:
                        try:
                            if getattr(cfg, "attach_to_marker_edge", True):
                                label_disp = ax.transData.transform((lx, ly))
                                attach_x, attach_y = _marker_edge_data_point(ox, oy, label_disp)
                            else:
                                attach_x, attach_y = ox, oy
                        except Exception:
                            attach_x, attach_y = ox, oy

                        if getattr(cfg, "connector_color_use_point_color", False) and point_color:
                            conn_color = point_color
                        else:
                            conn_color = _select_connector_color(is_sig, ox)
                        ax.plot(
                            [attach_x, lx],
                            [attach_y, ly],
                            color=conn_color,
                            linewidth=cfg.connector_width,
                            alpha=0.8,
                            zorder=3.5,
                        )
    # Build a group -> color map for left/right labels if possible
    group_side_color = {}
    try:
        if cfg.direction_col and cfg.direction_col in df.columns:
            # Build means for each group and assign colors strictly by sign
            g_kwargs = cfg.group_label_kwargs or {}
            color_val = g_kwargs.get("color", {})
            color_map = color_val if isinstance(color_val, dict) else {}
            means = df.groupby(cfg.direction_col)[cfg.x_col].mean()
            for grp in means.index:
                lab = str(grp)
                # explicit override first
                if lab in color_map:
                    group_side_color[lab] = color_map[lab]
                    continue
                if cfg.direction_colors and lab in (cfg.direction_colors or {}):
                    group_side_color[lab] = cfg.direction_colors.get(lab)
                    continue
                # assign by mean sign
                try:
                    if means.loc[grp] < 0:
                        group_side_color[lab] = cfg.palette.get("sig_down")
                    else:
                        group_side_color[lab] = cfg.palette.get("sig_up")
                except Exception:
                    group_side_color[lab] = cfg.palette.get("nonsig")
    except Exception:
        group_side_color = {}

    for coord, items in coord_to_labels.items():
        x, y = coord
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        items = [it for it in items if it[1] in values_to_label_resolved]
        sig_items = [it for it in items if it[2]]
        nonsig_items = [it for it in items if not it[2]]
        stacked = sig_items + nonsig_items
        labels = [it[1] for it in stacked]
        if not labels:
            continue
        text_str = "\n".join(labels)
        # Determine group-based color if available
        group_val = stacked[0][3] if stacked and len(stacked) and len(stacked[0]) > 3 else None
        ann_color = None
        if group_val is not None and str(group_val) in group_side_color:
            ann_color = group_side_color[str(group_val)]

        # If no group color found, fall back to the actual marker color for the representative point
        point_color = None
        try:
            rep_idx = stacked[0][0]
            # map index to position in colors list
            pos = list(df.index).index(rep_idx)
            point_color = colors[pos]
        except Exception:
            point_color = None

        # Choose placement mode: forced outward by point sign, or adjustable
        forced_mode = getattr(cfg, "force_label_side_by_point_sign", False)
        if forced_mode:
            # deterministic outward placement (no adjust_text for these)
            # compute offset in data units according to `label_offset_mode`
            mode = getattr(cfg, "label_offset_mode", "fraction")
            raw_offset = getattr(cfg, "label_offset", 0.05)
            if mode == "fraction":
                x0, x1 = ax.get_xlim()
                span = float(x1 - x0) if x1 != x0 else 1.0
                offset_data = raw_offset * span
            elif mode == "axes":
                # convert axis fraction to display then to data units using a small dx
                try:
                    disp0 = ax.transAxes.transform((0.0, 0.0))
                    disp1 = ax.transAxes.transform((raw_offset, 0.0))
                    dx_disp = disp1[0] - disp0[0]
                    data_dx = (
                        ax.transData.inverted().transform((dx_disp, 0))[0]
                        - ax.transData.inverted().transform((0, 0))[0]
                    )
                    offset_data = data_dx
                except Exception:
                    offset_data = raw_offset
            else:
                # data mode
                offset_data = raw_offset

            if x < 0:
                tx = x - offset_data
                ha = "right"
            else:
                tx = x + offset_data
                ha = "left"
            ty = y
            is_sig = bool(stacked and stacked[0][2])
            if is_sig:
                color_final = (
                    cfg.annotation_sig_color
                    or ann_color
                    or point_color
                    or cfg.palette.get("sig_up")
                )
                weight = getattr(cfg, "annotation_fontweight_sig", "bold")
                fontsize = cfg.fontsize_sig
            else:
                # Always use the configured nonsignificant annotation color
                # if provided; otherwise use a medium gray.
                color_final = getattr(cfg, "annotation_nonsig_color", "#7f7f7f")
                weight = getattr(cfg, "annotation_fontweight_nonsig", "normal")
                fontsize = cfg.fontsize_nonsig

            # Compute label y so the connector from label->point is ~45 degrees
            try:
                # point display coords
                p_disp = ax.transData.transform((x, y))
                # display x of the label (same y assumed initially)
                label_x_disp = ax.transData.transform((tx, y))[0]
                dx_disp = label_x_disp - p_disp[0]
                # aim for dy_disp ~= abs(dx_disp) to get ~45°; place label above the point
                dy_disp = abs(dx_disp)
                label_y_disp = p_disp[1] + dy_disp
                # convert back to data coords for the y position
                ty = ax.transData.inverted().transform((label_x_disp, label_y_disp))[1]
            except Exception:
                ty = y

            t = ax.text(
                tx,
                ty,
                text_str,
                fontsize=fontsize,
                fontweight=weight,
                color=color_final,
                ha=ha,
                clip_on=False,
                zorder=4,
            )
            # Nudge label if it overlaps its marker
            _nudge_label_if_overlapping(t, x, y)
            force_adjust = getattr(cfg, "force_labels_adjustable", False)
            if force_adjust:
                # include forced labels in the adjustable/adjust_text flow
                adjustable_texts.append(t)
                adjustable_points.append((x, y))
                adjustable_point_sigs.append(is_sig)
            else:
                # straight connector from label to point
                try:
                    # Attach connector to the marker edge in the direction of
                    # the label (so the connector points toward the label).
                    try:
                        if getattr(cfg, "attach_to_marker_edge", True):
                            label_disp = ax.transData.transform((tx, ty))
                            attach_x, attach_y = _marker_edge_data_point(x, y, label_disp)
                        else:
                            attach_x, attach_y = x, y
                    except Exception:
                        attach_x, attach_y = x, y
                    conn_color = _select_connector_color(is_sig, x)
                    # draw a simple straight connector line from marker edge to label
                    ax.plot(
                        [attach_x, tx],
                        [attach_y, ty],
                        color=conn_color,
                        linewidth=cfg.connector_width,
                        alpha=0.8,
                        zorder=3.5,
                    )
                except Exception:
                    pass
                forced_texts.append(t)
                forced_points.append((x, y))
            all_texts.append(t)
        else:
            # adjustable placement (subject to adjust_text)
            if stacked and stacked[0][2]:
                color_final = (
                    cfg.annotation_sig_color
                    or ann_color
                    or point_color
                    or cfg.palette.get("sig_up")
                )
                # place significant labels slightly offset so connectors can be drawn
                lo, hi = getattr(cfg, "horiz_offset_range", (0.02, 0.06))
                samp = np.random.uniform(lo, hi)
                if getattr(cfg, "label_offset_mode", "fraction") == "fraction":
                    x0, x1 = ax.get_xlim()
                    span = float(x1 - x0) if x1 != x0 else 1.0
                    horiz_offset = -abs(samp * span) if x < 0 else abs(samp * span)
                elif getattr(cfg, "label_offset_mode", "fraction") == "axes":
                    try:
                        disp0 = ax.transAxes.transform((0.0, 0.0))
                        disp1 = ax.transAxes.transform((samp, 0.0))
                        dx_disp = disp1[0] - disp0[0]
                        data_dx = (
                            ax.transData.inverted().transform((dx_disp, 0))[0]
                            - ax.transData.inverted().transform((0, 0))[0]
                        )
                        horiz_offset = -abs(data_dx) if x < 0 else abs(data_dx)
                    except Exception:
                        horiz_offset = -abs(samp) if x < 0 else abs(samp)
                else:
                    horiz_offset = -abs(samp) if x < 0 else abs(samp)

                vlo, vhi = getattr(cfg, "vert_jitter_range", (-0.03, 0.03))
                vj = np.random.uniform(vlo, vhi)
                if getattr(cfg, "label_offset_mode", "fraction") == "fraction":
                    x0, x1 = ax.get_xlim()
                    span = float(x1 - x0) if x1 != x0 else 1.0
                    vert_jitter = vj * span
                else:
                    vert_jitter = vj

                t = ax.text(
                    x + horiz_offset,
                    y + vert_jitter,
                    text_str,
                    fontsize=cfg.fontsize_sig,
                    fontweight=getattr(cfg, "annotation_fontweight_sig", "bold"),
                    color=color_final,
                    clip_on=False,
                    zorder=4,
                )
            else:
                # Force nonsig annotation text color from config
                color_final = getattr(cfg, "annotation_nonsig_color", "#7f7f7f")
                ha = "right" if x < 0 else "left"
                # random horizontal offset and vertical jitter (interpreted per mode)
                lo, hi = getattr(cfg, "horiz_offset_range", (0.02, 0.06))
                samp = np.random.uniform(lo, hi)
                if getattr(cfg, "label_offset_mode", "fraction") == "fraction":
                    x0, x1 = ax.get_xlim()
                    span = float(x1 - x0) if x1 != x0 else 1.0
                    horiz_offset = -abs(samp * span) if x < 0 else abs(samp * span)
                elif getattr(cfg, "label_offset_mode", "fraction") == "axes":
                    # convert axes fraction to data dx
                    try:
                        disp0 = ax.transAxes.transform((0.0, 0.0))
                        disp1 = ax.transAxes.transform((samp, 0.0))
                        dx_disp = disp1[0] - disp0[0]
                        data_dx = (
                            ax.transData.inverted().transform((dx_disp, 0))[0]
                            - ax.transData.inverted().transform((0, 0))[0]
                        )
                        horiz_offset = -abs(data_dx) if x < 0 else abs(data_dx)
                    except Exception:
                        horiz_offset = -abs(samp) if x < 0 else abs(samp)
                else:
                    horiz_offset = -abs(samp) if x < 0 else abs(samp)

                vlo, vhi = getattr(cfg, "vert_jitter_range", (-0.03, 0.03))
                vj = np.random.uniform(vlo, vhi)
                if getattr(cfg, "label_offset_mode", "fraction") == "fraction":
                    x0, x1 = ax.get_xlim()
                    span = float(x1 - x0) if x1 != x0 else 1.0
                    vert_jitter = vj * span
                else:
                    vert_jitter = vj
                t = ax.text(
                    x + horiz_offset,
                    y + vert_jitter,
                    text_str,
                    fontsize=cfg.fontsize_nonsig,
                    color=color_final,
                    fontweight=getattr(cfg, "annotation_fontweight_nonsig", "normal"),
                    ha=ha,
                    clip_on=False,
                    zorder=4,
                )
                _nudge_label_if_overlapping(t, x, y)
            adjustable_texts.append(t)
            adjustable_points.append((x, y))
            adjustable_point_sigs.append(stacked and stacked[0][2])
            all_texts.append(t)

    # Axis labels: show transformed math-style labels when appropriate
    # Axis labels (overrides allowed)
    if cfg.x_label:
        ax.set_xlabel(cfg.x_label)
    else:
        lx = cfg.x_col.lower()
        if "log2" in lx or "log_2" in lx:
            # Keep 'OR' non-italicized inside math mode
            ax.set_xlabel(r"$\log_{2}(\mathrm{OR})$")
        else:
            ax.set_xlabel(cfg.x_col)

    if cfg.y_label:
        ax.set_ylabel(cfg.y_label)
    else:
        if transformed_y:
            # Render the original column name literally inside math text to avoid
            # interpreting underscores as subscripts (e.g. p_adj)
            safe_col = cfg.y_col.replace("_", r"\_")
            ax.set_ylabel(r"$-\log_{10}(\text{%s})$" % safe_col)
        else:
            ax.set_ylabel(cfg.y_col)

    # Title and font sizes
    if cfg.title:
        ax.set_title(cfg.title, fontsize=cfg.title_fontsize)
    ax.xaxis.label.set_size(cfg.axis_label_fontsize)
    ax.yaxis.label.set_size(cfg.axis_label_fontsize)
    for tick in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
        tick.set_fontsize(cfg.tick_label_fontsize)

    # Group labels at top: infer from direction_col if not explicitly provided
    if cfg.group_label_top is None and cfg.direction_col and cfg.direction_col in df.columns:
        try:
            means = df.groupby(cfg.direction_col)[cfg.x_col].mean().dropna()
            if len(means) >= 2:
                sorted_idx = means.sort_values().index.tolist()
                cfg_group = (str(sorted_idx[0]), str(sorted_idx[-1]))
            elif len(means) == 1:
                cfg_group = (str(means.index[0]), "")
            else:
                cfg_group = None
        except Exception:
            unique_vals = list(pd.Series(df[cfg.direction_col].astype(str)).unique())
            if len(unique_vals) >= 2:
                cfg_group = (unique_vals[0], unique_vals[1])
            elif len(unique_vals) == 1:
                cfg_group = (unique_vals[0], "")
            else:
                cfg_group = None
    else:
        cfg_group = cfg.group_label_top

    if cfg_group:
        try:
            left_label, right_label = cfg_group
            g_kwargs = cfg.group_label_kwargs or {}
            color_val = g_kwargs.get("color", {})
            color_map = color_val if isinstance(color_val, dict) else {}
            fontsize_g = g_kwargs.get("fontsize", int(cfg.axis_label_fontsize * 0.9))
            left_rot = g_kwargs.get("rotation", 12)
            right_rot = g_kwargs.get("rotation_right", -12)
            ax.text(
                0.02,
                1.02,
                left_label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize_g,
                fontweight="bold",
                color=color_map.get(left_label, cfg.palette.get("sig_up", "#000000")),
                rotation=left_rot,
            )
            ax.text(
                0.98,
                1.02,
                right_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=fontsize_g,
                fontweight="bold",
                color=color_map.get(right_label, cfg.palette.get("nonsig", "gainsboro")),
                rotation=right_rot,
            )
        except Exception:
            pass

    # local staggering for adjustable texts
    for i in range(1, len(adjustable_texts)):
        x_prev, y_prev = adjustable_texts[i - 1].get_position()
        x_curr, y_curr = adjustable_texts[i].get_position()
        if abs(x_curr - x_prev) < 0.2 and abs(y_curr - y_prev) < 0.2:
            adjustable_texts[i].set_position((x_curr, y_prev + 0.4))

    # apply adjust_text only to adjustable labels
    if getattr(cfg, "use_adjust_text", True) and cfg.adjust and adjustable_texts:
        try:
            adjust_text(
                adjustable_texts,
                x=[p[0] for p in adjustable_points],
                y=[p[1] for p in adjustable_points],
                ax=ax,
                expand=(2.5, 2.5),
                force_text=(1.2, 1.5),
                force_points=(0.01, 0.01),
                autoalign="xy",
                arrowprops=None,
                lim=30000,
                ensure_inside_axes=True,
            )
        except Exception:
            pass

    # Draw connector lines from adjustable points to their text bboxes.
    # Forced texts already received straight connectors at placement time.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    for txt, orig, was_sig in zip(adjustable_texts, adjustable_points, adjustable_point_sigs):
        try:
            tx, ty = txt.get_position()
            ox, oy = orig
            if math.hypot(tx - ox, ty - oy) <= 1e-8:
                continue

            if renderer is None:
                # Attach to marker edge when renderer isn't available; aim
                # toward the text position (display coords of tx/ty)
                try:
                    if getattr(cfg, "attach_to_marker_edge", True):
                        label_disp = ax.transData.transform((tx, ty))
                        attach_x, attach_y = _marker_edge_data_point(ox, oy, label_disp)
                    else:
                        attach_x, attach_y = ox, oy
                except Exception:
                    attach_x, attach_y = ox, oy
                try:
                    conn_color = _select_connector_color(was_sig, ox)
                    ax.plot(
                        [attach_x, tx],
                        [attach_y, ty],
                        color=conn_color,
                        linewidth=cfg.connector_width,
                        alpha=0.8,
                        zorder=3.5,
                    )
                except Exception:
                    pass
                continue

            bbox = txt.get_window_extent(renderer=renderer)
            # Convert data point to display coords
            point_disp = ax.transData.transform((ox, oy))

            # Determine which horizontal edge of the text bbox is closer
            # to the point (left or right) and connect to that edge.
            left_edge_x = bbox.x0
            right_edge_x = bbox.x1
            # If point is left of text, attach to left edge; if right, to right edge;
            # if inside horizontally, attach to nearest edge.
            if point_disp[0] <= left_edge_x:
                attach_x = left_edge_x
            elif point_disp[0] >= right_edge_x:
                attach_x = right_edge_x
            else:
                # inside horizontally -> choose nearest edge
                attach_x = (
                    left_edge_x
                    if (point_disp[0] - left_edge_x) < (right_edge_x - point_disp[0])
                    else right_edge_x
                )

            attach_y = bbox.y0 + bbox.height / 2.0
            attach_data = ax.transData.inverted().transform((attach_x, attach_y))
            # compute marker-edge attach point in data coords
            try:
                if getattr(cfg, "attach_to_marker_edge", True):
                    # Aim the marker-edge attach point toward the text bbox attach
                    # display coordinate (attach_x, attach_y) computed above.
                    label_disp = (attach_x, attach_y)
                    attach_marker_x, attach_marker_y = _marker_edge_data_point(ox, oy, label_disp)
                else:
                    attach_marker_x, attach_marker_y = ox, oy
            except Exception:
                attach_marker_x, attach_marker_y = ox, oy

            try:
                conn_color = _select_connector_color(was_sig, ox)
                ax.plot(
                    [attach_marker_x, attach_data[0]],
                    [attach_marker_y, attach_data[1]],
                    color=conn_color,
                    linewidth=cfg.connector_width,
                    alpha=0.8,
                    zorder=3.5,
                )
            except Exception:
                pass
        except Exception:
            pass

        xs = [t.get_position()[0] for t in all_texts]
        ys = [t.get_position()[1] for t in all_texts]
        if xs and ys:
            max_x = max(abs(min(xs)), abs(max(xs)), x_limit)
            ax.set_xlim(-max_x - 0.5, max_x + 0.5)
            max_y = max(max(ys), y_limit)
            ax.set_ylim(bottom=-0.5, top=max_y + 0.5)

    # Expand axis limits slightly by the marker radius (converted from
    # display pixels to data units) so large markers near the plot edge are
    # not visually clipped when saving.
    try:
        # Only expand limits when the caller did not explicitly provide them.
        do_pad_x = getattr(cfg, "xlim", None) is None
        do_pad_y = getattr(cfg, "ylim", None) is None
        # Respect the caller's preference via cfg.pad_by_marker
        if (do_pad_x or do_pad_y) and getattr(cfg, "pad_by_marker", True):
            r_points = math.sqrt(max(cfg.marker_size, 1.0)) / 2.0
            r_pixels = r_points * fig.dpi / 72.0
            # Convert pixel deltas to data-space deltas
            zero_data = ax.transData.inverted().transform((0.0, 0.0))
            dx_data = ax.transData.inverted().transform((r_pixels, 0.0))[0] - zero_data[0]
            dy_data = ax.transData.inverted().transform((0.0, r_pixels))[1] - zero_data[1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            pad_x = abs(dx_data) + 0.05
            pad_y = abs(dy_data) + 0.05
            if do_pad_x:
                ax.set_xlim(x0 - pad_x, x1 + pad_x)
            if do_pad_y:
                ax.set_ylim(y0 - pad_y, y1 + pad_y)
    except Exception:
        pass

    # Respect explicit ticks from config if provided, otherwise keep existing logic
    if getattr(cfg, "xticks", None) is not None:
        try:
            ax.set_xticks(list(cfg.xticks))
        except Exception:
            pass
    elif cfg.xtick_step is not None:
        left = int(math.floor(ax.get_xlim()[0] / cfg.xtick_step) * cfg.xtick_step)
        right = int(math.ceil(ax.get_xlim()[1] / cfg.xtick_step) * cfg.xtick_step)
        ax.set_xticks(list(range(left, right + 1, int(cfg.xtick_step))))

    if getattr(cfg, "yticks", None) is not None:
        try:
            ax.set_yticks(list(cfg.yticks))
        except Exception:
            pass

    plt.tight_layout()
    return fig, ax


class VolcanoPlotter:
    """Stateful, interactive wrapper around the functional API.

    Mirrors the interaction pattern of `OncoPlotter`: the instance
    exposes `.df` and `.config` attributes, and the constructor accepts
    either `(df, config)` or `(config, df)` for backwards compatibility.
    Rendering delegates to `plot_volcano` so the pure function remains
    the canonical implementation.
    """

    def __init__(self, df: pd.DataFrame, config: VolcanoConfig | dict):
        """Construct with `(df, config)` matching `OncoPlotter`.

        `config` may be a `VolcanoConfig` or a dict understood by it.
        """
        if isinstance(config, dict):
            config = VolcanoConfig(**config)
        self.df: pd.DataFrame = df.copy()
        self.last_df: pd.DataFrame = self.df
        self.config: VolcanoConfig = config
        self.cfg: VolcanoConfig = config  # backward alias
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        # history of explicit annotations added via .annotate()
        self.annotation_history: List[Dict] = []

    # Data / rendering -------------------------------------------------
    def set_data(self, df: pd.DataFrame) -> "VolcanoPlotter":
        self.df = df.copy()
        self.last_df = self.df
        return self

    def plot(self, df: Optional[pd.DataFrame] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Render the volcano. If `df` is provided, set it as the current data.

        Returns the `(fig, ax)` produced by `plot_volcano` and stores them
        on the instance.
        """
        if df is not None:
            self.set_data(df)
        if self.df is None or self.config is None:
            raise RuntimeError(
                "Both dataframe and config are required; set them before calling .plot()"
            )
        # Delegate to the canonical function so behavior stays centralized
        self.fig, self.ax = plot_volcano(self.config, self.df)
        return self.fig, self.ax

    def save(self, path: str, **save_kwargs) -> None:
        from pathlib import Path

        if self.fig is None:
            raise RuntimeError("No figure available; call .plot(df) first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, **save_kwargs)

    # Convenience interactive operations --------------------------------
    def update_config(self, **kwargs) -> "VolcanoPlotter":
        """Update configuration in-place and return self for chaining."""
        for k, v in kwargs.items():
            try:
                setattr(self.cfg, k, v)
            except Exception:
                continue
        return self

    def annotate(
        self, explicit_positions: Mapping[str, Tuple[float, float]], replace: bool = True
    ) -> "VolcanoPlotter":
        """Add explicit label placements and re-render.

        `explicit_positions` should be a mapping label -> (x, y).
        If `replace` is True the explicit placements replace auto labels
        (cfg.explicit_label_replace=True). The placements are recorded in
        `annotation_history` so callers can inspect what was added.
        """
        try:
            # normalize to dict[str, (x,y)]
            new_map = {str(k): (float(v[0]), float(v[1])) for k, v in explicit_positions.items()}
        except Exception:
            raise ValueError("explicit_positions must be a mapping label->(x,y)")

        # record
        self.annotation_history.append({"explicit": new_map, "replace": replace})

        # apply to config and re-render
        try:
            if self.config is None:
                self.config = VolcanoConfig(
                    explicit_label_positions=new_map, explicit_label_replace=bool(replace)
                )
            else:
                self.config.explicit_label_positions = new_map
                self.config.explicit_label_replace = bool(replace)
            self.cfg = self.config
        except Exception:
            pass
        # Replot with updated config
        if self.df is not None:
            self.plot(self.df)
        return self

    def label_more(self, n: int = 10) -> "VolcanoPlotter":
        """Convenience to expand `cfg.values_to_label` using the internal
        resolver -- useful for interactive 'label more' flows.
        """
        if self.df is None or self.config is None:
            raise RuntimeError("No dataframe available; call .set_data(df) or .plot(df) first")
        resolved = resolve_labels(self.df, self.config)
        if not resolved:
            return self
        already = (
            list(self.config.values_to_label)
            if getattr(self.config, "values_to_label", None)
            else []
        )
        to_add = [v for v in resolved if v not in already][:n]
        new_vals = list(dict.fromkeys(already + to_add))
        try:
            self.config.values_to_label = new_vals
            self.cfg = self.config
        except Exception:
            pass
        # re-render
        self.plot(self.df)
        return self

    # Serialization / utilities ----------------------------------------
    def to_dict(self) -> Dict:
        try:
            return {
                "cfg": self.config.model_dump() if self.config is not None else {},
                "annotations": list(self.annotation_history),
            }
        except Exception:
            return {"cfg": {}, "annotations": list(self.annotation_history)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "VolcanoPlotter":
        c = data.get("cfg", {})
        vp = cls(c if isinstance(c, VolcanoConfig) else c)
        # restore annotation history if present
        ah = data.get("annotations", None)
        if ah is not None:
            try:
                vp.annotation_history = list(ah)
            except Exception:
                vp.annotation_history = []
        return vp

    def close(self) -> None:
        try:
            if self.fig is not None:
                plt.close(self.fig)
        finally:
            self.fig = None
            self.ax = None

    def resolve_labels(self) -> List[str]:
        if self.last_df is None:
            raise RuntimeError("No dataframe plotted yet; call .plot(df) first")
        return resolve_labels(self.last_df, self.cfg)

    def __enter__(self) -> "VolcanoPlotter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
