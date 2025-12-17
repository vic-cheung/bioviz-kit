from __future__ import annotations

from typing import Optional, Iterable, List, Dict, Tuple, Any, Literal

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, validator


class VolcanoConfig(BaseModel):
    # ------ Data Columns ------
    # Required: names of the dataframe columns to use for plotting
    x_col: str = Field(..., description="Name of the x-axis column (e.g. log2_or)")
    y_col: str = Field(..., description="Name of the y-axis column (e.g. p_adj)")

    # Label col not required; if provided, used to match labels to points
    # if no label_col is provided, labels are matched by point index
    label_col: Optional[str] = None

    # ------ Selection & Thresholds ------
    values_to_label: Optional[List[str]] = None
    additional_values_to_label: Optional[List[str]] = None

    label_mode: Literal[
        "auto",
        "sig",
        "sig_and_thresh",
        "thresh",
        "sig_or_thresh",
        "all",
    ] = Field(
        "auto",
        description=(
            "Controls how labels are selected when `values_to_label` is not provided. "
            "Options: 'auto' (default: label points considered significant by `y_col_thresh`), 'sig' (y threshold only), "
            "'sig_and_thresh' (require both y threshold and |x| >= `abs_x_thresh`), 'thresh' (x magnitude only), "
            "'sig_or_thresh' (union), and 'all' (label every point)."
        ),
    )

    y_col_thresh: float = Field(
        0.05,
        description=(
            "Numeric cutoff applied to `y_col` to mark significance. "
            "(e.g., p-value threshold). Formerly named `sig_thresh`; "
            "the name was changed to avoid implying the column must be a p-value."
        ),
    )

    abs_x_thresh: float = Field(
        2.0,
        description=(
            "Absolute x-axis magnitude threshold used by some label/"
            "color selection modes (points with |x| >= this value are considered large)."
        ),
    )

    y_thresh: Optional[float] = Field(
        None,
        description=(
            "Optional y-axis threshold position (in data units). "
            "When set, a horizontal threshold line is drawn at this value after any configured transform."
        ),
    )

    x_thresh: Optional[Iterable[float]] = Field(
        None,
        description=(
            "Optional x-axis threshold line positions. Provide an iterable of numeric positions to draw vertical threshold lines."
        ),
    )

    # Threshold styling and per-axis overrides
    thresh_line_color: str = Field(
        "gainsboro",
        description="Default color used for threshold lines (applies when per-axis override is not provided).",
    )

    thresh_line_style: str = Field(
        "--",
        description="Line style used for threshold lines (e.g. '--').",
    )

    thresh_line_width: float = Field(
        1.0,
        description="Line width used for threshold lines (applies when per-axis override is not provided).",
    )

    x_thresh_line_color: Optional[str] = Field(
        None,
        description="Optional color override for x-axis threshold lines.",
    )

    x_thresh_line_style: Optional[str] = Field(
        None,
        description="Optional line-style override for x-axis threshold lines.",
    )

    x_thresh_line_width: Optional[float] = Field(
        None,
        description="Optional line-width override for x-axis threshold lines.",
    )

    y_thresh_line_color: Optional[str] = Field(
        None,
        description="Optional color override for y-axis threshold lines.",
    )

    y_thresh_line_style: Optional[str] = Field(
        None,
        description="Optional line-style override for y-axis threshold lines.",
    )

    y_thresh_line_width: Optional[float] = Field(
        None,
        description="Optional line-width override for y-axis threshold lines.",
    )

    # ------ Coloring & Direction ------
    direction_col: Optional[str] = None
    direction_colors: Optional[Dict[str, str]] = None
    palette: Dict[str, str] = Field(
        default_factory=lambda: {
            "nonsig": "gainsboro",
            "sig_up": "#009E73",
            "sig_down": "#D55E00",
        }
    )

    color_mode: Literal["sig", "thresh", "sig_and_thresh", "sig_or_thresh", "all"] = Field(
        "sig",
        description=(
            "Controls how point colors are assigned relative to thresholds/significance. "
            "Options: 'sig', 'thresh', 'sig_and_thresh', 'sig_or_thresh', 'all'."
        ),
    )

    # ------ Labeling & Annotation ------
    label_offset_mode: str = Field(
        "fraction",
        description=(
            "How to interpret `label_offset`: 'fraction' interprets the value as a fraction of the x-axis span, "
            "'data' treats it as raw data units, and 'axes' interprets it as an axis fraction (0..1) converted to data units."
        ),
    )

    label_offset: float = Field(
        0.03,
        description=(
            "Default label offset used when `label_offset_mode` == 'fraction' (fraction of x-axis span)."
        ),
    )

    force_label_side_by_point_sign: bool = Field(
        False,
        description=(
            "If True, force labels to appear on the outward side of each point: left when x<0, right when x>0. "
            "When False, labels may be placed by other rules or `adjust_text`."
        ),
    )

    force_labels_adjustable: bool = Field(
        False,
        description=(
            "When True, forced outward labels are included in the `adjust_text` pass and may be moved to avoid overlaps."
        ),
    )

    annotation_fontweight_sig: str = Field(
        "bold",
        description="Font weight used for labels on significant points (default: 'bold').",
    )

    annotation_fontweight_nonsig: str = Field(
        "normal",
        description="Font weight used for labels on non-significant points (default: 'normal').",
    )

    annotation_sig_color: Optional[str] = None
    annotation_nonsig_color: str = "#7f7f7f"

    horiz_offset_range: Tuple[float, float] = (0.02, 0.06)
    vert_jitter_range: Tuple[float, float] = (-0.03, 0.03)

    use_adjust_text: bool = Field(
        True,
        description=(
            "Whether to use the `adjust_text` package (if available) to tidy label positions before drawing connectors."
        ),
    )

    adjust: bool = True

    # Whether to transform the y-column using -log10 (e.g., p-values -> -log10(p))
    log_transform_ycol: bool = Field(
        False,
        description=("When True, the y column will be transformed with -log10 before plotting."),
    )

    # Nudging / label layout knobs used by the plotting code but optional
    nudge_padding_pixels: float = Field(
        6.0,
        description=(
            "Display-space padding used when nudging labels away from nearby markers (pixels)."
        ),
    )

    horiz_offset_range: Tuple[float, float] = Field(
        (0.02, 0.06),
        description=("Range (lo,hi) for horizontal offset fractions used when placing labels."),
    )

    vert_jitter_range: Tuple[float, float] = Field(
        (-0.03, 0.03),
        description=(
            "Range (lo,hi) for vertical jitter applied to labels as fraction of axis span."
        ),
    )

    # ------ Layout & Axes ------
    x_label: Optional[str] = Field(
        None,
        description="Optional x-axis label (string or TeX). If not provided a sensible default is used.",
    )

    y_label: Optional[str] = Field(
        None,
        description="Optional y-axis label (string or TeX). If not provided a sensible default is used.",
    )

    xlim: Optional[Tuple[float, float]] = Field(
        None,
        description=(
            "Optional explicit x-axis limits (min, max) in data units. When provided these override automatic expansion."
        ),
    )

    ylim: Optional[Tuple[float, float]] = Field(
        None,
        description=(
            "Optional explicit y-axis limits (min, max) in data units. When provided these override automatic expansion."
        ),
    )

    xticks: Optional[Iterable[float]] = Field(
        None,
        description="Optional explicit x-tick locations (iterable of numeric values).",
    )

    yticks: Optional[Iterable[float]] = Field(
        None,
        description="Optional explicit y-tick locations (iterable of numeric values).",
    )

    xtick_step: Optional[int] = None
    fontsize_sig: int = 12
    fontsize_nonsig: int = 11
    tick_label_fontsize: int = 16
    axis_label_fontsize: int = 18
    title: Optional[str] = None
    title_fontsize: int = 20
    figsize: Tuple[int, int] = (5, 5)
    group_label_top: Optional[Tuple[str, str]] = None
    group_label_kwargs: Optional[Dict] = None

    # ------ Marker & Connectors ------
    marker_size: float = 50.0
    attach_to_marker_edge: bool = True
    pad_by_marker: bool = Field(
        True,
        description=(
            "When True, expand axis limits by the marker display radius so large markers near the edge are not clipped. "
            "Set False to preserve exact axis limits (useful when caller set `xlim`/`ylim` explicitly)."
        ),
    )

    connector_color: str = "gray"
    connector_width: float = 0.8
    connector_color_sig: Optional[str] = Field(
        None,
        description="Optional connector color for significant points (falls back to `connector_color` if None).",
    )

    connector_color_nonsig: Optional[str] = Field(
        None,
        description="Optional connector color for non-significant points (falls back to `connector_color` if None).",
    )

    connector_color_left: Optional[str] = Field(
        None,
        description="Optional connector color override for left-side labels.",
    )

    connector_color_right: Optional[str] = Field(
        None,
        description="Optional connector color override for right-side labels.",
    )

    connector_color_sig_left: Optional[str] = Field(
        None,
        description="Most-specific override: connector color for significant points on the left side.",
    )

    connector_color_sig_right: Optional[str] = Field(
        None,
        description="Most-specific override: connector color for significant points on the right side.",
    )

    connector_color_nonsig_left: Optional[str] = Field(
        None,
        description="Most-specific override: connector color for non-significant points on the left side.",
    )

    connector_color_nonsig_right: Optional[str] = Field(
        None,
        description="Most-specific override: connector color for non-significant points on the right side.",
    )

    connector_color_use_point_color: bool = False

    # ------ Execution / Explicit placements ------
    ax: Optional[plt.Axes] = None
    explicit_label_positions: Optional[Any] = Field(
        None,
        description=(
            "Optional explicit label positions. Accepts a dict label->(x,y), an iterable of (label,(x,y)), "
            "or a pandas DataFrame with columns ('label','x','y') or matching `label_col`/`x_col`/`y_col`."
        ),
    )

    explicit_label_replace: bool = Field(
        True,
        description=(
            "When True, labels provided in `explicit_label_positions` replace automatic labeling for those labels. "
            "When False, explicit positions are added alongside automatic labels."
        ),
    )

    explicit_label_adjustable: bool = Field(
        False,
        description=(
            "When True, explicit labels participate in the `adjust_text` flow and may be moved; otherwise explicit positions are respected."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    @validator("palette")
    def _ensure_palette_keys(cls, v):
        # Ensure minimal palette keys exist
        v = dict(v)
        v.setdefault("nonsig", "gainsboro")
        if "sig_up" not in v and "sig" in v:
            v.setdefault("sig_up", v.get("sig"))
        v.setdefault("sig_up", "#009E73")
        v.setdefault("sig_down", "#D55E00")
        return v
