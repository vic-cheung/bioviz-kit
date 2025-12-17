from __future__ import annotations

from typing import Optional, Iterable, List, Dict, Tuple, Any, Literal

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, validator


class VolcanoConfig(BaseModel):
    # Data columns: x_col and y_col are required; label_col is optional.
    x_col: str = "log2_or"
    y_col: str = "p_adj"
    label_col: Optional[str] = None

    # Label selection
    values_to_label: Optional[List[str]] = None
    additional_values_to_label: Optional[List[str]] = None
    # Generic threshold for the `y_col` (e.g. p-value cutoff). If your
    # dataframe has a column of values to threshold (named arbitrarily),
    # set this to the numeric cutoff to mark significance. The name was
    # changed from `sig_thresh` to `y_col_thresh` to avoid implying the
    # column must be a p-value.
    y_col_thresh: float = 0.05
    abs_x_thresh: float = 2.0
    # Whether significance requires both y threshold AND abs(x) >= abs_x_thresh.
    # When True (default) a point is considered significant only if it meets
    # the y threshold and the x magnitude threshold. When False, significance
    # is determined by the y threshold alone.
    sig_requires_x_thresh: bool = True
    # `sig_only` was removed; prefer using `label_mode` to control label selection.
    # The plotting code treats `label_mode='auto'` as: label points that are
    # significant AND beyond the x-axis threshold (i.e. intersection), which
    # matches the simplified, less-ambiguous default behavior.

    # Direction / coloring
    direction_col: Optional[str] = None
    direction_colors: Optional[Dict[str, str]] = None
    palette: Dict[str, str] = Field(
        default_factory=lambda: {
            "nonsig": "gainsboro",
            "sig_up": "#009E73",
            "sig_down": "#D55E00",
        }
    )

    # Plot appearance
    # Y-axis threshold (e.g., p-value line rendered on y axis after transform)
    y_thresh: Optional[float] = None
    # X-axis threshold lines (explicit positions)
    x_thresh: Optional[Iterable[float]] = None
    # Threshold line style (applies to both x and y threshold lines)
    # Default threshold line styling (can be overridden per-axis)
    thresh_line_color: str = "gainsboro"
    thresh_line_style: str = "--"
    thresh_line_width: float = 1.0
    # Per-axis overrides: if provided these will be used for the x or y
    # threshold lines respectively. Each can be None (fall back to the
    # generic `thresh_line_*` values) or a specific value.
    x_thresh_line_color: Optional[str] = None
    x_thresh_line_style: Optional[str] = None
    x_thresh_line_width: Optional[float] = None
    y_thresh_line_color: Optional[str] = None
    y_thresh_line_style: Optional[str] = None
    y_thresh_line_width: Optional[float] = None
    xtick_step: Optional[int] = None
    fontsize_sig: int = 12
    fontsize_nonsig: int = 11
    adjust: bool = True
    figsize: Tuple[int, int] = (5, 5)
    # Explicit control: whether to apply a -log10 transform to `y_col` values.
    # Use True to perform the transform, False to leave values as-is.
    log_transform_ycol: bool = False
    # Top group labels (left, right) — if not provided infer from direction_col
    group_label_top: Optional[Tuple[str, str]] = None
    group_label_kwargs: Optional[Dict] = None
    # Title and font sizes
    title: Optional[str] = None
    title_fontsize: int = 20
    axis_label_fontsize: int = 18
    tick_label_fontsize: int = 16
    # Optional explicit axis label overrides (raw strings or TeX)
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    # Optional explicit axis limits (data units). When provided these override
    # automatic expansion performed by the plot routine.
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    # Optional explicit tick locations. If provided, these will be set on the
    # axes after plotting. Use lists/tuples of numeric ticks.
    xticks: Optional[Iterable[float]] = None
    yticks: Optional[Iterable[float]] = None
    # Annotation (point label) styling
    annotation_fontweight_sig: str = "bold"
    annotation_fontweight_nonsig: str = "normal"
    annotation_sig_color: Optional[str] = None
    annotation_nonsig_color: str = "#7f7f7f"
    # Force labels outward by point sign: left for x<0, right for x>0
    force_label_side_by_point_sign: bool = False
    # Horizontal label offset (data units) when forcing side
    label_offset: float = 0.6
    # How to interpret label offsets: 'fraction' of axis span, 'data' units, or 'axes' fraction
    label_offset_mode: str = "fraction"
    # When in 'fraction' mode, `label_offset` is proportion of x-axis span (0.05 = 5%)
    # When in 'axes' mode, offsets are in axis fraction (0..1) and converted to data units
    # When in 'data' mode, offsets are interpreted as raw data units (legacy behavior)
    # Default label_offset acts as fraction of axis span when mode=='fraction'
    label_offset: float = 0.03

    # How labels are selected when `values_to_label` is not provided.
    # Allowed values:
    # - 'auto': default. Label points that are considered significant according
    #   to `y_col_thresh`. If `sig_requires_x_thresh` is True, the point must
    #   also meet the `abs_x_thresh` magnitude (i.e. both tests) to be
    #   considered significant for 'auto'. This is the recommended behavior.
    # - 'sig': label only points that meet the y threshold (transformed if
    #   `log_transform_ycol` is True) regardless of x magnitude.
    # - 'sig_and_thresh': label only points that meet BOTH the y threshold
    #   and the |x| >= `abs_x_thresh` requirement. This is equivalent to the
    #   intersection of the two tests and is useful when you want to require
    #   both criteria explicitly (different from 'sig', which ignores x).
    # - 'thresh': label points that meet the x-axis magnitude threshold
    #   (|x| >= `abs_x_thresh`) regardless of the y threshold.
    # - 'sig_or_thresh': label points that either meet the y threshold OR have
    #   |x| >= `abs_x_thresh` (union of the two tests).
    # - 'all': label every point in the dataset.
    # Note: `sig_requires_x_thresh` is retained for backward compatibility
    # and influences the meaning of 'auto'. Prefer using explicit
    # `label_mode` values ('sig', 'thresh', 'sig_and_thresh', 'sig_or_thresh')
    # for unambiguous behavior.
    label_mode: Literal[
        "auto",
        "sig",
        "sig_and_thresh",
        "thresh",
        "sig_or_thresh",
        "all",
    ] = "auto"

    # Control how points are colored with respect to thresholds/significance.
    # Allowed values:
    # - 'sig': color points considered significant by y threshold (and
    #    optionally x magnitude if `sig_requires_x_thresh` is True).
    # - 'thresh': color points that meet the x-axis magnitude threshold
    #    (|x| >= `abs_x_thresh`) only.
    # - 'sig_and_thresh': color points that meet BOTH the y threshold and
    #    the x magnitude threshold.
    # - 'sig_or_thresh': color points that meet either the y threshold OR
    #    the x magnitude threshold.
    # - 'all': color all points as 'sig' (useful for debugging or forcing
    #    single-color annotation behavior).
    color_mode: Literal["sig", "thresh", "sig_and_thresh", "sig_or_thresh", "all"] = "sig"

    # Ranges used for adjustable label horizontal offset and vertical jitter.
    # Interpreted according to `label_offset_mode` (fractions if mode='fraction').
    horiz_offset_range: Tuple[float, float] = (0.02, 0.06)
    vert_jitter_range: Tuple[float, float] = (-0.03, 0.03)

    # If True, forced labels (outward-side) are included in the adjust_text pass
    # so they may be moved to avoid overlapping other labels. Default False.
    force_labels_adjustable: bool = False
    # Whether to use adjust_text to tidy text positions before drawing connectors
    use_adjust_text: bool = True
    # Marker size for scatter (points)
    marker_size: float = 50.0
    # Whether connectors should attach to the marker edge instead of the center
    attach_to_marker_edge: bool = True
    # When True, expand axis limits by the marker display radius so large
    # markers near the edge are not visually clipped when saving figures.
    # Set to False to preserve exact axis limits (useful when caller set
    # `xlim`/`ylim` explicitly and does not want automatic padding).
    pad_by_marker: bool = True
    # Connector (annotation line) styling
    connector_color: str = "gray"
    connector_width: float = 0.8
    # Optional per-category connector colors. Each can be None to fall back
    # to the generic `connector_color`.
    connector_color_sig: Optional[str] = None
    connector_color_nonsig: Optional[str] = None
    # Per-side overrides (left/right) useful when you want different colors
    # for left vs right labels (e.g., negative vs positive x).
    connector_color_left: Optional[str] = None
    connector_color_right: Optional[str] = None
    # Optional explicit per-(significance × side) colors. These are the most
    # specific overrides and will be consulted first when selecting connector
    # colors. Each can be None to fall back to the less-specific fields.
    connector_color_sig_left: Optional[str] = None
    connector_color_sig_right: Optional[str] = None
    connector_color_nonsig_left: Optional[str] = None
    connector_color_nonsig_right: Optional[str] = None

    # When True, connectors inherit the marker color for the matched data
    # point (useful when you want connectors to visually match point color).
    connector_color_use_point_color: bool = False

    # Execution
    ax: Optional[plt.Axes] = None
    # Explicit label placements: allows the caller to provide explicit
    # positions for particular labels. Accepts one of:
    # - dict mapping label -> (x, y)
    # - iterable of (label, (x, y)) tuples
    # - pandas.DataFrame with columns ('label','x','y') or with label column
    #   matching `label_col` and coordinate columns matching `x_col`/`y_col` or 'x'/'y'.
    explicit_label_positions: Optional[Any] = None
    # If True, labels provided in `explicit_label_positions` will replace
    # any automatic labeling for those labels (they won't also be auto-placed).
    # If False, explicit labels will be placed in addition to any automatic
    # labels selected by `values_to_label` or significance rules.
    explicit_label_replace: bool = True
    # If True, explicit labels participate in the adjust_text flow and may be
    # moved by `adjust_text`. Default False (explicit positions are respected
    # unless the user opts into adjustment).
    explicit_label_adjustable: bool = False

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
