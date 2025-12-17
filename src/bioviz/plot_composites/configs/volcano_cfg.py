from __future__ import annotations

from typing import Optional, Iterable, List, Dict, Tuple

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, validator


class VolcanoConfig(BaseModel):
    # Data columns: x_col and y_col are required; label_col is optional.
    x_col: str = "log2_or"
    y_col: str = "p_adj"
    label_col: Optional[str] = "label"

    # Label selection
    values_to_label: Optional[List[str]] = None
    additional_values_to_label: Optional[List[str]] = None
    # Generic significance threshold (e.g. p-value threshold). If your
    # dataframe has a column of p-values (named arbitrarily) set this to
    # the numeric cutoff to mark significance.
    sig_thresh: float = 0.05
    abs_x_thresh: float = 2.0
    sig_only: bool = True

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

    # Execution
    ax: Optional[plt.Axes] = None

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
