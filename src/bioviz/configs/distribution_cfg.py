from typing import Optional, List, Tuple, Any, Literal

from pydantic import Field, model_validator

from .base_cfg import BasePlotConfig


class DistributionConfig(BasePlotConfig):
    bins: int = Field(20, description="Number of bins for histogram")
    show_hist: bool = Field(True, description="Render histogram panel")
    show_box: bool = Field(True, description="Render box+swarm panel")
    xlabel: Optional[str] = Field(None, description="X-axis label (defaults to variable name)")
    ylabel: Optional[str] = Field(None, description="Y-axis label for histogram/boxplot")
    y_ticks: Optional[List[float]] = Field(
        default_factory=lambda: [1], description="Y ticks for boxplot"
    )
    y_ticklabels: Optional[List[str]] = Field(
        default_factory=lambda: [""], description="Y tick labels for boxplot"
    )
    ylim: Optional[Tuple[float, float]] = Field((0.5, 1.5), description="Y-limits for boxplot")
    title_template: Optional[str] = Field(None, description="Default title template")
    title_prefix: Optional[str] = Field(None, description="Optional title prefix")
    alpha: float = Field(1.0, description="Alpha for plot elements")
    grid_alpha: float = Field(0.3, description="Alpha for grid lines")
    style: Optional[Any] = Field(None, description="Optional style object")

    # alpha controls for specific plot elements
    hist_alpha: float = Field(0.7, description="Alpha for histogram bars")
    box_alpha: float = Field(1.0, description="Alpha for box face")

    # ------ Grid controls ------
    hist_grid: bool = Field(False, description="Whether to show grid on histogram")
    box_grid: bool = Field(False, description="Whether to show grid on boxplot panel")

    # ------ Histogram appearance ------
    hist_color: Optional[str] = Field(
        None, description="Explicit histogram color (overrides style)"
    )
    hist_edgecolor: str = Field("black", description="Histogram bar edge color")
    median_color: str = Field("black", description="Color for median line/label")
    median_linestyle: str = Field("--", description="Median line style")
    median_linewidth: float = Field(2.0, description="Median line width")
    median_alpha: float = Field(0.8, description="Median line alpha")
    median_label_fmt: str = Field(
        "Median = {median:.2f}", description="Format for median annotation"
    )
    show_median_label: bool = Field(True, description="Whether to show median annotation")

    # ------ Box + swarm appearance ------
    box_color: Optional[str] = Field(None, description="Box face color (overrides style)")
    swarm_facecolor: str = Field("white", description="Swarm face color")
    swarm_edgecolor: str = Field("black", description="Swarm edge color")
    swarm_linewidth: float = Field(0.5, description="Swarm marker edge linewidth")
    swarm_size: int = Field(40, description="Swarm marker size")
    swarm_alpha: float = Field(0.8, description="Swarm marker alpha")
    jitter_std: float = Field(0.02, description="Swarm jitter standard deviation")
    random_seed: int = Field(42, description="Random seed for swarm jitter")

    # ------ Plot-level controls ------
    return_fig: bool = Field(False, description="Whether to return the figure from plot()")

    # ------ Font sizes ------
    title_fontsize: int = Field(14, description="Font size for plot titles")
    xlabel_fontsize: int = Field(12, description="Font size for x-axis label")
    ylabel_fontsize: int = Field(12, description="Font size for y-axis label")
    xtick_fontsize: int = Field(10, description="Font size for x tick labels")
    ytick_fontsize: int = Field(10, description="Font size for y tick labels")

    # ------ Median label controls ------
    median_label_fontsize: int = Field(10, description="Font size for median annotation")
    median_label_location: Literal["auto", "upper_right", "off_right"] = Field(
        "upper_right", description="Where to place the median label on the histogram"
    )
    show_box_median_label: bool = Field(
        False, description="Whether to show a median label on the boxplot panel"
    )
    box_median_label_location: Literal["auto", "upper_right", "off_right"] = Field(
        "auto", description="Where to place the median label on the boxplot"
    )

    # ------ Grouping / hue support ------
    hue: Optional[str] = Field(None, description="Column name to color/group by (DataFrame input)")
    value_col: Optional[str] = Field(
        None, description="Column name containing numeric values when passing a DataFrame"
    )
    hue_palette: Optional[Any] = Field(
        None, description="Optional mapping of hue category -> color"
    )
    hist_mode: Literal["bar", "kde", "both"] = Field(
        "bar", description="Histogram rendering mode when hue is present"
    )
    hist_hue_overlap: bool = Field(
        True, description="When True, group histograms overlap instead of stacking"
    )
    hue_alpha: Optional[float] = Field(
        None, description="Per-group alpha for overlapped histograms"
    )
    hue_swarm_legend: bool = Field(
        True, description="Whether to show a legend for swarm hue colors"
    )

    # ------ Group median controls ------
    show_group_medians: bool = Field(
        True, description="Draw per-group median markers/lines when hue is set"
    )
    group_median_label: Literal["onplot", "legend", "off"] = Field(
        "legend", description="Where to show group median labels"
    )
    group_median_fmt: str = Field(
        "{group}: {median:.2f}", description="Label format for group medians"
    )
    group_median_marker: str = Field("v", description="Marker for group median on boxplot/hist")
    group_median_markersize: int = Field(8, description="Marker size for group median")

    # optional explicit group order (top-to-bottom in the plot)
    group_order: Optional[List[str]] = Field(
        None, description="Explicit group order for grouped plots (first = top)"
    )

    # convenience single color field: when set, applies to both hist_color and box_color
    color: Optional[str] = Field(
        None, description="Convenience single color applied to both histogram and boxplot"
    )

    @model_validator(mode="after")
    def _apply_single_color(self):
        if getattr(self, "color", None):
            if getattr(self, "hist_color", None) is None:
                object.__setattr__(self, "hist_color", self.color)
            if getattr(self, "box_color", None) is None:
                object.__setattr__(self, "box_color", self.color)
        return self


__all__ = ["DistributionConfig"]
