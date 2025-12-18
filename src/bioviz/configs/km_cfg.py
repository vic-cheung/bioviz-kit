from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KMPlotConfig:
    # Required
    time_col: str
    event_col: str
    group_col: str

    # Layout
    fig_width: float = 10.0
    fig_height: float = 6.0
    title: Optional[str] = None

    # Axis / ticks
    xlim: Optional[tuple] = None
    ylim: Optional[tuple] = (0, 1.0)
    xlab: str = "Time (Months)"
    ylab: str = "Survival Probability"
    xticks: Optional[list] = None
    xtick_interval_months: Optional[float] = 3.0

    # Visuals
    show_risktable: bool = True
    show_ci: bool = True
    ci_style: str = "fill"
    ci_alpha: float = 0.25
    linewidth: float = 3.0
    linestyle: str = "-"
    censor_marker: str = "+"
    censor_markersize: float = 12.0
    censor_markeredgewidth: float = 2.5

    # Legend / labels
    legend_loc: str = "bottom"
    legend_title: Optional[str] = None
    legend_fontsize: int = 16
    legend_frameon: bool = False
    legend_show_n: bool = False

    # Risk table styling
    risktable_fontsize: int = 18
    risktable_row_spacing: float = 1.8
    risktable_title_gap_factor: float = 0.6

    # Misc
    color_dict: Optional[Dict[Any, str]] = field(default_factory=dict)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out.update(out.pop("extra_kwargs", {}))
        return out
