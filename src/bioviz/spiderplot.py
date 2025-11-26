"""
Spider plot utilities (top-level bioviz module)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

from bioviz.plot_configs import (
    ScanOverlayPlotConfig,
    StyledSpiderPlotConfig,
)
from bioviz.plot_utils import adjust_legend, forward_fill_groups
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

__all__ = [
    "generate_styled_spiderplot",
    "generate_styled_spiderplot_with_scan_overlay",
]

def generate_styled_spiderplot(df: pd.DataFrame, config: StyledSpiderPlotConfig, ax: plt.Axes | None = None, draw_legend: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure
    return fig

def generate_styled_spiderplot_with_scan_overlay(df, scan_data, spider_config, scan_overlay_config, recist_color_dict=None):
    fig, ax = plt.subplots()
    return fig
