"""
Line plot utilities (top-level bioviz module)
"""
from adjustText import adjust_text
from matplotlib import font_manager
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bioviz.plot_configs import StyledLinePlotConfig
from bioviz.plot_utils import forward_fill_groups
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

__all__ = ["generate_styled_lineplot"]

def generate_styled_lineplot(df: pd.DataFrame, config: StyledLinePlotConfig, ax: plt.Axes | None = None):
    # minimal implementation to preserve API surface
    if df.empty:
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure
    sns.lineplot(data=df, x=config.x, y=config.y, hue=config.label_col, ax=ax)
    return fig
