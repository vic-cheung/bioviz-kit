"""
Table utilities (top-level bioviz module)
"""
import matplotlib.pyplot as plt
import pandas as pd

from bioviz.plot_configs import StyledTableConfig
from bioviz.style import DefaultStyle

DefaultStyle().apply_theme()

__all__ = ["generate_styled_table"]

def generate_styled_table(df: pd.DataFrame, config: StyledTableConfig, ax: plt.Axes | None = None):
    if df.empty:
        return None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.axis('off')
    return fig
