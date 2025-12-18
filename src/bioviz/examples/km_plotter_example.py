"""KMPlotter example: generates a synthetic KM plot using KMPlotter.from_dataframe

Run:
    python3 -m examples.km_plotter_example
or
    python3 src/bioviz/examples/km_plotter_example.py

This script requires matplotlib, lifelines, pandas, numpy.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from bioviz.configs.km_cfg import KMPlotConfig
from bioviz.plots.km_plotter import KMPlotter


def make_synthetic_data():
    rng = np.random.RandomState(1)
    n = 120
    groups = np.random.choice(["A", "B"], size=n, p=[0.5, 0.5])
    times = np.concatenate([rng.exponential(12, size=n // 2), rng.exponential(8, size=n // 2)])
    events = (times < 24).astype(int)
    df = pd.DataFrame({"time": times, "event": events, "arm": groups})
    return df


if __name__ == "__main__":
    df = make_synthetic_data()
    cfg = KMPlotConfig(time_col="time", event_col="event", group_col="arm", legend_show_n=True)

    plotter = KMPlotter.from_dataframe(df, cfg)
    fig, ax, p = plotter.plot(cfg, output_path=None)
    print("Generated KM plot; p-value:", p)
    fig.show()
