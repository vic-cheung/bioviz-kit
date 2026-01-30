"""
Example usage of bioviz DistributionPlotter and modeling wrapper
Creates two example plots using random data and saves them to the examples folder.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from bioviz.configs.distribution_cfg import DistributionConfig
from bioviz.plots.distribution import DistributionPlotter

OUT = Path(__file__).resolve().parent

# Example 1: Direct DistributionPlotter usage
s = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=200))
cfg = DistributionConfig(title="Direct DistributionPlotter Example", swarm_size=25)
plotter = DistributionPlotter(data=s, config=cfg)
fig, axes = plotter.plot(return_fig=True)
fig.savefig(OUT / "distribution_plotter_example.png", bbox_inches="tight")
plotter.close()
print("Saved:", OUT / "distribution_plotter_example.png")

# %%

# Example 2: Grouped DataFrame usage (hue/value_col)
np.random.seed(1)
df = pd.DataFrame(
    {
        "value": np.concatenate(
            [
                np.random.normal(loc=-1, scale=0.8, size=150),
                np.random.normal(loc=1, scale=0.6, size=150),
            ]
        ),
        "group": ["A"] * 150 + ["B"] * 150,
    }
)

cfg2 = DistributionConfig(
    title="Grouped Distribution Example",
    hue="group",
    value_col="value",
    hist_mode="both",
    show_box=True,
    show_hist=True,
    swarm_size=20,
    return_fig=True,
)

plotter2 = DistributionPlotter(data=df, config=cfg2)
fig2, axes2 = plotter2.plot(return_fig=True)
fig2.savefig(OUT / "distribution_grouped_example.png", bbox_inches="tight")
plotter2.close()
print("Saved:", OUT / "distribution_grouped_example.png")

# %%

# Example 3: Explicit group_order (visual top-to-bottom)
np.random.seed(2)
df3 = pd.DataFrame(
    {
        "value": np.concatenate(
            [
                np.random.normal(loc=-2, scale=0.5, size=80),
                np.random.normal(loc=0.5, scale=0.7, size=80),
                np.random.normal(loc=2, scale=0.6, size=80),
            ]
        ),
        "group": ["C"] * 80 + ["B"] * 80 + ["A"] * 80,
    }
)

cfg3 = DistributionConfig(
    title="Grouped with Explicit Order",
    hue="group",
    value_col="value",
    show_box=True,
    show_hist=False,
    swarm=True,
    swarm_size=20,
    group_order=["A", "B", "C"],
    return_fig=True,
)

plotter3 = DistributionPlotter(data=df3, config=cfg3)
fig3, axes3 = plotter3.plot(return_fig=True)
fig3.savefig(OUT / "distribution_group_order_example.png", bbox_inches="tight")
plotter3.close()
print("Saved:", OUT / "distribution_group_order_example.png")

# Example 4: Categorical dtype ordering (ordered categories respected)
np.random.seed(3)
df4 = pd.DataFrame(
    {
        "value": np.concatenate(
            [
                np.random.normal(loc=-1, scale=0.4, size=60),
                np.random.normal(loc=0.0, scale=0.5, size=60),
                np.random.normal(loc=1.5, scale=0.6, size=60),
            ]
        ),
        "group": ["low"] * 60 + ["mid"] * 60 + ["high"] * 60,
    }
)

cat_type = pd.CategoricalDtype(categories=["high", "mid", "low"], ordered=True)
df4["group"] = df4["group"].astype(cat_type)

cfg4 = DistributionConfig(
    title="Grouped with Categorical Order (high, mid, low)",
    hue="group",
    value_col="value",
    show_box=True,
    show_hist=True,
    swarm=True,
    swarm_size=18,
    return_fig=True,
)

plotter4 = DistributionPlotter(data=df4, config=cfg4)
fig4, axes4 = plotter4.plot(return_fig=True)
fig4.savefig(OUT / "distribution_categorical_order_example.png", bbox_inches="tight")
plotter4.close()
print("Saved:", OUT / "distribution_categorical_order_example.png")

# %%
