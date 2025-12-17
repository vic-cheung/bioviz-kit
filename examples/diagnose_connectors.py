from bioviz.configs.volcano_cfg import VolcanoConfig
from bioviz.plots import plot_volcano
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from pathlib import Path

np.random.seed(1)
idx = [f"g{i}" for i in range(1, 11)]
df = pd.DataFrame(
    {
        "log2_or": np.random.uniform(-4, 4, size=10),
        "p_adj": np.random.uniform(0.001, 0.2, size=10),
        "label": idx,
    }
)

cfg = VolcanoConfig(values_to_label=idx[:6], connector_width=1.4, log_transform_ycol=True)
fig, ax = plot_volcano(cfg, df)

# Find horizontal lines (Line2D) and their y data

lines = [a for a in ax.get_lines() if isinstance(a, Line2D)]
print("Total Line2D objects on axes:", len(lines))
for i, ln in enumerate(lines):
    xdata = ln.get_xdata()
    ydata = ln.get_ydata()
    print(i, "xdata len", len(xdata), "ydata unique", set([round(float(v), 3) for v in ydata]))

# Count lines that look like connectors: short lines not spanning axis
connector_like = []
for ln in lines:
    xdata = np.array(ln.get_xdata())
    ydata = np.array(ln.get_ydata())
    # skip verticals or full-span lines
    if np.ptp(xdata) < 5 and np.ptp(ydata) < 5:
        connector_like.append(ln)

print("Connector-like Line2D count:", len(connector_like))

# Inspect horizontal threshold line y coordinate
hlines = [ln for ln in lines if len(set(np.round(ln.get_ydata(), 6))) == 1]
print("Horizontal Line2D count (constant y):", len(hlines))
for ln in hlines:
    print("  y =", float(ln.get_ydata()[0]))

fig.canvas.draw()

Path("examples").mkdir(parents=True, exist_ok=True)
fig.savefig("examples/diagnose_connectors_out.png")
print("Wrote examples/diagnose_connectors_out.png")
