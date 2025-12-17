from bioviz.plot_composites.configs.volcano_cfg import VolcanoConfig
from bioviz.plot_composites.volcano import plot_volcano
import pandas as pd
import numpy as np

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
from matplotlib.lines import Line2D

lines = [a for a in ax.get_lines() if isinstance(a, Line2D)]
print("Total Line2D objects on axes:", len(lines))
for i, l in enumerate(lines):
    xdata = l.get_xdata()
    ydata = l.get_ydata()
    print(i, "xdata len", len(xdata), "ydata unique", set([round(float(v), 3) for v in ydata]))

# Count lines that look like connectors: short lines not spanning axis
connector_like = []
for l in lines:
    xdata = np.array(l.get_xdata())
    ydata = np.array(l.get_ydata())
    # skip verticals or full-span lines
    if np.ptp(xdata) < 5 and np.ptp(ydata) < 5:
        connector_like.append(l)

print("Connector-like Line2D count:", len(connector_like))

# Inspect horizontal threshold line y coordinate
hlines = [l for l in lines if len(set(np.round(l.get_ydata(), 6))) == 1]
print("Horizontal Line2D count (constant y):", len(hlines))
for l in hlines:
    print("  y =", float(l.get_ydata()[0]))

fig.canvas.draw()
fig.savefig("examples/diagnose_connectors_out.png")
print("Wrote examples/diagnose_connectors_out.png")
