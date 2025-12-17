from bioviz.configs.volcano_cfg import VolcanoConfig
from bioviz.plots import plot_volcano
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

np.random.seed(1)
idx = [f"g{i}" for i in range(1, 11)]
df = pd.DataFrame(
    {
        "log2_or": np.random.uniform(-4, 4, size=10),
        "p_adj": np.random.uniform(0.001, 0.2, size=10),
        "label": idx,
    }
)

configs = []
configs.append(("default", VolcanoConfig(values_to_label=idx[:6], label_col="label")))
configs.append(
    (
        "custom",
        VolcanoConfig(
            values_to_label=idx[:6],
            connector_color="#ff00aa",
            connector_width=1.4,
            log_transform_ycol=True,
            label_col="label",
        ),
    )
)
configs.append(
    (
        "center",
        VolcanoConfig(
            values_to_label=idx[:6],
            attach_to_marker_edge=False,
            connector_color="#0077cc",
            log_transform_ycol=True,
            label_col="label",
        ),
    )
)
configs.append(
    (
        "hier",
        VolcanoConfig(
            values_to_label=idx[:6],
            connector_color_sig_left="#880000",
            connector_color_sig_right="#008800",
            connector_color_nonsig_left="#888800",
            connector_color_nonsig_right="#000088",
            connector_color_left="#ff00aa",
            connector_color_right="#00aaff",
            connector_color_sig="#aa00ff",
            connector_color_nonsig="#00ffaa",
            label_col="label",
        ),
    )
)
configs.append(
    (
        "explicit_replace",
        VolcanoConfig(
            explicit_label_positions={"g1": (-0.5, 1.0), "g2": (2.5, 0.5)},
            explicit_label_replace=True,
            label_col="label",
        ),
    )
)
configs.append(
    (
        "explicit_add",
        VolcanoConfig(
            values_to_label=idx[:6],
            explicit_label_positions=[("g1", (-0.5, 1.0)), ("g2", (2.5, 0.5))],
            explicit_label_replace=False,
            label_col="label",
        ),
    )
)

for name, cfg in configs:
    fig, ax = plot_volcano(cfg, df)
    # collect lines
    lines = [a for a in ax.get_lines() if isinstance(a, Line2D)]
    # identify connector-like lines (short segments)
    connector_like = []
    for ln in lines:
        xdata = np.array(ln.get_xdata())
        ydata = np.array(ln.get_ydata())
        if (
            np.ptp(xdata) < (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.9
            and np.ptp(ydata) < (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.9
        ):
            connector_like.append(ln)
    # find horizontal threshold
    thr_y = None
    hlines = [ln for ln in lines if len(set(np.round(ln.get_ydata(), 6))) == 1]
    if hlines:
        thr_y = float(hlines[-1].get_ydata()[0])
    print(f"{name}: total Line2D={len(lines)}, connector_like={len(connector_like)}, thr_y={thr_y}")
    fig.savefig(f"examples/diagnose_{name}.png")
    print(f"  wrote examples/diagnose_{name}.png")

print("Done")
