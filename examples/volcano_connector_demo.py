"""Minimal demo showing connector customization options for plot_volcano.
Writes two PNGs demonstrating different connector colors/widths and attach behavior.
"""

import pandas as pd
import numpy as np
from bioviz.configs.volcano_cfg import VolcanoConfig
from bioviz.plots import plot_volcano

# Create a minimal dataframe
np.random.seed(1)
idx = [f"g{i}" for i in range(1, 11)]
df = pd.DataFrame(
    {
        "log2_or": np.random.uniform(-4, 4, size=10),
        "p_adj": np.random.uniform(0.001, 0.2, size=10),
        "label": idx,
    }
)


def write_demo(cfg, fname):
    fig, ax = plot_volcano(cfg, df)
    # Ensure target directory exists
    from pathlib import Path

    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    print("Wrote", fname)


# Default behavior (attach to marker edge, gray connectors)
cfg_default = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    values_to_label=idx[:6],
    label_col="label",
)
write_demo(cfg_default, "examples/volcano_connector_demo.default.png")

# Custom connectors
cfg_custom = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    # values_to_label=idx[:6], <--if nothing is passed, automatically labels sig points
    connector_color="#ff00aa",
    connector_width=1.4,
    log_transform_ycol=True,
    label_col="label",
    label_mode="auto",
    color_mode="sig_or_thresh",
)
write_demo(cfg_custom, "examples/volcano_connector_demo.custom.png")
cfg_custom.additional_values_to_label = idx[:6]
write_demo(cfg_custom, "examples/volcano_connector_demo.custom.with_added_labels.png")

# Disable edge-attach (connect to center)
cfg_center = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    # values_to_label=idx[:6],
    attach_to_marker_edge=False,
    connector_color="#0077cc",
    log_transform_ycol=True,
    label_col="label",
    label_mode="sig_and_thresh",
    color_mode="sig_and_thresh",
)

write_demo(cfg_center, "examples/volcano_connector_demo.center.png")

# Hierarchical connector color demo: set explicit sign+side colors so we can
# visually verify the precedence (sig_left, sig_right, nonsig_left, nonsig_right)
cfg_hier = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
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
    log_transform_ycol=True,
    label_mode="all",
    color_mode="all",
)
write_demo(cfg_hier, "examples/volcano_connector_demo.hier.png")

# Explicit label placements: dict-style (replace defaults)
cfg_explicit_replace = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    explicit_label_positions={"g1": (-0.5, 1.0), "g2": (2.5, 0.5)},
    explicit_label_replace=True,
    label_col="label",
    log_transform_ycol=True,
    label_mode="all",
    color_mode="all",
)
write_demo(cfg_explicit_replace, "examples/volcano_connector_demo.explicit_replace.png")

# Explicit label placements: iterable-style (in addition to auto labels)
cfg_explicit_add = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    values_to_label=idx[:6],
    explicit_label_positions=[("g1", (-0.5, 1.0)), ("g2", (2.5, 0.5))],
    explicit_label_replace=False,
    label_col="label",
    log_transform_ycol=True,
    label_mode="auto",
    color_mode="sig_or_thresh",
)
write_demo(cfg_explicit_add, "examples/volcano_connector_demo.explicit_add.png")
