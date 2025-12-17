# %%
from pathlib import Path
from bioviz.configs.volcano_cfg import VolcanoConfig
from bioviz.plots import plot_volcano
import pandas as pd
import numpy as np

np.random.seed(2)
idx = [f"g{i}" for i in range(1, 11)]
# create a label column 'gene' that we'll pass as label_col
labels = [f"gene_{i}" for i in range(1, 11)]
df = pd.DataFrame(
    {
        "log2_or": np.random.uniform(-4, 4, size=10),
        "p_adj": np.random.uniform(0.001, 0.2, size=10),
        "gene": labels,
    }
)

# Place explicit labels using gene names and coordinates; pass label_col so
# explicit labels will match and get connectors.
explicit_positions = {
    labels[0]: (df.loc[0, "log2_or"] - 0.5, df.loc[0, "p_adj"]),
    labels[1]: (df.loc[1, "log2_or"] + 0.5, df.loc[1, "p_adj"] + 0.2),
}

cfg = VolcanoConfig(
    x_col="log2_or",
    y_col="p_adj",
    label_col="gene",
    log_transform_ycol=True,
    explicit_label_positions=explicit_positions,
    explicit_label_replace=True,
    label_mode="all",
    color_mode="all",
)
fig, ax = plot_volcano(cfg, df)

Path("examples").mkdir(parents=True, exist_ok=True)
fig.savefig("examples/explicit_with_labelcol.png")
print("Wrote examples/explicit_with_labelcol.png")
