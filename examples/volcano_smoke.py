"""Smoke test for `plot_volcano`.

Creates a tiny synthetic dataframe and writes a PNG to verify plotting runs.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from bioviz.configs.volcano_cfg import VolcanoConfig
from bioviz.plots import VolcanoPlotter


def make_df():
    df = pd.DataFrame(
        {
            "label": ["A", "B", "C", "D", "E"],
            "log2_or": [3.2, -2.5, 0.5, -4.2, 1.1],
            "p_adj": [0.01, 0.04, 0.5, 0.001, 0.2],
        }
    )
    return df


def main():
    df = make_df()

    # Variant 1: forced outward labels with rotation
    out1 = Path(__file__).with_suffix(".forced.png")
    cfg1 = VolcanoConfig(x_col="log2_or", y_col="p_adj")
    cfg1.label_col = "label"
    cfg1.force_label_side_by_point_sign = True
    cfg1.log_transform_ycol = True
    cfg1.label_mode = "sig_and_thresh"
    cfg1.color_mode = "sig_and_thresh"
    vp1 = VolcanoPlotter(df, cfg1)
    fig1, ax1 = vp1.plot()
    vp1.save(out1)
    print("Wrote", out1)

    # Variant 2: forced outward labels with rotation
    out2 = Path(__file__).with_suffix(".forced_adjusted.png")
    cfg2 = VolcanoConfig(x_col="log2_or", y_col="p_adj")
    cfg2.label_col = "label"
    cfg2.force_label_side_by_point_sign = True
    cfg2.force_labels_adjustable = True
    cfg2.log_transform_ycol = True
    cfg2.label_mode = "sig_and_thresh"
    cfg2.color_mode = "sig_and_thresh"
    vp2 = VolcanoPlotter(df, cfg2)
    fig2, ax2 = vp2.plot()
    vp2.save(out2)
    print("Wrote", out2)

    # Variant 3: use adjust_text branch (no forced placement)
    out3 = Path(__file__).with_suffix(".adjust.png")
    cfg3 = VolcanoConfig(x_col="log2_or", y_col="p_adj")
    cfg3.label_col = "label"
    cfg3.force_label_side_by_point_sign = False
    cfg3.log_transform_ycol = True
    cfg3.use_adjust_text = True
    cfg3.adjust = True
    cfg3.label_mode = "auto"
    cfg3.color_mode = "sig_or_thresh"
    vp3 = VolcanoPlotter(df, cfg3)
    fig3, ax3 = vp3.plot()
    vp3.save(out3)
    print("Wrote", out3)


if __name__ == "__main__":
    main()
