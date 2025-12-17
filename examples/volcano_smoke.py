"""Smoke test for `plot_volcano`.

Creates a tiny synthetic dataframe and writes a PNG to verify plotting runs.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from bioviz.plot_composites.configs.volcano_cfg import VolcanoConfig
from bioviz.plot_composites.volcano import plot_volcano


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
    cfg1 = VolcanoConfig()
    cfg1.label_col = "label"
    cfg1.force_label_side_by_point_sign = True
    cfg1.log_transform_ycol = True
    fig1, ax1 = plot_volcano(cfg1, df)
    fig1.savefig(out1)
    print("Wrote", out1)

    # Variant 2: forced outward labels with rotation
    out2 = Path(__file__).with_suffix(".forced_adjusted.png")
    cfg2 = VolcanoConfig()
    cfg2.label_col = "label"
    cfg2.force_label_side_by_point_sign = True
    cfg2.force_labels_adjustable = True
    cfg2.log_transform_ycol = True
    fig2, ax2 = plot_volcano(cfg2, df)
    fig2.savefig(out2)
    print("Wrote", out2)

    # Variant 3: use adjust_text branch (no forced placement)
    out3 = Path(__file__).with_suffix(".adjust.png")
    cfg3 = VolcanoConfig()
    cfg3.label_col = "label"
    cfg3.force_label_side_by_point_sign = False
    cfg3.log_transform_ycol = True
    cfg3.use_adjust_text = True
    cfg3.adjust = True
    fig3, ax3 = plot_volcano(cfg3, df)
    fig3.savefig(out3)
    print("Wrote", out3)


if __name__ == "__main__":
    main()
