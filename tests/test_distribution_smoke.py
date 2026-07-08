import matplotlib.pyplot as plt
import pandas as pd

from bioviz.plots.distribution import generate_histogram


def test_histogram_accepts_explicit_hue_palette():
    df = pd.DataFrame(
        {
            "value": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    fig, ax = plt.subplots()

    try:
        generate_histogram(
            ax,
            df,
            variable_name="value",
            indication="demo",
            median_value=2.5,
            hue="group",
            value_col="value",
            hue_palette={"A": "#EC745C", "B": "#44A9CC"},
        )

        assert ax.patches
    finally:
        plt.close(fig)
