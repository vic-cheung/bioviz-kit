"""ForestPlotter example: creates a small hazard-ratio table and plots it.

Run:
    python3 src/bioviz/examples/forest_plotter_example.py
"""
import pandas as pd
import numpy as np
from bioviz.plots.forest import ForestPlotter


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "comparator": ["A vs Ref", "B vs Ref", "C vs Ref"],
            "hr": [0.8, 1.4, 1.0],
            "ci_lower": [0.6, 1.1, 0.8],
            "ci_upper": [1.0, 1.8, 1.2],
            "p_value": [0.04, 0.003, 0.9],
        }
    )
    plotter = ForestPlotter(df)
    fig, ax = plotter.plot(title="Example Forest Plot")
    print("Generated forest plot")
    fig.show()
