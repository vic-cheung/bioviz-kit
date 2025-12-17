"""
Minimal example demonstrating bioviz oncoplot plotting with shifted pathway bars.

Run inside your project venv where `bioviz` deps are installed.
"""

# %%
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

from bioviz.configs import (
    OncoplotConfig,
    HeatmapAnnotationConfig,
    TopAnnotationConfig,
)
from bioviz.plots import OncoplotPlotter


# %%
# Oncoplot minimal data with pathway (row-group) bars
df = pd.DataFrame(
    {
        # include Popular, SV (bottom-left triangle) and Unpopular (upper-right triangle)
        "Participant": [
            "PARTICIPANT-1",
            "PARTICIPANT-1",
            "PARTICIPANT-2",
            "PARTICIPANT-2",
            "PARTICIPANT-1",
            "PARTICIPANT-2",
            "PARTICIPANT-3",
            "PARTICIPANT-1",
            "PARTICIPANT-2",
            "PARTICIPANT-3",
            "PARTICIPANT-3",
            "PARTICIPANT-3",
            "PARTICIPANT-4",
            "PARTICIPANT-4",
            "PARTICIPANT-4",
            "PARTICIPANT-5",
            "PARTICIPANT-5",
            "PARTICIPANT-5",
            "PARTICIPANT-6",
            "PARTICIPANT-6",
            "PARTICIPANT-6",
        ],
        "Food": [
            "YAM",
            "EGGPLANT",
            "EGGPLANT",
            "YAM",
            "POTATO",
            "TOMATO",
            "CARROT",
            "DAIKON",
            "DAIKON",
            "DAIKON",
            "ALMOND",
            "PECAN",
            "APPLE",
            "APPLE",
            "BANANA",
            "LETTUCE",
            "NOODLES",
            "BURGER",
            "BUN",
            "DUMPLINGS",
            "SOUP",
        ],
        "Popularity": [
            "Popular",
            "Popular",
            "Unpopular",
            "Neutral",
            "Popular",
            "Neutral",
            "Unpopular",
            "Popular",
            "Popular",
            "Neutral",
            "Neutral",
            "Neutral",
            "Popular",
            "Unpopular",
            "Neutral",
            "Unpopular",
            "Popular",
            "Neutral",
            "Unpopular",
            "Popular",
            "Neutral",
        ],
        # minimal top-annotation column for the example
        "Age": [
            "Child",
            "Child",
            "Adult",
            "Child",
            "Adult",
            "Child",
            "Adult",
            "Child",
            "Adult",
            "Child",
            "Child",
            "Child",
            "Adult",
            "Adult",
            "Adult",
            "Adult",
            "Child",
            "Child",
            "Child",
            "Adult",
            "Adult",
        ],
        "Sex": [
            "M",
            "M",
            "F",
            "F",
            "M",
            "F",
            "F",
            "M",
            "F",
            "F",
            "F",
            "F",
            "M",
            "M",
            "F",
            "M",
            "F",
            "M",
            "F",
            "M",
            "F",
        ],
    }
)

# Create row_groups DataFrame: index matches the gene field (Food)
# This drives pathway bar drawing
row_groups = pd.DataFrame(
    {
        "Bucket": {
            "YAM": "Tuber",
            "POTATO": "Tuber",
            "EGGPLANT": "Nightshade",
            "TOMATO": "Nightshade",
            "DAIKON": "Root Veg",
            "CARROT": "Root Veg",
            "ALMOND": "Nut",
            "PECAN": "Nut",
        }
    }
).rename_axis("Food")

# Pathway bar colors
row_groups_color_dict = {
    "Tuber": "#000000",
    "Nightshade": "#000000",
    "Root Veg": "#000000",
    "Nut": "#000000",
}

# assume df has Participant and some category column for top annotations
age_series = df.drop_duplicates("Participant").set_index("Participant")["Age"]
sex_series = df.drop_duplicates("Participant").set_index("Participant")["Sex"]

# Map colors for all mutation/value types
colors = {"Popular": "#1f77b4", "Unpopular": "#ff7f0e", "Neutral": "gainsboro"}

# TopAnnotationConfig uses tuned package defaults (height, fonts, borders)
age_ann = TopAnnotationConfig(
    values=age_series,
    colors={
        "Child": "#007352",
        "Adult": "#860F0F",
    },
    legend_title="Age",
    legend_value_order=["Child", "Adult"],
    merge_labels=False,
    show_category_labels=False,
)

sex_ann = TopAnnotationConfig(
    values=sex_series,
    colors={
        "M": "#0066FF",
        "F": "#FF6A00",
    },
    legend_title="Sex",
    legend_value_order=["M", "F"],
    merge_labels=False,
    show_category_labels=False,
)

heat_ann = HeatmapAnnotationConfig(
    values="Popularity",
    colors=colors,
    bottom_left_triangle_values=["Popular"],
    upper_right_triangle_values=["Unpopular"],
    legend_title="Popularity",
    legend_value_order=["Popular", "Unpopular", "Neutral"],
)

aspects = [1.0, 1.3, 0.6]
with PdfPages("oncoplot_aspects.pdf") as pdf:
    for aspect in aspects:
        onc_cfg = OncoplotConfig(
            heatmap_annotation=heat_ann,
            x_col="Participant",
            y_col="Food",
            value_col="Popularity",
            row_group_col="Bucket",
            top_annotations={"Age": age_ann, "Sex": sex_ann},
            top_annotation_order=["Age", "Sex"],
            col_split_by=["Age"],
            col_split_order={"Age": ["Child", "Adult"]},
            row_group_order=["Nightshade", "Tuber"],
            aspect=aspect,
            legend_category_order=["Age", "Sex", "Popularity"],
        )
        plotter = OncoplotPlotter(
            df,
            onc_cfg,
            row_groups=row_groups,
            row_groups_color_dict=row_groups_color_dict,
        )
        fig_oncoplot = plotter.plot()
        fig_oncoplot.suptitle(f"Aspect = {aspect}", fontsize=20)
        fig_oncoplot.canvas.draw()
        plt.show()
        pdf.savefig(fig_oncoplot, bbox_inches="tight", pad_inches=0.1)
        fig_oncoplot.clf()

# Optional: also save the last plot as PNG/PDF if desired
# fig_oncoplot.savefig("oncoplot.png", bbox_inches="tight", pad_inches=0.1, dpi=300)

# %%
