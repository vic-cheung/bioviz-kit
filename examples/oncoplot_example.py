"""
Minimal example demonstrating bioviz oncoplot plotting with shifted pathway bars.

Run inside your project venv where `bioviz` deps are installed.
"""

# %%
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from bioviz.configs import (
    OncoplotConfig,
    TopAnnotationConfig,
)
from bioviz.oncoplot import OncoplotPlotter


# %%
# Oncoplot minimal data with pathway (row-group) bars
df = pd.DataFrame(
    {
        # include SNV, SV (bottom-left triangle) and CNV (upper-right triangle)
        "Patient_ID": ["p1", "p1", "p2", "p2", "p1", "p2", "p3", "p3"],
        "Gene": ["TP53", "KRAS", "KRAS", "TP53", "PIK3CA", "PIK3CA", "PIK3CA", "FAKE_GENE"],
        "Mut_aa": ["R175H", "R175H", "G12D", "G12D", "E545K", "E545K", "R273H", "R273H"],
        "Variant_type": ["SNV", "SNV", "CNV", "Fusion", "SNV", "SNV", "Fusion", "Fusion"],
        # minimal top-annotation column for the example
        "Cohort": ["A", "A", "B", "B", "A", "B", "A", "A"],
        "Dose": ["100 mg", "100 mg", "200 mg", "200 mg", "100 mg", "200 mg", "100 mg", "100 mg"],
    }
)

# Create row_groups DataFrame: index=Gene, column='Pathway'
# This drives pathway bar drawing
row_groups = pd.DataFrame(
    {
        "Pathway": {
            "TP53": "Tumor Suppressor",
            "KRAS": "RAS Signaling",
            "PIK3CA": "PI3K/AKT",
            "FAKE_GENE": "Other",
        }
    }
).rename_axis("Gene")

# Pathway bar colors
row_groups_color_dict = {
    "Tumor Suppressor": "#000000",
    "RAS Signaling": "#000000",
    "PI3K/AKT": "#000000",
}

# assume df has Patient_ID and some category column like 'Cohort'
cohort_series = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Cohort"]
dose_series = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Dose"]

# Map colors for all mutation/value types
colors = {"SNV": "#1f77b4", "CNV": "#ff7f0e", "Fusion": "#2ca02c"}

# TopAnnotationConfig uses tuned package defaults (height, fonts, borders)
top_ann = TopAnnotationConfig(
    values=cohort_series,
    colors={
        "A": "#FFFFFF",
        "B": "#9d0ca2",
    },
    merge_labels=False,
    show_category_labels=False,
)

dose_ann = TopAnnotationConfig(
    values=dose_series,
    colors={
        "100 mg": "#007352",
        "200 mg": "#860F0F",
    },
    merge_labels=False,
    show_category_labels=False,
)

onc_cfg = OncoplotConfig(
    # Let the plotter construct the HeatmapAnnotation from the
    # oncoplot-level `row_values_color_dict` so callers don't need to
    # instantiate `HeatmapAnnotationConfig` themselves.
    heatmap_annotation=None,
    top_annotations={"Cohort": top_ann, "Dose": dose_ann},
    aspect=1,
    # Provide the colors mapping here so the plotter can use it
    # row_values_color_dict=colors,
    target_cell_width=1.5,
    target_cell_height=1.5,
    heatmap_bottom_left_triangle_values=["SNV"],
    heatmap_upper_right_triangle_values=["CNV"],
)
plotter = OncoplotPlotter(
    df, onc_cfg, row_groups=row_groups, row_groups_color_dict=row_groups_color_dict
)
fig_oncoplot = plotter.plot()

ax = fig_oncoplot.axes[0]

# Shift pathway bars and labels to the left (negative values move left)
# bar_shift: how far to move the bars horizontally
# label_shift: how far to move the labels horizontally
plotter.shift_row_group_bars_and_labels(
    ax,
    row_groups,
    bar_shift=-5.5,  # move bars 3 units left
    label_shift=-5,  # move labels 3 units left (slightly more than bars)
)

# After shifting, redraw to get updated text bounding boxes
fig_oncoplot.canvas.draw()

fig_oncoplot.savefig(
    "oncoplot.pdf",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
    facecolor="white",
)


fig_oncoplot.savefig(
    "oncoplot.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=150,
    transparent=True,
)

# %%
