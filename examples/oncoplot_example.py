"""
Minimal example demonstrating bioviz oncoplot plotting with shifted pathway bars.

Run inside your project venv where `bioviz` deps are installed.
"""

# %%
# import matplotlib

# matplotlib.use("Agg")
import pandas as pd

from bioviz.configs import (
    OncoplotConfig,
    HeatmapAnnotationConfig,
    TopAnnotationConfig,
)
from bioviz.oncoplot import OncoplotPlotter


# %%
# Oncoplot minimal data with pathway (row-group) bars
df = pd.DataFrame(
    {
        # include SNV, SV (bottom-left triangle) and CNV (upper-right triangle)
        "Patient_ID": [
            "p1",
            "p1",
            "p2",
            "p2",
            "p1",
            "p2",
            "p3",
            "p1",
            "p2",
            "p3",
            "p3",
            "p3",
        ],
        "Gene": [
            "TP53",
            "KRAS",
            "KRAS",
            "TP53",
            "POTATO",
            "TOMATO",
            "CARROT",
            "PIK3CA",
            "PIK3CA",
            "PIK3CA",
            "FAKE_GENE",
            "FAKE_GENE_2",
        ],
        "Variant_type": [
            "SNV",
            "SNV",
            "CNV",
            "Fusion",
            "SNV",
            "Fusion",
            "CNV",
            "SNV",
            "SNV",
            "Fusion",
            "Fusion",
            "Fusion",
        ],
        # minimal top-annotation column for the example
        "Cohort": [
            "A",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "A",
            "A",
        ],
        "Dose": [
            "100 mg",
            "100 mg",
            "200 mg",
            "100 mg",
            "200 mg",
            "100 mg",
            "200 mg",
            "100 mg",
            "200 mg",
            "100 mg",
            "100 mg",
            "100 mg",
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
        ],
    }
)

# Create row_groups DataFrame: index=Gene, column='Pathway'
# This drives pathway bar drawing
row_groups = pd.DataFrame(
    {
        "Pathway": {
            "TP53": "Tumor Suppressor",
            "POTATO": "Tumor Suppressor",
            "KRAS": "RAS Signaling",
            "TOMATO": "RAS Signaling",
            "PIK3CA": "PI3K/AKT",
            "CARROT": "PI3K/AKT",
            "FAKE_GENE": "Other",
            "FAKE_GENE_2": "Other",
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
sex_series = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Sex"]

# Map colors for all mutation/value types
colors = {"SNV": "#1f77b4", "CNV": "#ff7f0e", "Fusion": "#2ca02c"}

# TopAnnotationConfig uses tuned package defaults (height, fonts, borders)
cohort_ann = TopAnnotationConfig(
    values=cohort_series,
    colors={
        "A": "#FFFFFF",
        "B": "#9d0ca2",
    },
    legend_title="Cohort",
    legend_value_order=["A", "B"],
    merge_labels=False,
    show_category_labels=False,
)

dose_ann = TopAnnotationConfig(
    values=dose_series,
    colors={
        "100 mg": "#007352",
        "200 mg": "#860F0F",
    },
    legend_title="Dose",
    legend_value_order=["100 mg", "200 mg"],
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
    values="Variant_type",
    colors=colors,
    bottom_left_triangle_values=["SNV"],
    upper_right_triangle_values=["CNV"],
    legend_title="Mutation Type",
    legend_value_order=["SNV", "CNV", "Fusion"],
)

onc_cfg = OncoplotConfig(
    heatmap_annotation=heat_ann,
    top_annotations={"Cohort": cohort_ann, "Dose": dose_ann, "Sex": sex_ann},
    top_annotation_order=["Sex", "Dose", "Cohort"],
    col_split_by=["Sex"],
    col_split_order={"Sex": ["M", "F"]},
    aspect=1,
    legend_category_order=["Sex", "Dose", "Cohort", "Mutation Type"],
    row_group_post_bar_shift=-6,
    row_group_post_label_shift=-5.5,
)
plotter = OncoplotPlotter(
    df, onc_cfg, row_groups=row_groups, row_groups_color_dict=row_groups_color_dict
)
fig_oncoplot = plotter.plot()

# # Shift pathway bars and labels to the left (negative values move left)
# # bar_shift: how far to move the bars horizontally
# # label_shift: how far to move the labels horizontally
# ax = fig_oncoplot.axes[0]
# # Protect layout when applying manual shifts: freeze limits so autoscale
# # does not expand the axes and distort spacing before saving.
# orig_xlim = ax.get_xlim()
# orig_ylim = ax.get_ylim()
# ax.set_autoscale_on(False)
# plotter.shift_row_group_bars_and_labels(
#     ax,
#     row_groups,
#     bar_shift=-5.5,  # move bars 3 units left
#     label_shift=-5,  # move labels 3 units left (slightly more than bars)
# )
# # Restore limits after shifting
# ax.set_xlim(orig_xlim)
# ax.set_ylim(orig_ylim)

# # After shifting, redraw to get updated text bounding boxes
fig_oncoplot.canvas.draw()

fig_oncoplot.savefig(
    "oncoplot.pdf",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=300,
    facecolor="white",
)


fig_oncoplot.savefig(
    "oncoplot.png",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=300,
    transparent=True,
)

# %%
