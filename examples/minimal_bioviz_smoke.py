"""Minimal smoke examples demonstrating bioviz line/spider/oncoplot/table.

Run inside your project venv where `bioviz` deps are installed.
"""

# %%
import matplotlib

matplotlib.use("Agg")
import pandas as pd

from bioviz.configs import (
    StyledLinePlotConfig,
    StyledSpiderPlotConfig,
    OncoplotConfig,
    HeatmapAnnotationConfig,
    StyledTableConfig,
    TopAnnotationConfig,
)
from bioviz.lineplot import generate_styled_lineplot
from bioviz.spiderplot import generate_styled_spiderplot
from bioviz.oncoplot import OncoplotPlotter
from bioviz.table import generate_styled_table

# %%
# 1) Line plot minimal data
line_df = pd.DataFrame(
    {
        "Patient_ID": ["p1", "p1"],
        "label": ["A", "A"],
        "Timepoint": pd.Categorical(["T1", "T2"], categories=["T1", "T2"], ordered=True),
        "Value": [0.5, 1.0],
        "Variant_type": ["SNV", "SNV"],
    }
)
line_cfg = StyledLinePlotConfig(patient_id="p1")
fig = generate_styled_lineplot(line_df, line_cfg)
if fig:
    fig.savefig("line_smoke.pdf")
print("line_smoke.pdf")
# %%
# 2) Spider plot minimal data
spider_df = pd.DataFrame(
    {
        "group": ["g1", "g1", "g2", "g2"],
        "Timepoint": pd.Categorical(
            ["T1", "T2", "T1", "T2"], categories=["T1", "T2"], ordered=True
        ),
        "Value": [10, 12, 5, 6],
    }
)
spider_cfg = StyledSpiderPlotConfig(group_col="group", x="Timepoint", y="Value")
fig_spider, handles, labels = generate_styled_spiderplot(spider_df, spider_cfg)
fig_spider.savefig("spider_smoke.pdf")
print("spider_smoke.pdf")

# %%
# 3) Table minimal data
table_df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
table_cfg = StyledTableConfig()
fig_table = generate_styled_table(table_df, table_cfg)
if fig_table:
    fig_table.savefig("table_smoke.pdf")
print("table_smoke.pdf")

# %%
# 4) Oncoplot minimal data with pathway (row-group) bars
mut_df = pd.DataFrame(
    {
        # include SNV, SV (bottom-left triangle) and CNV (upper-right triangle)
        "Patient_ID": ["p1", "p1", "p2", "p2", "p1", "p2"],
        "Gene": ["TP53", "KRAS", "KRAS", "TP53", "PIK3CA", "PIK3CA"],
        "Mut_aa": ["R175H", "R175H", "G12D", "G12D", "E545K", "E545K"],
        "Variant_type": ["SNV", "SNV", "CNV", "Fusion", "SNV", "SNV"],
        # minimal top-annotation column for the example
        "Cohort": ["A", "A", "B", "B", "A", "B"],
        "Dose": ["100 mg", "100 mg", "200 mg", "200 mg", "100 mg", "200 mg"],
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
        }
    }
).rename_axis("Gene")

# Pathway bar colors
row_groups_color_dict = {
    "Tumor Suppressor": "#000000",
    "RAS Signaling": "#000000",
    "PI3K/AKT": "#000000",
}

# assume mut_df has Patient_ID and some category column like 'Cohort'
cohort_series = mut_df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Cohort"]
dose_series = mut_df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Dose"]

# Map colors for all mutation/value types
colors = {"SNV": "#1f77b4", "CNV": "#ff7f0e", "Fusion": "#2ca02c"}

# TopAnnotationConfig uses tuned package defaults (height, fonts, borders)
top_ann = TopAnnotationConfig(
    values=cohort_series,
    colors={
        "A": "#003975",
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

heat_ann = HeatmapAnnotationConfig(
    values="Variant_type",  # or a pd.Series mapping patient->value
    colors=colors,
    bottom_left_triangle_values=[
        "SNV"
    ],  # these render as bottom-left triangles. make empty list to disable.
    upper_right_triangle_values=["CNV"],  # these render as top-right triangles
)

onc_cfg = OncoplotConfig(
    heatmap_annotation=heat_ann, top_annotations={"Cohort": top_ann, "Dose": dose_ann}, aspect=1
)
plotter = OncoplotPlotter(
    mut_df, onc_cfg, row_groups=row_groups, row_groups_color_dict=row_groups_color_dict
)
fig_oncoplot = plotter.plot()

ax = fig_oncoplot.axes[0]

# Shift pathway bars and labels to the left (negative values move left)
# bar_shift: how far to move the bars horizontally
# label_shift: how far to move the labels horizontally
plotter.shift_row_group_bars_and_labels(
    ax,
    row_groups,
    bar_shift=-3.5,  # move bars 3 units left
    label_shift=-3,  # move labels 3 units left (slightly more than bars)
)

# After shifting, redraw to get updated text bounding boxes
fig_oncoplot.canvas.draw()

fig_oncoplot.savefig("oncoplot.pdf", bbox_inches="tight", pad_inches=0.1, dpi=150)
print("Saved oncoplot.pdf with shifted pathway bars")

# %%
