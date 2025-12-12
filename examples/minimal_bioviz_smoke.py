"""
Minimal smoke examples demonstrating bioviz line/line+annotation/oncoplot/table.

Run inside your project venv where `bioviz` deps are installed.
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
from bioviz.configs import (
    LinePlotConfig,
    OncoplotConfig,
    HeatmapAnnotationConfig,
    StyledTableConfig,
    TopAnnotationConfig,
)
from bioviz.lineplot import (
    generate_lineplot,
    generate_lineplot_twinx,
)
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
line_cfg = LinePlotConfig(
    entity_id="p1",
    label_col="label",
    x="Timepoint",
    y="Value",
    secondary_group_col="Variant_type",
    label_points=True,  # show point labels like the prior defaults
    threshold=0.6,  # optional threshold line
    threshold_label=r"$LoD_{95}$",  # optional threshold label
    figure_transparent=True,
    ylim=(0, 1.2),
    xlim=None,
)
fig = generate_lineplot(line_df, line_cfg)
if fig:
    fig.savefig("line_smoke.pdf")
print("line_smoke.pdf")
# %%
# 2) Line overlay (multi-group trajectories)
primary_df = pd.DataFrame(
    {
        "group": ["g1", "g1", "g2", "g2"],
        "Timepoint": pd.Categorical(
            ["T1", "T2", "T1", "T2"], categories=["T1", "T2"], ordered=True
        ),
        "Value": [0, 12, 0, -6],
    }
)
primary_config = LinePlotConfig(
    group_col="group",
    x="Timepoint",
    y="Value",
    title=r"$\Delta$ Value from First Timepoint",
    ylabel=r"$\Delta$ Value from First Timepoint",
    baseline=0,
    align_first_tick_to_origin=True,
)
fig = generate_lineplot(primary_df, primary_config)
plt.show()
if fig:
    fig.savefig("line_overlay_smoke.pdf")
print("line_overlay_smoke.pdf")

# %%
# 3a) Line + twin x-axis overlay
# Use a second LinePlotConfig for the overlay, specifying overlay_* fields and shared x.
twinx_df = pd.DataFrame(
    {
        "Timepoint": pd.Categorical(
            ["T1", "T1", "T2", "T2"], categories=["T1", "T2"], ordered=True
        ),
        "DiameterChange": [5, -20, 10, -15],
        "Location": ["Lung", "Liver", "Lung", "Liver"],
        "Assessment": ["BL", "BL", "EOT", "EOT"],
    }
)
secondary_config = LinePlotConfig(
    x="Timepoint",
    y="DiameterChange",
    label_col="Location",
    overlay_col="Assessment",
    title="Twin x-axis overlay",
    align_first_tick_to_origin=False,
)
fig_combined = generate_lineplot_twinx(
    df=primary_df,
    twinx_data=twinx_df,
    primary_config=primary_config,
    secondary_config=secondary_config,
)
fig_combined.savefig("line_plus_annotation_smoke.pdf")
print("line_plus_annotation_smoke.pdf")

# %%
# 3a-raw) Line + twin x-axis overlay with raw values on secondary axis
raw_twin_df = pd.DataFrame(
    {
        "Timepoint": pd.Categorical(
            ["T1", "T1", "T2", "T2"], categories=["T1", "T2"], ordered=True
        ),
        "RawDiameterMM": [32, 28, 24, 18],
        "Location": ["Lung", "Liver", "Lung", "Liver"],
        "Assessment": ["BL", "BL", "EOT", "EOT"],
    }
)
raw_secondary_cfg = LinePlotConfig(
    x="Timepoint",
    y="RawDiameterMM",  # twin axis uses standard y/hue; overlay_* inferred automatically
    label_col="Location",
    overlay_col="Assessment",
    title="Twin overlay (raw values)",
    symmetric_ylim=False,
)
# Set independent limits per axis: primary -100..100, secondary 0..100
primary_config.ylim = (-101, 101)
raw_secondary_cfg.ylim = (-0.5, 100.5)
fig_combined_raw = generate_lineplot_twinx(
    df=primary_df,
    twinx_data=raw_twin_df,
    primary_config=primary_config,
    secondary_config=raw_secondary_cfg,
)
fig_combined_raw.savefig("line_plus_annotation_raw_smoke.pdf")
print("line_plus_annotation_raw_smoke.pdf")

# %%
# 3b) Line + twin x-axis overlay reusing a single DataFrame
# Here we keep everything in one long DataFrame and let the overlay config pull its columns.
twin_single_df = pd.DataFrame(
    {
        "group": ["g1", "g1", "g2", "g2"],
        "Timepoint": pd.Categorical(
            ["T1", "T2", "T1", "T2"], categories=["T1", "T2"], ordered=True
        ),
        "Value": [0, 12, 0, -6],
        # Mirror the overlay points used in 3a so each Location spans both timepoints.
        "DiameterChange": [5, 10, -20, -15],
        "Location": ["Lung", "Lung", "Liver", "Liver"],
        "Assessment": ["BL", "EOT", "BL", "EOT"],
    }
)
twin_single_cfg_main = LinePlotConfig(
    group_col="group",
    x="Timepoint",
    y="Value",
    title=r"$\Delta$ Value (single df)",
    ylabel=r"$\Delta$ Value",
    baseline=0,  # optional baseline line
    align_first_tick_to_origin=False,
)
twin_single_cfg_overlay = LinePlotConfig(
    x="Timepoint",
    y="DiameterChange",
    label_col="Location",
    overlay_col="Assessment",
    title="Twin overlay (single df)",
)
fig_combined_single_df = generate_lineplot_twinx(
    df=twin_single_df,
    twinx_data=None,  # reuses df for overlay layer
    primary_config=twin_single_cfg_main,
    secondary_config=twin_single_cfg_overlay,
)
fig_combined_single_df.savefig("line_plus_annotation_single_df_smoke.pdf")
print("line_plus_annotation_single_df_smoke.pdf")

# %%
# 3) Table minimal data
table_df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
table_cfg = StyledTableConfig(
    table_width=2,
    row_height=0.3,
    title="Minimal Table Example",
)
fig_table = generate_styled_table(table_df, table_cfg)
if fig_table:
    # fig_table.savefig("table_smoke.pdf", bbox_inches="tight", pad_inches=0.05)
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

heat_ann = HeatmapAnnotationConfig(
    values="Variant_type",  # or a pd.Series mapping patient->value
    colors=colors,
    bottom_left_triangle_values=[
        "SNV"
    ],  # these render as bottom-left triangles. make empty list to disable.
    upper_right_triangle_values=["CNV"],  # these render as top-right triangles
    legend_title="Mutation Type",
    legend_value_order=["SNV", "CNV", "Fusion"],
)

onc_cfg = OncoplotConfig(
    heatmap_annotation=heat_ann,
    top_annotations={"Cohort": top_ann, "Dose": dose_ann},
    legend_category_order=["Dose", "Cohort", "Mutation Type"],
)
plotter = OncoplotPlotter(
    mut_df,
    onc_cfg,
    row_groups=row_groups,
    row_groups_color_dict=row_groups_color_dict,
)
fig_oncoplot = plotter.plot()

fig_oncoplot.savefig("oncoplot.pdf", bbox_inches="tight", pad_inches=0.1, dpi=150)
print("Saved oncoplot.pdf")


# %%
