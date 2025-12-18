import pandas as pd
from bioviz.plots import OncoPlotter
from bioviz.configs import HeatmapAnnotationConfig, TopAnnotationConfig, OncoplotConfig

# %%
# Build a minimal dataframe: two patients, only one mutation value present
df = pd.DataFrame(
    {
        "Patient_ID": ["P1", "P2"],
        "Variant_type": ["SNV", pd.NA],
        "Dose": ["100 mg", "200 mg"],
    }
)

# Heatmap config has colors for SNV, CNV, Fusion
heatmap = HeatmapAnnotationConfig(
    values="Variant_type",
    colors={"SNV": "red", "CNV": "blue", "Fusion": "green"},
    legend_value_order=["SNV", "CNV", "Fusion"],
)

# Top annotation for Dose: has color keys as strings but dataframe may have them
top = {
    "Dose": TopAnnotationConfig(
        values=df.set_index("Patient_ID")["Dose"],
        colors={"100 mg": "#111111", "200 mg": "#222222"},
        legend_value_order=["100 mg", "200 mg"],
    )
}

cfg = OncoplotConfig(
    x_col="Patient_ID",
    y_col="Gene_Mutation",
    value_col="Variant_type",
    heatmap_annotation=heatmap,
    top_annotations=top,
    remove_unused_keys_in_legend=True,
)

# Create dummy mutation rows so plotter can initialize
df_plot = pd.DataFrame(
    {
        "Patient_ID": ["P1"],
        "Gene_Mutation": ["GENE1"],
        "Variant_type": ["SNV"],
        "Dose": ["100 mg"],
    }
)

plotter = OncoPlotter(df_plot, cfg)
# Call internal legend assembly by invoking plot() then inspect legend_categories
fig = plotter.plot()

# %%
