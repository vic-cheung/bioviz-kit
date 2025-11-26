"""Helper functions for processing and annotating plotting data.

This module contains utility functions to assist with:
- Mapping indication values to plot-specific categories.
- Generating informative labels for genetic alterations.

These functions support consistent data annotation for downstream plotting
routines.
"""

# %%
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes

try:
    from tm_toolbox.data_preprocessing import get_filtered_entity_data
except Exception:  # pragma: no cover - optional dependency

    def get_filtered_entity_data(*args, **kwargs):
        raise ImportError(
            "get_filtered_entity_data requires tm_toolbox.data_preprocessing; "
            "install tm-toolbox or provide this function"
        )


# Expose all public functions
__all__ = [
    "set_plot_indication",
    "label_row",
    "forward_fill_groups",
    "adjust_legend",
    "rename_recist_and_melt",
    "merge_biodesix_to_edc_and_melt",
    "apply_recist_merge",
    "infer_fusion_pathway",
    "assign_gene_mutation",
    "get_oncoplot_fig_top_margin",
    "get_scaled_oncoplot_dimensions",
    "get_oncoplot_dimensions_fixed_cell",
]


# %%
def set_plot_indication(
    df: pd.DataFrame,
    indication_map: dict,
    col_name: str = "Indication",
    default: str = "OTHER",
) -> pd.DataFrame:
    df = df.copy()
    df["Plot_Indication"] = df[col_name].map(indication_map).fillna(default)
    return df


# %%
def label_row(
    row: pd.Series,
    gene_col: str = "Gene",
    mut_aa_col: str = "Mut_aa",
    cna_col: str = "Copy_Number_Alteration",
    variant_type_col: str = "Variant_type",
) -> str:
    gene = row.get(gene_col, "")
    mut_aa = row.get(mut_aa_col, "")
    cna = row.get(cna_col, "")
    variant_type = row.get(variant_type_col, "")

    if cna == "Amplification":
        return f"{gene}_AMP"
    elif cna == "Deletion":
        return f"{gene}_DEL"
    elif variant_type == "Fusion":
        return gene
    elif pd.isna(mut_aa) or mut_aa == "":
        return f"{gene}_SPLICE"
    else:
        return f"{gene}_{mut_aa}"


# %%
def forward_fill_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    x: str,
    y: str,
    sort_order: list | None = None,
) -> pd.DataFrame:
    df = df.copy()

    dedupe_cols = group_cols + [x]
    if df.duplicated(subset=dedupe_cols).any():
        num_dupes = df.duplicated(subset=dedupe_cols).sum()
        logger.warning(f"Found {num_dupes} duplicate rows in data, removing duplicates")
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")

    if sort_order is not None:
        df[x] = pd.Categorical(df[x], categories=sort_order, ordered=True)
    elif not pd.api.types.is_categorical_dtype(df[x]):
        df[x] = pd.Categorical(df[x], categories=sorted(df[x].unique()), ordered=True)

    df = df.set_index(group_cols + [x])

    index_levels = [df.index.get_level_values(col).unique() for col in group_cols]
    time_values = df.index.get_level_values(x).unique()

    full_index = pd.MultiIndex.from_product(index_levels + [time_values], names=group_cols + [x])

    if not full_index.is_unique:
        full_index = full_index.drop_duplicates()

    df = df.reindex(full_index)

    df[y] = df.groupby(level=group_cols)[y].ffill()

    df = df.reset_index()

    return df


# %%
def adjust_legend(
    ax: Axes, bbox: tuple[float, float], loc: str = "center left", redraw: bool = False
) -> None:
    leg = ax.get_legend()
    if leg:
        leg.set_bbox_to_anchor(bbox)
        leg.set_loc(loc)
        if redraw:
            ax.figure.canvas.draw_idle()


# %%
def rename_recist_and_melt(recist_df: pd.DataFrame) -> pd.DataFrame:
    recist_df.rename(columns={"rm_patient_full": "Patient_ID", "Screening": "Pre"}, inplace=True)
    return (
        recist_df.melt(id_vars=["Patient_ID"], var_name="Timepoint", value_name="RECIST")
        .replace(
            {
                "RECIST": {
                    "Partial Response": "PR",
                    "Stable Disease": "SD",
                    "Progressive Disease": "PD",
                }
            }
        )
        .reset_index(drop=True)
    )


# %%
def merge_biodesix_to_edc_and_melt(biodesix_df: pd.DataFrame, edc_df: pd.DataFrame) -> pd.DataFrame:
    if "Patient_ID" in biodesix_df.columns and "Patient_ID" in edc_df.columns:
        merge_col = "Patient_ID"
        rename_patient_id = False
    elif "rm_patient_full" in biodesix_df.columns and "rm_patient_full" in edc_df.columns:
        merge_col = "rm_patient_full"
        rename_patient_id = True
    else:
        raise ValueError("No matching patient ID column found between biodesix_df and edc_df")

    df = biodesix_df.merge(edc_df, how="left", on=merge_col)

    rename_dict = {
        "Enrollment Mutation - KRAS G12 Mutations": "enrollment_mutation",
        "Actual Majority Dose": "Dose",
    }

    if rename_patient_id:
        rename_dict["rm_patient_full"] = "Patient_ID"

    df.rename(columns=rename_dict, inplace=True)

    coi = [
        "Patient_ID",
        "enrollment_mutation",
        "Target Lesions - Best Percent Change from Baseline",
        "Dose",
        "Cancer Groups",
        "Pre_MVF",
        "C2D1_MVF",
        "C3D1_MVF",
        "C5D1_MVF",
    ]
    df = df.loc[:, coi]
    df_melt = df.melt(id_vars=coi[:5], var_name="Timepoint", value_name="ddPCR Value")
    df_melt["Timepoint"] = df_melt["Timepoint"].str.split("_MVF", expand=True)[0]
    return df_melt


# %%
def apply_recist_merge(
    df: pd.DataFrame,
    recist_df_melt: pd.DataFrame,
    recist_order: list[str],
    sort_order: list[str],
) -> pd.DataFrame:
    df = df.merge(recist_df_melt, how="left", on=["Patient_ID", "Timepoint"])
    df["RECIST"] = pd.Categorical(df["RECIST"], categories=recist_order, ordered=True)
    df["Timepoint"] = pd.Categorical(df["Timepoint"], categories=sort_order, ordered=True)
    return df


# %%
def infer_fusion_pathway(
    eot_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    resistance_pathway_df: pd.DataFrame,
    gene_col: str = "Gene",
    variant_type_col: str = "Variant_type",
    oncogenic_col: str = "ONCOGENIC",
    pathway_gene_col: str = "Gene",
    pathway_pathway_col: str = "Pathway",
    filter_dict: dict = None,
) -> pd.DataFrame:
    if filter_dict is None:
        filter_dict = {
            variant_type_col: ["Fusion"],
            oncogenic_col: ["Oncogenic", "Likely Oncogenic"],
        }
    fusion_df = get_filtered_entity_data(df=eot_df, filter_dict=filter_dict)

    gene_split = fusion_df[gene_col].str.split("-", n=1, expand=True).fillna("")
    gene_split.columns = ["GeneA", "GeneB"]
    fusion_df = fusion_df.assign(GeneA=gene_split["GeneA"], GeneB=gene_split["GeneB"])

    fusion_df = fusion_df[["GeneA", "GeneB"]].drop_duplicates().reset_index(drop=True)
    fusion_df["id"] = fusion_df.index + 1

    melted = fusion_df.melt(
        id_vars="id",
        value_vars=["GeneA", "GeneB"],
        var_name="Partner",
        value_name="Gene",
    )

    annotated = (
        melted.merge(
            pathway_df.rename(columns={pathway_gene_col: "Gene", pathway_pathway_col: "Pathway"}),
            on="Gene",
            how="left",
        )
        .pivot(index="id", columns="Partner", values="Pathway")
        .reset_index()
    )

    def assign_pathway(row):
        if pd.notna(row["GeneA"]) and pd.notna(row["GeneB"]) and row["GeneA"] and row["GeneB"]:
            return ""
        elif pd.notna(row["GeneA"]) and row["GeneA"]:
            return row["GeneA"]
        elif pd.notna(row["GeneB"]) and row["GeneB"]:
            return row["GeneB"]
        else:
            return pd.NA

    annotated["Pathway"] = annotated.apply(assign_pathway, axis=1)

    fusion_df = fusion_df.merge(annotated[["id", "Pathway"]], on="id", how="left")
    fusion_df["Gene"] = fusion_df.apply(
        lambda row: f"{row['GeneA']}-{row['GeneB']}" if row["GeneB"] else row["GeneA"],
        axis=1,
    )

    fusion_df = fusion_df.dropna(subset=["Pathway"])

    novel_fusions = fusion_df[["Gene", "Pathway"]]
    updated_pathways = pd.concat([resistance_pathway_df, novel_fusions], ignore_index=True)

    return updated_pathways


def assign_gene_mutation(
    row,
    type_col: str = "Type",
    gene_col: str = "Gene",
    mut_aa_col: str = "Mut_aa",
    snv_value: str = "SNV",
    splice_value: str = "SPLICE",
    splice_label: str = "TRUNC",
):
    if row[type_col] == snv_value:
        return f"{row[gene_col]} {row[mut_aa_col]}"
    if row[type_col] == splice_value:
        return f"{row[gene_col]} {splice_label}"
    return row[gene_col]


def get_oncoplot_fig_top_margin(
    fig_height: float | int,
    min_height: float | int = 5.0,
    mid_height: float | int = 33.33,
    max_height: float | int = 131.56,
    min_margin: float | int = 0.78,
    mid_margin: float | int = 0.82,
    max_margin: float | int = 0.85,
) -> float | int:
    x = np.array([min_height, mid_height, max_height])
    y = np.array([min_margin, mid_margin, max_margin])
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    margin = a * fig_height**2 + b * fig_height + c
    margin = max(min_margin, min(max_margin, margin))
    if fig_height < min_height:
        margin = min_margin + (0.90 - min_margin) * (1 - fig_height / min_height)
    if 18.0 <= fig_height <= 22.0:
        margin = 0.85
    elif fig_height > 100:
        margin = 0.82
    elif fig_height > 60:
        margin = 0.84
    return margin


def get_scaled_oncoplot_dimensions(
    ncols: int,
    nrows: int,
    num_top_annotations: int,
    top_annotation_height: float = 1.0,
    col_scale: float = 16 / 9,
    row_scale: float = 10 / 30,
    aspect: float = 1.0,
    max_width: None | float = None,
    max_height: None | float = None,
) -> tuple[float, float]:
    base_width = ncols * col_scale
    base_height = nrows * row_scale + num_top_annotations * top_annotation_height
    min_figure_height = 6.0
    if nrows <= 10:
        if nrows <= 3:
            adjusted_cell_size = min(0.8, 7.0 / max(nrows, 1))
        else:
            adjusted_cell_size = min(0.6, 6.0 / max(nrows, 1))
        desired_height = max(nrows * adjusted_cell_size, min_figure_height)
        base_height = max(base_height, desired_height)
    if ncols <= 10:
        adjusted_cell_width = min(0.8, 8.0 / max(ncols, 1))
        desired_width = ncols * adjusted_cell_width
        base_width = min(base_width, desired_width)
    if nrows == 1:
        base_height = max(min_figure_height, base_height)
    if ncols == 1:
        min_width = 2.0
        base_width = max(min_width, base_width)
    if aspect < 1:
        fig_width = base_width / aspect
        fig_height = base_height
    elif aspect > 1:
        fig_width = base_width
        fig_height = base_height * aspect
    else:
        fig_width = base_width
        fig_height = base_height
    if max_width is not None:
        fig_width = min(fig_width, max_width)
    if max_height is not None:
        fig_height = min(fig_height, max_height)
    return (fig_width, fig_height)


def get_oncoplot_dimensions_fixed_cell(
    ncols: int,
    nrows: int,
    target_cell_width: float = 0.8,
    target_cell_height: float = 0.8,
    min_width: float = 8.0,
    max_width: float = 200.0,
    max_height: float = 60.0,
    num_top_annotations: int = 0,
    top_annotation_height: float = 0.25,
    aspect: float = 1.0,
    auto_adjust_cell_size: bool = True,
) -> tuple[float, float]:
    adjusted_cell_width = target_cell_width
    adjusted_cell_height = target_cell_height
    if auto_adjust_cell_size:
        patient_threshold = int(0.6 * max_width / target_cell_width)
        gene_threshold = int(0.6 * max_height / target_cell_height)
        min_cell_ratio = 1 / 3
        min_cell_width = target_cell_width * min_cell_ratio
        min_cell_height = target_cell_height * min_cell_ratio
        if ncols > patient_threshold:
            scale_factor = min(1.0, 0.9 * (patient_threshold / ncols) ** 0.4)
            adjusted_cell_width = max(min_cell_width, target_cell_width * scale_factor)
        if nrows > gene_threshold:
            scale_factor = min(1.0, 0.95 * (gene_threshold / nrows) ** 0.3)
            adjusted_cell_height = max(min_cell_height, target_cell_height * scale_factor)
    width = ncols * adjusted_cell_width
    height = nrows * adjusted_cell_height
    if num_top_annotations > 0:
        height += num_top_annotations * top_annotation_height
    padding_width = 1.0
    padding_height = 1.0
    width += padding_width
    height += padding_height
    if height > max_height:
        scale_factor = max_height / height
        height = max_height
        width = width * scale_factor
    if aspect != 1.0:
        target_width = height / aspect
        if target_width > width:
            width = target_width
        else:
            height = width * aspect
    width = min(max(width, min_width), max_width)
    height = min(height, max_height)
    width_scale_factor = 1.0
    height_scale_factor = 1.0
    if auto_adjust_cell_size:
        if ncols > patient_threshold:
            width_scale_factor = min(1.0, 0.9 * (patient_threshold / ncols) ** 0.4)
        if nrows > gene_threshold:
            height_scale_factor = min(1.0, 0.95 * (gene_threshold / nrows) ** 0.3)
    font_scale_factor = min(width_scale_factor, height_scale_factor)
    if font_scale_factor < 1.0:
        font_scale_factor = min(1.0, font_scale_factor**0.6)
    return (width, height, font_scale_factor)
