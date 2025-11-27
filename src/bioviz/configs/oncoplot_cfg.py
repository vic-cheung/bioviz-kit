from typing import Annotated, Any

from pydantic import BaseModel, Field
import pandas as pd

from .annotations_cfg import TopAnnotationConfig, HeatmapAnnotationConfig


class OncoplotConfig(BaseModel):
    fig_title: Annotated[str | None, Field(default=None)]
    fig_title_fontsize: Annotated[float | int, Field(default=22)]
    fig_top_margin: Annotated[float, Field(default=0.9)]
    fig_bottom_margin: Annotated[float, Field(default=0.18)]
    col_split_by: Annotated[list[str], Field(default_factory=list)]
    col_split_order: Annotated[dict[str, list[str]], Field(default_factory=dict)]
    col_sort_by: Annotated[list[str], Field(default_factory=list)]
    figsize: Annotated[tuple[float, float], Field(default=(12, 8))]
    bar_width: Annotated[float, Field(default=0.1)]
    row_group_label_fontsize: Annotated[float | int, Field(default=16)]
    row_label_fontsize: Annotated[float | int, Field(default=16)]
    column_label_fontsize: Annotated[float | int, Field(default=16)]
    legend_fontsize: Annotated[float | int, Field(default=16)]
    legend_title_fontsize: Annotated[float | int, Field(default=16)]
    rotate_left_annotation_label: Annotated[bool, Field(default=False)]
    x_col: Annotated[str, Field(default="Patient_ID")]
    y_col: Annotated[str, Field(default="Gene")]
    row_group_col: Annotated[str, Field(default="Pathway")]
    heatmap_annotation: Annotated[HeatmapAnnotationConfig | None, Field(default=None)]
    value_col: Annotated[str, Field(default="Variant_type")]
    top_annotation_order: Annotated[list[str] | None, Field(default=None)]
    # Additional layout and annotation defaults to make OncoplotPlotter usable
    # tuned defaults for bioviz: tighter top-annotation spacing and margins
    cell_aspect: Annotated[float, Field(default=1.0)]
    # Target rendered cell size (inches) used when auto-computing `figsize`.
    # Lowering these reduces the physical size of each cell on the figure.
    target_cell_width: Annotated[float, Field(default=1.5)]
    target_cell_height: Annotated[float, Field(default=1.5)]
    # Whether the automatic cell-size adjustment logic in the sizing helper runs
    auto_adjust_cell_size: Annotated[bool, Field(default=True)]
    top_annotations: Annotated[dict[str, Any] | None, Field(default_factory=dict)]
    top_annotation_inter_spacer: Annotated[float, Field(default=0.7)]
    top_annotation_intra_spacer: Annotated[float, Field(default=0.15)]
    col_split_gap: Annotated[float, Field(default=0.25)]
    row_split_gap: Annotated[float, Field(default=0.25)]
    bar_offset: Annotated[float, Field(default=2.5)]
    bar_buffer: Annotated[float, Field(default=0.5)]
    legend_category_order: Annotated[list[str] | None, Field(default=None)]
    xticklabel_xoffset: Annotated[float, Field(default=0.0)]
    xticklabel_yoffset: Annotated[float, Field(default=0.6)]
    legend_bbox_to_anchor: Annotated[tuple[float, float] | None, Field(default=None)]
    legend_offset: Annotated[float, Field(default=0.1)]
    fig_y_margin: Annotated[float, Field(default=0.02)]
    aspect: Annotated[float, Field(default=1.0)]
    value_legend_title: Annotated[str | None, Field(default=None)]
    remove_unused_keys_in_legend: Annotated[bool, Field(default=True)]

    # Figure-level display controls. Keep axes facecolor opaque; these
    # control the `Figure` patch used for export/transparent backgrounds.
    figure_facecolor: Annotated[str | None, Field(default=None)]
    figure_transparent: Annotated[bool, Field(default=False)]

    # Provide a default color mapping so a basic heatmap legend can be generated
    row_values_color_dict: Annotated[
        dict[str, str] | None,
        Field(default_factory=lambda: {"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"}),
    ]
    # Defaults for rendering triangles in heatmap cells. These are passed
    # through to the constructed `HeatmapAnnotationConfig` when the
    # caller doesn't provide one explicitly.
    heatmap_bottom_left_triangle_values: Annotated[
        list[str], Field(default_factory=lambda: ["SNV"])
    ]
    heatmap_upper_right_triangle_values: Annotated[
        list[str], Field(default_factory=lambda: ["CNV"])
    ]
