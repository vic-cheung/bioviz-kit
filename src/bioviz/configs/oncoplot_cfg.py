from typing import Annotated, Any

from pydantic import BaseModel, Field

try:
    # pydantic v1 compatibility
    from pydantic import root_validator  # type: ignore
except Exception:
    root_validator = None

from .oncoplot_annotations_cfg import (
    TopAnnotationConfig,  # noqa: F401
    HeatmapAnnotationConfig,
)


class OncoplotConfig(BaseModel):
    """
    Config for oncoplot rendering.

    The plotter treats `x_col`, `y_col`, `value_col`, and `row_group_col` as
    required logical columns even though defaults exist. Callers should set
    these explicitly to match their DataFrame so patient IDs, feature rows,
    pathway/group labels, and heatmap values are drawn correctly.
    """

    fig_title: Annotated[str | None, Field(default=None)]
    fig_title_fontsize: Annotated[float | int, Field(default=22)]
    fig_top_margin: Annotated[float, Field(default=0.9)]
    fig_bottom_margin: Annotated[float, Field(default=0.18)]
    col_split_by: Annotated[list[str], Field(default_factory=list)]
    col_split_order: Annotated[dict[str, list[str]], Field(default_factory=dict)]
    col_sort_by: Annotated[list[str], Field(default_factory=list)]
    figsize: Annotated[tuple[float, float], Field(default=(12, 8))]
    bar_width: Annotated[float, Field(default=0.1)]
    bar_width_points: Annotated[float, Field(default=5.0)]
    bar_width_use_points: Annotated[bool, Field(default=True)]
    # Optional post-draw row-group shifts to mimic manual helper calls
    apply_post_row_group_shift: Annotated[bool, Field(default=True)]
    row_group_post_bar_shift: Annotated[float, Field(default=-5.5)]
    row_group_post_label_shift: Annotated[float, Field(default=-5.0)]
    row_group_post_bar_shift_points: Annotated[float, Field(default=-240.0)]
    row_group_post_label_shift_points: Annotated[float, Field(default=-220.0)]
    row_group_post_shift_use_points: Annotated[bool, Field(default=True)]
    row_group_label_fontsize: Annotated[float | int, Field(default=16)]
    row_label_fontsize: Annotated[float | int, Field(default=16)]
    column_label_fontsize: Annotated[float | int, Field(default=16)]
    rowlabel_xoffset: Annotated[float, Field(default=-0.05)]
    rowlabel_use_points: Annotated[bool, Field(default=True)]
    legend_fontsize: Annotated[float | int, Field(default=16)]
    legend_title_fontsize: Annotated[float | int, Field(default=16)]
    rotate_left_annotation_label: Annotated[bool, Field(default=False)]
    x_col: Annotated[str, Field(default_factory=str)]
    y_col: Annotated[str, Field(default_factory=str)]
    row_group_col: Annotated[str, Field(default_factory=str)]
    row_group_order: Annotated[list[str] | None, Field(default=None)]
    heatmap_annotation: Annotated[HeatmapAnnotationConfig | None, Field(default=None)]
    value_col: Annotated[str, Field(default_factory=str)]
    top_annotation_order: Annotated[list[str] | None, Field(default=None)]
    # Additional layout and annotation defaults to make OncoPlotter usable
    # tuned defaults for bioviz: tighter top-annotation spacing and margins
    cell_aspect: Annotated[float, Field(default=1.0)]
    # Target rendered cell size (inches) used when auto-computing `figsize`.
    # Lowering these reduces the physical size of each cell on the figure.
    target_cell_width: Annotated[float, Field(default=0.55)]
    target_cell_height: Annotated[float, Field(default=0.55)]
    # Whether the automatic cell-size adjustment logic in the sizing helper runs
    auto_adjust_cell_size: Annotated[bool, Field(default=True)]
    top_annotations: Annotated[dict[str, Any] | None, Field(default_factory=dict)]
    top_annotation_inter_spacer: Annotated[float, Field(default=1.3)]
    top_annotation_intra_spacer: Annotated[float, Field(default=0.17)]
    top_annotation_label_offset: Annotated[float, Field(default=0.3)]
    top_annotation_label_offset_points: Annotated[float, Field(default=12.0)]
    top_annotation_label_use_points: Annotated[bool, Field(default=True)]
    col_split_gap: Annotated[float, Field(default=0.25)]
    row_split_gap: Annotated[float, Field(default=0.25)]
    bar_offset: Annotated[float, Field(default=2.5)]
    bar_buffer: Annotated[float, Field(default=0)]
    bar_offset_use_points: Annotated[bool, Field(default=True)]
    row_group_label_gap_use_points: Annotated[bool, Field(default=True)]
    legend_category_order: Annotated[list[str] | None, Field(default=None)]
    xticklabel_xoffset: Annotated[float, Field(default=0.0)]
    xticklabel_yoffset: Annotated[float, Field(default=0.0)]
    legend_bbox_to_anchor: Annotated[tuple[float, float] | None, Field(default=None)]
    legend_offset: Annotated[float, Field(default=0.1)]
    legend_offset_points: Annotated[float, Field(default=24.0)]
    legend_offset_use_points: Annotated[bool, Field(default=True)]
    fig_y_margin: Annotated[float, Field(default=0.02)]
    aspect: Annotated[float, Field(default=1.0)]
    # How strongly aspect rescales horizontal spacing (0 = ignore aspect, 1 = full)
    spacing_aspect_scale: Annotated[float, Field(default=0.0)]
    # How strongly aspect rescales x-tick vertical offset (0 = ignore aspect, 1 = full)
    xtick_aspect_scale: Annotated[float, Field(default=0.0)]
    # Whether to interpret xtick offsets in points (axes transform) instead of data units
    xticklabel_use_points: Annotated[bool, Field(default=True)]
    # Gap between the row-group bar and its label
    row_group_label_gap: Annotated[float, Field(default=30.0)]
    value_legend_title: Annotated[str | None, Field(default=None)]
    remove_unused_keys_in_legend: Annotated[bool, Field(default=True)]

    # Figure-level display controls. Keep axes facecolor opaque; these
    # control the `Figure` patch used for export/transparent backgrounds.
    figure_facecolor: Annotated[str | None, Field(default=None)]
    figure_transparent: Annotated[bool, Field(default=False)]

    # Provide a default color mapping so a basic heatmap legend can be generated
    # Optional color mapping for heatmap values. Default None for generic use.
    row_values_color_dict: Annotated[dict[str, str] | None, Field(default=None)]
    # Defaults for rendering triangles in heatmap cells. These are passed
    # through to the constructed `HeatmapAnnotationConfig` when the
    # caller doesn't provide one explicitly.
    heatmap_bottom_left_triangle_values: Annotated[list[str], Field(default_factory=list)]
    heatmap_upper_right_triangle_values: Annotated[list[str], Field(default_factory=list)]

    # Compatibility mapping: allow callers to specify `bar_width` and treat
    # it as an alias for `bar_width_points` when they intend point-based widths.
    # Implement both a pre-root validator (pydantic v1) and a v2 post-init hook.
    if root_validator is not None:

        @root_validator(pre=True)
        def _map_bar_width_to_points(cls, values):
            if "bar_width" in values and "bar_width_points" not in values:
                # Treat user-supplied bar_width as points if they provided it.
                values["bar_width_points"] = values["bar_width"]
            return values

    def model_post_init(self, __context: dict | None = None) -> None:  # pydantic v2
        # When using pydantic v2 the pre-validator above may not be present.
        # If the user explicitly set `bar_width` but not `bar_width_points`,
        # copy the value across so `bar_width` serves as an alias.
        try:
            fields_set = getattr(self, "__pydantic_fields_set__", set())
        except Exception:
            fields_set = set()
        if "bar_width" in fields_set and "bar_width_points" not in fields_set:
            try:
                object.__setattr__(self, "bar_width_points", float(getattr(self, "bar_width", 0)))
            except Exception:
                pass
