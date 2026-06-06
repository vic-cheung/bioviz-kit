# %%
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import numpy as np
import pandas as pd

from bioviz.configs import (
    HeatmapAnnotationConfig,
    OncoplotConfig,
    RightSummaryBarsConfig,
    TopAnnotationConfig,
)
from bioviz.plots import OncoGeneBarPlotter, OncoPlotter, OncoPrevalencePlotter


# %%
def test_oncoplot_shapes_centered():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "Fusion"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
    )
    config = OncoplotConfig(heatmap_annotation=heat, x_col="patient_id", y_col="gene")
    plotter = OncoPlotter(pdf, config=config)
    fig = plotter.plot()
    ax = fig.axes[0]

    # Expect shapes to be drawn as patches; compute their centers and assert they are near half-integer centers
    # (cells are typically 1 unit wide centered at 0.5, 1.5, etc.)
    centers = []
    for p in ax.patches:
        try:
            bbox = p.get_bbox()
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            centers.append((cx, cy))
        except Exception:
            # polygons or other patches may not expose get_bbox; approximate via path
            path = getattr(p, "get_path", None)
            if path is not None:
                verts = p.get_path().vertices
                cx = verts[:, 0].mean()
                cy = verts[:, 1].mean()
                centers.append((cx, cy))

    assert centers, "No patch centers found"

    # Heatmap cells are centered at half-integer positions (0.5, 1.5, etc.)
    tol = 0.3
    for cx, cy in centers:
        # Check centers are near half-integers (e.g., 0.5, 1.5)
        x_offset = cx - int(cx)
        y_offset = cy - int(cy)
        assert (abs(x_offset - 0.5) <= tol or abs(x_offset) <= tol) and (
            abs(y_offset - 0.5) <= tol or abs(y_offset) <= tol
        ), f"Center ({cx}, {cy}) not near expected grid position"


def test_oncoplot_cell_alignment(tmp_path):
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "SV"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type", colors={"SNV": "#ff0000", "CNV": "#00ff00", "SV": "#0000ff"}
    )
    config = OncoplotConfig(heatmap_annotation=heat, x_col="patient_id", y_col="gene")
    plotter = OncoPlotter(pdf, config=config)
    fig = plotter.plot()
    ax = fig.axes[0]

    # Collect patch centers
    centers = []
    for p in ax.patches:
        try:
            x0, y0 = p.get_xy()
            w = getattr(p, "get_width", lambda: None)()
            h = getattr(p, "get_height", lambda: None)()
            if w is None or h is None:
                # polygons; compute centroid from path
                path = p.get_path().vertices
                cx = path[:, 0].mean()
                cy = path[:, 1].mean()
            else:
                cx = x0 + w / 2
                cy = y0 + h / 2
            centers.append((cx, cy))
        except Exception:
            continue

    # Oncoplot cells are centered at half-integer positions (0.5, 1.5, etc.)
    # Expected centers are at (0.5, 0.5), (0.5, 1.5), (1.5, 0.5) for P1-TP53, P1-KRAS, P2-TP53
    expected = {(0.5, 0.5), (0.5, 1.5), (1.5, 0.5)}

    rounded = {(round(cx, 1), round(cy, 1)) for cx, cy in centers}
    assert expected.issubset(rounded), f"Expected centers {expected} in {rounded}"


def test_oncoplot_transparent_figure_patch():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "SNV"},
    ])

    heat = HeatmapAnnotationConfig(values="mut_type", colors={"SNV": "#EC745C"})
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        figure_facecolor="#123456",
        figure_transparent=True,
    )
    fig = OncoPlotter(pdf, config=cfg).plot()

    assert fig.patch.get_alpha() == 0.0
    face = fig.patch.get_facecolor()
    # Face color should retain the provided RGB even when fully transparent
    expected_rgb = mcolors.to_rgba("#123456")[:3]
    assert tuple(round(v, 3) for v in face[:3]) == tuple(round(v, 3) for v in expected_rgb)


def test_oncoplot_forces_opaque_cell_colors():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
    ])

    # Provide fully transparent colors; renderer should coerce to opaque fills
    heat = HeatmapAnnotationConfig(
        values="mut_type", colors={"SNV": (1, 0, 0, 0), "CNV": (0, 1, 0, 0)}
    )
    cfg = OncoplotConfig(heatmap_annotation=heat, x_col="patient_id", y_col="gene")
    fig = OncoPlotter(pdf, config=cfg).plot()
    ax = fig.axes[0]

    alphas = []
    for patch in ax.patches:
        face = patch.get_facecolor()
        if isinstance(face, np.ndarray):
            face = face[0] if face.ndim > 1 else face
        if len(face) >= 4:
            alpha_val = float(face[3])
            alphas.append(alpha_val)

    assert alphas, "No patch facecolors found"
    # Filter out any background patches (alpha 0) - we're testing cell colors
    cell_alphas = [a for a in alphas if a > 0.001]
    assert cell_alphas, "No non-transparent cell colors found"
    assert min(cell_alphas) > 0.01, f"Found transparent cell: {alphas}"


def test_oncoplot_can_hide_column_labels():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "CNV"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=False,
    )

    fig = OncoPlotter(pdf, config=cfg).plot()
    ax = fig.axes[0]

    rotated_xtick_text = [
        text.get_text()
        for text in ax.texts
        if isinstance(text, mtext.Text)
        and text.get_rotation() == 90
        and text.get_ha() == "center"
        and text.get_va() == "top"
    ]
    assert rotated_xtick_text == []


def test_oncoplot_auto_axes_aspect_preserves_plot_body_width_for_skinny_cells():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "CNV"},
        {"patient_id": "P3", "gene": "EGFR", "mut_type": "Fusion"},
        {"patient_id": "P4", "gene": "PIK3CA", "mut_type": "SNV"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
    )
    equal_cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        figsize=(12, 6),
        auto_adjust_cell_size=False,
        cell_aspect=0.35,
        axes_aspect_mode="equal",
    )
    auto_cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        figsize=(12, 6),
        auto_adjust_cell_size=False,
        cell_aspect=0.35,
        axes_aspect_mode="auto",
    )

    fig_equal = OncoPlotter(pdf, config=equal_cfg).plot()
    fig_auto = OncoPlotter(pdf, config=auto_cfg).plot()

    width_equal = fig_equal.axes[0].get_position().width
    width_auto = fig_auto.axes[0].get_position().width

    assert width_auto > width_equal


def test_oncoplot_adds_right_summary_bars_with_overall_counts():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P1", "gene": "TP53", "mut_type": "CNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "Fusion", "arm": "A"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "CNV", "arm": "B"},
        {"patient_id": "P4", "gene": "KRAS", "mut_type": "SNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
        legend_value_order=["SNV", "CNV", "Fusion"],
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        col_split_by=["arm"],
        col_split_order={"arm": ["A", "B"]},
        right_summary_bars=RightSummaryBarsConfig(include_overall=True),
    )

    fig = OncoPlotter(pdf, config=cfg).plot()

    assert len(fig.axes) >= 3
    summary_axes = fig.axes[1:]
    assert [axis.get_title() for axis in summary_axes[:3]] == ["All", "A", "B"]

    all_axis = summary_axes[0]
    summary_text = [t.get_text() for t in all_axis.texts if isinstance(t, mtext.Text)]
    assert "75%" in summary_text
    assert "50%" in summary_text

    widths = sorted([
        round(float(p.get_width()), 3)
        for p in all_axis.patches
        if round(float(p.get_width()), 3) > 0
    ])
    assert widths == [1.0, 1.0, 2.0, 2.0]


def test_oncoplot_right_summary_bars_respect_split_panels():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "SNV", "arm": "B"},
        {"patient_id": "P4", "gene": "KRAS", "mut_type": "Fusion", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        col_split_by=["arm"],
        col_split_order={"arm": ["A", "B"]},
        right_summary_bars=RightSummaryBarsConfig(include_overall=False),
    )

    fig = OncoPlotter(pdf, config=cfg).plot()

    assert [axis.get_title() for axis in fig.axes[1:]] == ["A", "B"]


def test_oncoplot_right_summary_uses_panel_gap_for_first_gap_by_default_and_allows_override():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "CNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    default_cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        right_summary_bars=RightSummaryBarsConfig(
            include_overall=True,
            split_by=["arm"],
            panel_width=0.08,
            panel_gap=0.03,
        ),
    )
    override_cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        right_summary_bars=RightSummaryBarsConfig(
            include_overall=True,
            split_by=["arm"],
            panel_width=0.08,
            panel_gap=0.03,
            heatmap_gap=0.07,
        ),
    )

    fig_default = OncoPlotter(pdf, config=default_cfg).plot()
    fig_override = OncoPlotter(pdf, config=override_cfg).plot()

    main_default = fig_default.axes[0].get_position()
    first_default = fig_default.axes[1].get_position()
    second_default = fig_default.axes[2].get_position()
    default_first_gap = round(first_default.x0 - main_default.x1, 3)
    default_panel_gap = round(second_default.x0 - first_default.x1, 3)

    main_override = fig_override.axes[0].get_position()
    first_override = fig_override.axes[1].get_position()
    second_override = fig_override.axes[2].get_position()
    override_first_gap = round(first_override.x0 - main_override.x1, 3)
    override_panel_gap = round(second_override.x0 - first_override.x1, 3)

    assert default_first_gap == default_panel_gap == 0.03
    assert override_first_gap == 0.07
    assert override_panel_gap == 0.03


def test_onco_prevalence_plotter_aggregates_groups_and_preserves_annotations():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A", "dose": "100 mg"},
        {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV", "arm": "A", "dose": "100 mg"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A", "dose": "100 mg"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "SNV", "arm": "B", "dose": "200 mg"},
        {"patient_id": "P4", "gene": "KRAS", "mut_type": "Fusion", "arm": "B", "dose": "200 mg"},
    ])
    row_groups = pd.DataFrame({"Pathway": {"TP53": "DNA Repair", "KRAS": "RAS"}}).rename_axis(
        "gene"
    )

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        row_group_col="Pathway",
        top_annotations={
            "Dose": TopAnnotationConfig(
                values="dose",
                colors={"100 mg": "#007352", "200 mg": "#860F0F"},
                legend_title="Dose",
            )
        },
        show_column_labels=True,
    )

    fig = OncoPrevalencePlotter(
        pdf,
        cfg,
        row_groups=row_groups,
        row_groups_color_dict={"DNA Repair": "#000000", "RAS": "#000000"},
        group_by=["arm"],
        include_overall=True,
    ).plot()
    ax = fig.axes[0]

    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ["All", "A", "B"]
    texts = {text.get_text() for text in ax.texts}
    assert {"100%", "50%", "Dose", "DNA Repair", "RAS"}.issubset(texts)
    assert ax.yaxis_inverted()
    assert len(fig.axes) >= 2


def test_onco_prevalence_plotter_adds_gap_only_after_all_column():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
        cell_aspect=1.0,
    )

    fig = OncoPrevalencePlotter(
        pdf,
        cfg,
        group_by=["arm"],
        include_overall=True,
        all_column_gap=0.35,
    ).plot()
    ax = fig.axes[0]

    xticks = [round(float(value), 2) for value in ax.get_xticks()]
    assert xticks == [0.5, 1.85, 2.85]
    assert round(xticks[1] - xticks[0], 2) == 1.35
    assert round(xticks[2] - xticks[1], 2) == 1.00


def test_onco_prevalence_plotter_show_all_alias_hides_overall_column():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoPrevalencePlotter(
        pdf, cfg, group_by=["arm"], include_overall=True, show_all=False
    ).plot()
    ax = fig.axes[0]

    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ["A", "B"]


def test_onco_prevalence_plotter_splits_mixed_top_annotations_by_membership():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "dose": "100 mg", "cohort": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "dose": "100 mg", "cohort": "B"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "SNV", "dose": "200 mg", "cohort": "A"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        top_annotations={
            "Cohort": TopAnnotationConfig(
                values="cohort",
                colors={"A": "#003975", "B": "#9d0ca2"},
                legend_title="Cohort",
                legend_value_order=["A", "B"],
            ),
            "Dose": TopAnnotationConfig(
                values="dose",
                colors={"100 mg": "#007352", "200 mg": "#860F0F"},
                legend_title="Dose",
                legend_value_order=["100 mg", "200 mg"],
            ),
        },
        top_annotation_order=["Cohort", "Dose"],
        show_column_labels=True,
    )

    fig = OncoPrevalencePlotter(pdf, cfg, group_by=["dose"], include_overall=False).plot()
    ax = fig.axes[0]

    top_patch_colors = {
        mcolors.to_hex(patch.get_facecolor(), keep_alpha=False)
        for patch in ax.patches
        if hasattr(patch, "get_y")
        and round(float(patch.get_y()), 2) < 0
        and hasattr(patch, "get_height")
        and round(float(patch.get_height()), 2) == 1.0
    }
    assert "#003975" in top_patch_colors
    assert "#9d0ca2" in top_patch_colors
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()] if legend else []
    assert "Mixed" not in legend_labels


def test_onco_gene_bar_plotter_draws_grouped_in_cell_bars():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A", "cohort": "X"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A", "cohort": "X"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "SNV", "arm": "B", "cohort": "Y"},
        {"patient_id": "P4", "gene": "KRAS", "mut_type": "Fusion", "arm": "B", "cohort": "Y"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
        legend_title="Mutation Type",
        legend_value_order=["SNV", "CNV", "Fusion"],
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        top_annotations={
            "Cohort": TopAnnotationConfig(
                values="cohort",
                colors={"X": "#003975", "Y": "#9d0ca2"},
                legend_title="Cohort",
            )
        },
        show_column_labels=True,
    )

    fig = OncoGeneBarPlotter(pdf, cfg, group_by=["arm"], include_overall=False).plot()
    ax = fig.axes[0]

    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ["A", "B"]
    texts = {text.get_text() for text in ax.texts}
    assert {"100%", "50%", "0%", "Cohort"}.issubset(texts)
    positive_widths = [
        round(float(patch.get_width()), 3)
        for patch in ax.patches
        if hasattr(patch, "get_width") and round(float(patch.get_width()), 3) > 0
    ]
    positive_bar_heights = [
        round(float(patch.get_height()), 2)
        for patch in ax.patches
        if hasattr(patch, "get_width")
        and hasattr(patch, "get_height")
        and round(float(patch.get_width()), 3) > 0
        and mcolors.to_hex(patch.get_facecolor(), keep_alpha=False) != "#ffffff"
    ]
    assert any(width < 1.0 for width in positive_widths)
    assert 1.0 in positive_bar_heights


def test_onco_gene_bar_plotter_can_show_total_and_category_counts_in_labels():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A"},
        {"patient_id": "P3", "gene": "TP53", "mut_type": "Fusion", "arm": "A"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
        legend_value_order=["SNV", "CNV", "Fusion"],
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoGeneBarPlotter(
        pdf,
        cfg,
        group_by=["arm"],
        include_overall=False,
        percent_decimals=0,
        show_total_counts=True,
        show_category_breakdown=True,
        show_category_counts=True,
    ).plot()
    ax = fig.axes[0]

    labels = [text.get_text() for text in ax.texts if isinstance(text, mtext.Text)]
    assert any("100% (3/3)" in label for label in labels)
    assert any("SNV: 33% (1), CNV: 33% (1)" in label for label in labels)
    assert any("Fusion: 33% (1)" in label for label in labels)


def test_onco_gene_bar_plotter_omits_zero_breakdown_categories_from_labels():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"},
        legend_value_order=["SNV", "CNV", "Fusion"],
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoGeneBarPlotter(
        pdf,
        cfg,
        group_by=["arm"],
        include_overall=False,
        percent_decimals=0,
        show_total_counts=True,
        show_category_breakdown=True,
        show_category_counts=True,
    ).plot()
    ax = fig.axes[0]

    labels = [text.get_text() for text in ax.texts if isinstance(text, mtext.Text)]
    assert any("SNV: 50% (1), CNV: 50% (1)" in label for label in labels)
    assert all("Fusion: 0% (0)" not in label for label in labels)


def test_onco_prevalence_plotter_can_override_label_text_color():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "TP53", "mut_type": "CNV", "arm": "A"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoPrevalencePlotter(
        pdf,
        cfg,
        group_by=["arm"],
        include_overall=False,
        show_total_counts=True,
        label_text_color="#ff00aa",
    ).plot()
    ax = fig.axes[0]

    percent_texts = [text for text in ax.texts if text.get_text() == "100% (2/2)"]
    assert percent_texts
    assert mcolors.to_hex(percent_texts[0].get_color()) == "#ff00aa"


def test_onco_gene_bar_plotter_can_customize_empty_cell_color_and_disable_border():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "CNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoGeneBarPlotter(
        pdf,
        cfg,
        group_by=["arm"],
        include_overall=False,
        empty_cell_color="gainsboro",
        cell_border_color=None,
    ).plot()
    ax = fig.axes[0]

    background_patches = [
        patch
        for patch in ax.patches
        if hasattr(patch, "get_height")
        and hasattr(patch, "get_width")
        and round(float(patch.get_height()), 2) == 1.0
        and round(float(patch.get_width()), 2) == 1.0
    ]
    assert background_patches
    assert any(
        mcolors.to_hex(patch.get_facecolor(), keep_alpha=False) == "#dcdcdc"
        for patch in background_patches
    )
    assert all(
        mcolors.to_rgba(patch.get_edgecolor())[3] == 0 or patch.get_linewidth() == 0
        for patch in background_patches
    )


def test_onco_gene_bar_plotter_show_all_alias_hides_overall_column():
    pdf = pd.DataFrame([
        {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV", "arm": "A"},
        {"patient_id": "P2", "gene": "KRAS", "mut_type": "CNV", "arm": "B"},
    ])

    heat = HeatmapAnnotationConfig(
        values="mut_type",
        colors={"SNV": "#EC745C", "CNV": "#44A9CC"},
    )
    cfg = OncoplotConfig(
        heatmap_annotation=heat,
        x_col="patient_id",
        y_col="gene",
        show_column_labels=True,
    )

    fig = OncoGeneBarPlotter(pdf, cfg, group_by=["arm"], show_all=False).plot()
    ax = fig.axes[0]

    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xticklabels == ["A", "B"]


# %%
