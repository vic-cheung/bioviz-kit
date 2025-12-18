# %%
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from bioviz.plots import OncoPlotter
from bioviz.configs import HeatmapAnnotationConfig, OncoplotConfig


# %%
def test_oncoplot_shapes_centered():
    pdf = pd.DataFrame(
        [
            {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
            {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
            {"patient_id": "P2", "gene": "TP53", "mut_type": "Fusion"},
        ]
    )

    heat = HeatmapAnnotationConfig(
        values="mut_type", colors={"SNV": "#EC745C", "CNV": "#44A9CC", "Fusion": "#FFB600"}
    )
    config = OncoplotConfig(heatmap_annotation=heat, x_col="patient_id", y_col="gene")
    plotter = OncoPlotter(pdf, config=config)
    fig = plotter.plot()
    ax = fig.axes[0]

    # Expect shapes to be drawn as patches; compute their centers and assert they are near integer centers
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

    tol = 0.3
    for cx, cy in centers:
        assert abs(cx - round(cx)) <= tol and abs(cy - round(cy)) <= tol


def test_oncoplot_cell_alignment(tmp_path):
    pdf = pd.DataFrame(
        [
            {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
            {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
            {"patient_id": "P2", "gene": "TP53", "mut_type": "SV"},
        ]
    )

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

    # Expected centers are at (0,0) for P1-TP53, (0,1) for P1-KRAS, (1,0) for P2-TP53
    expected = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)}

    rounded = {(round(cx, 2), round(cy, 2)) for cx, cy in centers}
    assert expected.issubset(rounded), f"Expected centers {expected} in {rounded}"


def test_oncoplot_transparent_figure_patch():
    pdf = pd.DataFrame(
        [
            {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
            {"patient_id": "P2", "gene": "TP53", "mut_type": "SNV"},
        ]
    )

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
    pdf = pd.DataFrame(
        [
            {"patient_id": "P1", "gene": "TP53", "mut_type": "SNV"},
            {"patient_id": "P1", "gene": "KRAS", "mut_type": "CNV"},
        ]
    )

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
            alphas.append(face[3])

    assert alphas, "No patch facecolors found"
    assert min(alphas) > 0.01


# %%
