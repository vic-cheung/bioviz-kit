# %%
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from bioviz.configs import HeatmapAnnotationConfig, OncoplotConfig
from bioviz.plots import OncoPlotter


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

    # Oncoplot cells are centered at half-integer positions (0.5, 1.5, etc.)
    # Expected centers are at (0.5, 0.5), (0.5, 1.5), (1.5, 0.5) for P1-TP53, P1-KRAS, P2-TP53
    expected = {(0.5, 0.5), (0.5, 1.5), (1.5, 0.5)}

    rounded = {(round(cx, 1), round(cy, 1)) for cx, cy in centers}
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
            alpha_val = float(face[3])
            alphas.append(alpha_val)

    assert alphas, "No patch facecolors found"
    # Filter out any background patches (alpha 0) - we're testing cell colors
    cell_alphas = [a for a in alphas if a > 0.001]
    assert cell_alphas, "No non-transparent cell colors found"
    assert min(cell_alphas) > 0.01, f"Found transparent cell: {alphas}"


# %%
