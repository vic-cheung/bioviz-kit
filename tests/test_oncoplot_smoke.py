import math
from bioviz.oncoplot import OncoplotPlotter
from bioviz.configs import OncoplotConfig
from bioviz.plot_configs import HeatmapAnnotationConfig
import pandas as pd


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
    plotter = OncoplotPlotter(pdf, config=config)
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


import math
import pytest
import pandas as pd

from bioviz.oncoplot import OncoplotPlotter
from bioviz.plot_configs import OncoplotConfig, HeatmapAnnotationConfig


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
    plotter = OncoplotPlotter(pdf, config=config)
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
