import sys
from bioviz.oncoplot import OncoplotPlotter
from bioviz.configs import OncoplotConfig
from bioviz.configs import HeatmapAnnotationConfig
import pandas as pd


def run():
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

    centers = []
    for p in ax.patches:
        try:
            bbox = p.get_bbox()
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            centers.append((cx, cy))
        except Exception:
            path = getattr(p, "get_path", None)
            if path is not None:
                verts = p.get_path().vertices
                cx = verts[:, 0].mean()
                cy = verts[:, 1].mean()
                centers.append((cx, cy))

    if not centers:
        print("FAIL: no patch centers found", file=sys.stderr)
        return 2

    # Accept rectangles and diagonal triangles: centroid should be within 0.3 of an integer grid center
    tol = 0.3
    for cx, cy in centers:
        if not (abs(cx - round(cx)) <= tol and abs(cy - round(cy)) <= tol):
            print(
                f"FAIL: centroid ({cx:.3f},{cy:.3f}) not within {tol} of integer grid",
                file=sys.stderr,
            )
            return 3

    print("OK: oncoplot smoke test passed")
    return 0


if __name__ == "__main__":
    rc = run()
    sys.exit(rc)
