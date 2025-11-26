import sys
import math
from bioviz.oncoplot import OncoplotPlotter
from bioviz.plot_configs import OncoplotConfig, HeatmapAnnotationConfig
import pandas as pd


def run():
    pdf = pd.DataFrame([
        {'patient_id': 'P1', 'gene': 'TP53', 'mut_type': 'SNV'},
        {'patient_id': 'P1', 'gene': 'KRAS', 'mut_type': 'CNV'},
        {'patient_id': 'P2', 'gene': 'TP53', 'mut_type': 'SV'},
    ])

    heat = HeatmapAnnotationConfig(values='mut_type', colors={'SNV':'#ff0000','CNV':'#00ff00','SV':'#0000ff'})
    config = OncoplotConfig(heatmap_annotation=heat, x_col='patient_id', y_col='gene')
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
            path = getattr(p, 'get_path', None)
            if path is not None:
                verts = p.get_path().vertices
                cx = verts[:, 0].mean()
                cy = verts[:, 1].mean()
                centers.append((cx, cy))

    if not centers:
        print('FAIL: no patch centers found', file=sys.stderr)
        return 2

    # Assert centers are near integer or half-integer depending on drawing origin
    for cx, cy in centers:
        if not (math.isclose(cx, round(cx), abs_tol=1e-6) or math.isclose(cx, round(cx) - 0.5, abs_tol=1e-6)):
            print(f'FAIL: cx {cx} not aligned', file=sys.stderr)
            return 3
        if not (math.isclose(cy, round(cy), abs_tol=1e-6) or math.isclose(cy, round(cy) - 0.5, abs_tol=1e-6)):
            print(f'FAIL: cy {cy} not aligned', file=sys.stderr)
            return 4

    print('OK: oncoplot smoke test passed')
    return 0


if __name__ == '__main__':
    rc = run()
    sys.exit(rc)
