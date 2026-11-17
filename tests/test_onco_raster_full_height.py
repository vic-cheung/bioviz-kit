import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd

from bioviz.configs import HeatmapAnnotationConfig, OncoplotConfig
from bioviz.plots import OncoPrevalenceRasterPlotter


def test_per_sample_raster_fills_height_and_splits_only_present_events():
    frame = pd.DataFrame(
        [
            {"patient_id": "P1", "gene": "GENE1", "mut_type": "SNV", "arm": "A"},
            {"patient_id": "P1", "gene": "GENE1", "mut_type": "CNV", "arm": "A"},
            {"patient_id": "P2", "gene": "GENE1", "mut_type": "SNV", "arm": "A"},
            {"patient_id": "P3", "gene": "GENE2", "mut_type": "CNV", "arm": "A"},
        ]
    )
    config = OncoplotConfig(
        x_col="patient_id",
        y_col="gene",
        heatmap_annotation=HeatmapAnnotationConfig(
            values="mut_type", colors={"SNV": "#ff0000", "CNV": "#0000ff"}
        ),
    )
    figures = []
    try:
        for mode, expected_heights in [("global", {0.25}), ("per_sample", {0.5, 1.0})]:
            figure = OncoPrevalenceRasterPlotter(
                frame,
                config,
                group_by=["arm"],
                include_overall=False,
                event_band_order=["SNV", "SV", "CNV", "FUSION"],
                event_band_mode=mode,
                show_total_counts=True,
                percent_decimals=0,
            ).plot()
            figures.append(figure)
            axis = figure.axes[0]
            patches = [
                patch
                for patch in axis.patches
                if colors.to_hex(patch.get_facecolor()) in {"#ff0000", "#0000ff"}
            ]
            assert {round(patch.get_height(), 3) for patch in patches} == expected_heights
            assert "67% (2/3)" in {text.get_text() for text in axis.texts}
            if mode == "per_sample":
                assert len(patches) == 4
                assert all(patch.get_width() < 0.34 for patch in patches)
                # The two P1 event colors jointly fill exactly one row at one x slot.
                halves = [patch for patch in patches if patch.get_height() == 0.5]
                assert halves[0].get_x() == halves[1].get_x()
                assert abs(halves[0].get_y() - halves[1].get_y()) == 0.5
    finally:
        for figure in figures:
            plt.close(figure)
