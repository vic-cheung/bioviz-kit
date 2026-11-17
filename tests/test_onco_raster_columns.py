import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from bioviz.configs import OncoplotConfig
from bioviz.plots import OncoPrevalenceRasterPlotter

COLORS = {
    "SV": "#ff0000",
    "CNV": "#0000ff",
    "FUSION": "#ffcc00",
    "Amp": "#cc00cc",
    "Del": "#00cc00",
}


def source():
    return pd.DataFrame(
        [
            ("P1", "A", "SV", None),
            ("P1", "A", "CNV", "Amp"),
            ("P1", "A", "CNV", "Del"),
            ("P1", "A", "CNV", "Amp"),
            ("P2", "A", "CNV", "Amp"),
            ("P3", "A", "CNV", "Del"),
            ("P1", "A::B", "FUSION", None),
            ("P2", "A/B", "FUSION", None),
            ("P3", "A;B", "FUSION", None),
            ("P4", "A", None, None),
        ],
        columns=["subject", "feature", "category", "detail"],
    ).assign(expanded=lambda frame: frame.detail.fillna(frame.category))


def plotter(frame=None, event_col="category", **kwargs):
    return OncoPrevalenceRasterPlotter(
        source() if frame is None else frame,
        OncoplotConfig(
            x_col="subject", y_col="feature", value_col=event_col, row_values_color_dict=COLORS
        ),
        event_band_mode="per_sample",
        percent_decimals=0,
        **kwargs,
    )


def patches(figure, color):
    return [
        patch for patch in figure.axes[0].patches if mcolors.to_hex(patch.get_facecolor()) == color
    ]


def test_column_switch_preserves_rows_and_unique_patient_prevalence():
    for column, first_height in [("category", 0.5), ("expanded", 1 / 3)]:
        renderer = plotter(event_col=column)
        figure = renderer.plot()
        try:
            genes, _, _ = renderer._compute_gene_rows()
            assert genes == ["A", "A::B", "A/B", "A;B"]
            assert patches(figure, COLORS["SV"])[0].get_height() == pytest.approx(first_height)
            assert all(p.get_height() == 1 for p in patches(figure, COLORS["FUSION"]))
            assert "75% (3/4)" in [text.get_text() for text in figure.axes[0].texts]
            summaries = renderer._column_gene_event_summaries(
                renderer.df, [{"samples": ["P1", "P2", "P3", "P4"]}]
            )
            assert summaries[0]["A"]["altered"] == 3
            if column == "category":
                assert summaries[0]["A"]["counts_by_type"] == {"CNV": 3, "SV": 1}
        finally:
            plt.close(figure)


def nested(**kwargs):
    return plotter(
        event_subtype_col="detail",
        event_subtype_bands={"CNV": ["Amp", "Del"]},
        event_subtype_colors={"Amp": COLORS["Amp"], "Del": COLORS["Del"]},
        **kwargs,
    )


def test_nested_bands_fixed_order_and_no_double_counting():
    renderer = nested()
    figure = renderer.plot()
    try:
        assert figure.axes[0].yaxis_inverted()
        amps = patches(figure, COLORS["Amp"])
        dels = patches(figure, COLORS["Del"])
        assert [(p.get_y(), p.get_height()) for p in amps] == [(0.5, 0.25), (0.0, 0.5)]
        assert [(p.get_y(), p.get_height()) for p in dels] == [(0.75, 0.25), (0.5, 0.5)]
        assert "75% (3/4)" in [text.get_text() for text in figure.axes[0].texts]
        summary = renderer._column_gene_event_summaries(
            renderer.df, [{"samples": ["P1", "P2", "P3", "P4"]}], subtype_col="detail"
        )[0]["A"]
        assert summary["counts_by_type"] == {"CNV": 3, "SV": 1}
        legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert "CNV: Amp (top)" in legend and "CNV: Del (bottom)" in legend
    finally:
        plt.close(figure)


def test_missing_subtype_retains_generic_parent_mark():
    frame = pd.DataFrame(
        [("P1", "A", "CNV", None)], columns=["subject", "feature", "category", "detail"]
    )
    renderer = nested(frame=frame)
    figure = renderer.plot()
    try:
        assert patches(figure, COLORS["CNV"])[0].get_height() == 1
        assert not patches(figure, COLORS["Amp"])
    finally:
        plt.close(figure)


def test_arbitrary_column_labels_and_categories_work_with_global_bands():
    frame = pd.DataFrame({"id": ["a"], "row": ["foo/bar"], "kind": ["parent"], "child": ["low"]})
    renderer = OncoPrevalenceRasterPlotter(
        frame,
        OncoplotConfig(
            x_col="id", y_col="row", value_col="kind", row_values_color_dict={"parent": "#0000ff"}
        ),
        event_band_order=["other", "parent"],
        event_band_mode="global",
        event_subtype_col="child",
        event_subtype_bands={"parent": ["high", "low"]},
        event_subtype_colors={"high": "#ff0000", "low": "#00ff00"},
    )
    figure = renderer.plot()
    try:
        mark = patches(figure, "#00ff00")[0]
        assert (mark.get_y(), mark.get_height()) == (0.75, 0.25)
        assert renderer._compute_gene_rows()[0] == ["foo/bar"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"event_subtype_col": "detail"}, "supplied together"),
        (
            {"event_subtype_col": "detail", "event_subtype_bands": {"CNV": "Amp"}},
            "sequences of labels",
        ),
        ({"event_subtype_col": "category", "event_subtype_bands": {"CNV": ["Amp"]}}, "must differ"),
        (
            {"event_subtype_col": "missing", "event_subtype_bands": {"CNV": ["Amp"]}},
            "Missing event subtype column",
        ),
        (
            {"event_subtype_col": "detail", "event_subtype_bands": {"CNV": ["Amp", "Amp"]}},
            "Duplicate subtype",
        ),
        (
            {"event_subtype_col": "detail", "event_subtype_bands": {"CNV": ["Amp", "Del"]}},
            "Missing subtype colors",
        ),
        (
            {
                "event_subtype_col": "detail",
                "event_subtype_bands": {"CNV": ["Amp"]},
                "event_subtype_colors": {"Amp": "red"},
            },
            "Unconfigured subtypes",
        ),
    ],
)
def test_invalid_subtype_configuration_fails_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plotter(**kwargs)
