import pandas as pd
from bioviz.configs import LinePlotConfig
from bioviz.plots.lineplot import generate_styled_lineplot


def test_lineplot_accepts_non_categorical_x_and_respects_background():
    df = pd.DataFrame(
        {
            "Timepoint": ["C1D1", "C2D1", "C3D1"],
            "Value": [1.0, 2.0, 3.0],
            "label": ["KRAS"] * 3,
            "Variant_type": ["SNV"] * 3,
        }
    )
    cfg = LinePlotConfig(
        patient_id="P1",
        label_col="label",
        x="Timepoint",
        y="Value",
        figure_facecolor="#abcdef",
    )
    fig = generate_styled_lineplot(df, cfg)
    assert fig is not None
    assert fig.patch.get_facecolor()[:3] == (171 / 255, 205 / 255, 239 / 255)
    assert fig.patch.get_alpha() == 1.0


def test_lineplot_allows_transparent_background():
    df = pd.DataFrame(
        {
            "Timepoint": ["C1D1", "C2D1"],
            "Value": [1.0, 2.0],
            "label": ["KRAS", "KRAS"],
            "Variant_type": ["SNV", "SNV"],
        }
    )
    cfg = LinePlotConfig(
        patient_id="P1",
        label_col="label",
        x="Timepoint",
        y="Value",
        figure_transparent=True,
    )
    fig = generate_styled_lineplot(df, cfg)
    assert fig is not None
    assert fig.patch.get_alpha() == 0.0


# Spider plot tests removed — spiderplot functionality consolidated under LinePlotConfig/lineplot
