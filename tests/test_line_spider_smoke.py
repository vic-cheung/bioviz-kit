import pandas as pd
from bioviz.configs import StyledLinePlotConfig, StyledSpiderPlotConfig
from bioviz.lineplot import generate_styled_lineplot
from bioviz.spiderplot import generate_styled_spiderplot


def test_lineplot_accepts_non_categorical_x_and_respects_background():
    df = pd.DataFrame(
        {
            "Timepoint": ["C1D1", "C2D1"],
            "Value": [1.0, 2.0, 3.0],
            "label": ["KRAS"] * 3,
            "Variant_type": ["SNV"] * 3,
        }
    )
    cfg = StyledLinePlotConfig(patient_id="P1", figure_facecolor="#abcdef")
    fig = generate_styled_lineplot(df, cfg)
    assert fig is not None
    assert fig.patch.get_facecolor()[:3] == (171 / 255, 205 / 255, 239 / 255)
    assert fig.patch.get_alpha() == 1.0


def test_lineplot_allows_transparent_background():
    df = pd.DataFrame(
        {
            "Timepoint": ["C1D1"],
            "Value": [1.0, 2.0],
            "label": ["KRAS", "KRAS"],
            "Variant_type": ["SNV", "SNV"],
        }
    )
    cfg = StyledLinePlotConfig(patient_id="P1", figure_transparent=True)
    fig = generate_styled_lineplot(df, cfg)
    assert fig is not None
    assert fig.patch.get_alpha() == 0.0


def test_spiderplot_coerces_plain_x_and_background():
    df = pd.DataFrame(
        {
            "patient": ["P1", "P1", "P1"],
            "tp": ["C1D1", "C2D1"],
            "val": [0, 10, 20],
        }
    )
    cfg = StyledSpiderPlotConfig(
        group_col="patient",
        x="tp",
        y="val",
        marker_style="o",
        figure_facecolor="white",
    )
    fig, handles, labels = generate_styled_spiderplot(df, cfg)
    assert fig is not None
    assert fig.patch.get_alpha() == 1.0
    assert handles and labels
