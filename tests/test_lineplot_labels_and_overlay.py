"""Tests for label_points, legend_title, and overlay vline functionality."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from bioviz.configs import LinePlotConfig
from bioviz.plots.lineplot import (
    LinePlotter,
    generate_lineplot_twinx,
    generate_styled_lineplot,
)

# ---------------------------------------------------------------------------
# Fixtures — generic time-series data
# ---------------------------------------------------------------------------


@pytest.fixture
def timeseries_df():
    """Two series tracked across three timepoints."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                ["T1", "T2", "T3"] * 2,
                categories=["T1", "T2", "T3"],
                ordered=True,
            ),
            "value": [1.0, 2.5, 0.8, 4.0, 5.0, 6.0],
            "series": [
                "Series A",
                "Series A",
                "Series A",
                "Series B",
                "Series B",
                "Series B",
            ],
        }
    )


@pytest.fixture
def overlay_df():
    """Secondary series with per-timepoint annotations for overlay testing."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                ["T1", "T2", "T3"],
                categories=["T1", "T2", "T3"],
                ordered=True,
            ),
            "y": [0.0, -20.0, -40.0],
            "category": ["Sensor X", "Sensor X", "Sensor X"],
            "status": ["Low", "Medium", "Medium"],
        }
    )


# ---------------------------------------------------------------------------
# label_points: text artists have correct content
# ---------------------------------------------------------------------------


class TestLabelPoints:
    def test_label_points_creates_text_with_correct_labels(self, timeseries_df):
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            label_points=True,
            title="Label Points Test",
        )
        fig = generate_styled_lineplot(timeseries_df, cfg)
        assert fig is not None

        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts if t.get_text()]
        assert "Series A" in label_texts
        assert "Series B" in label_texts

    def test_label_points_false_produces_no_text(self, timeseries_df):
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            label_points=False,
            title="No Labels Test",
        )
        fig = generate_styled_lineplot(timeseries_df, cfg)
        assert fig is not None

        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts if t.get_text()]
        assert "Series A" not in label_texts
        assert "Series B" not in label_texts

    def test_label_points_one_per_series(self, timeseries_df):
        """Each series should get exactly one label (at its first timepoint)."""
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            label_points=True,
            title="One Label Per Series",
        )
        fig = generate_styled_lineplot(timeseries_df, cfg)
        ax = fig.axes[0]
        label_texts = [t.get_text() for t in ax.texts if t.get_text()]
        assert label_texts.count("Series A") == 1
        assert label_texts.count("Series B") == 1

    def test_label_points_via_plotter(self, timeseries_df):
        """LinePlotter wrapper should produce the same labels."""
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            label_points=True,
            title="Plotter Wrapper Test",
        )
        lp = LinePlotter(timeseries_df, cfg)
        fig, ax = lp.plot()
        assert fig is not None
        label_texts = [t.get_text() for t in ax.texts if t.get_text()]
        assert "Series A" in label_texts


# ---------------------------------------------------------------------------
# legend_title: override the legend header
# ---------------------------------------------------------------------------


class TestLegendTitle:
    def test_legend_title_overrides_label_col(self, timeseries_df):
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            legend_title="My Custom Legend",
            title="Legend Override Test",
        )
        fig = generate_styled_lineplot(timeseries_df, cfg)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert "My Custom Legend" in legend_labels
        assert "series" not in legend_labels

    def test_legend_title_defaults_to_label_col(self, timeseries_df):
        cfg = LinePlotConfig(
            x="timepoint",
            y="value",
            label_col="series",
            title="Legend Default Test",
        )
        fig = generate_styled_lineplot(timeseries_df, cfg)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [t.get_text() for t in legend.get_texts()]
        assert "series" in legend_labels


# ---------------------------------------------------------------------------
# overlay vlines: vertical lines per annotation value
# ---------------------------------------------------------------------------


class TestOverlayVlines:
    def test_overlay_vlines_appear_for_each_timepoint(self, overlay_df):
        primary_df = pd.DataFrame(
            {
                "timepoint": pd.Categorical(
                    ["T1", "T2", "T3"],
                    categories=["T1", "T2", "T3"],
                    ordered=True,
                ),
                "y_primary": [10.0, 20.0, 15.0],
                "group": ["Primary", "Primary", "Primary"],
            }
        )
        primary_cfg = LinePlotConfig(
            x="timepoint",
            y="y_primary",
            group_col="group",
            title="Primary Axis",
        )
        secondary_cfg = LinePlotConfig(
            x="timepoint",
            y="y",
            label_col="category",
            overlay_col="status",
            overlay_palette={"Low": "gray", "Medium": "green"},
            title="Secondary Axis",
        )
        fig = generate_lineplot_twinx(
            df=primary_df,
            twinx_data=overlay_df,
            primary_config=primary_cfg,
            secondary_config=secondary_cfg,
            annotation_color_dict={"Low": "gray", "Medium": "green"},
            annotation_source="secondary",
        )
        assert fig is not None
        # axvline creates Line2D objects; count those with x-data at a single point.
        ax = fig.axes[0]
        vlines = [
            line
            for line in ax.lines
            if hasattr(line, "get_xdata") and len(set(line.get_xdata())) == 1
        ]
        assert len(vlines) >= 1, f"Expected vlines but found {len(vlines)}"
