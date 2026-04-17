"""Tests and visual demo for preserve_x_categories on LinePlotConfig.

Run with pytest or directly:
    python tests/test_preserve_x_categories.py

Generates a PDF at tests/preserve_x_categories_demo.pdf with side-by-side
comparisons of preserve_x_categories=True vs False.
"""

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages

from bioviz.configs import LinePlotConfig
from bioviz.plots.lineplot import LinePlotter, generate_lineplot_twinx

TESTS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TIMEPOINTS = ["Week 0", "Week 4", "Week 8", "Week 12"]


@pytest.fixture
def full_df():
    """Two series tracked across all four timepoints."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                _TIMEPOINTS * 2,
                categories=_TIMEPOINTS,
                ordered=True,
            ),
            "measurement": [10.0, 25.0, 18.0, 30.0, 5.0, 12.0, 8.0, 22.0],
            "sensor": ["Alpha"] * 4 + ["Beta"] * 4,
        }
    )


@pytest.fixture
def sparse_first_only():
    """Only the first timepoint has data; later ones should still appear when preserved."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                ["Week 0"],
                categories=_TIMEPOINTS,
                ordered=True,
            ),
            "measurement": [15.0],
            "sensor": ["Gamma"],
        }
    )


@pytest.fixture
def sparse_middle():
    """Only middle timepoints present; first and last are empty."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                ["Week 4", "Week 8"],
                categories=_TIMEPOINTS,
                ordered=True,
            ),
            "measurement": [20.0, 35.0],
            "sensor": ["Delta", "Delta"],
        }
    )


@pytest.fixture
def sparse_last_only():
    """Only the final timepoint has data."""
    return pd.DataFrame(
        {
            "timepoint": pd.Categorical(
                ["Week 12"],
                categories=_TIMEPOINTS,
                ordered=True,
            ),
            "measurement": [42.0],
            "sensor": ["Epsilon"],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPreserveXCategories:
    """Verify preserve_x_categories keeps all category ticks."""

    def test_preserve_true_keeps_all_ticks(self, sparse_first_only):
        cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            preserve_x_categories=True,
            title="preserve=True (first only)",
        )
        lp = LinePlotter(sparse_first_only, cfg)
        fig, ax = lp.plot()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for tp in _TIMEPOINTS:
            assert tp in tick_labels

    def test_preserve_false_drops_unused(self, sparse_first_only):
        cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            preserve_x_categories=False,
            title="preserve=False (first only)",
        )
        lp = LinePlotter(sparse_first_only, cfg)
        fig, ax = lp.plot()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Week 0" in tick_labels
        assert "Week 4" not in tick_labels
        assert "Week 8" not in tick_labels
        assert "Week 12" not in tick_labels

    def test_preserve_true_middle_categories(self, sparse_middle):
        """Middle-only data should still show all timepoints when preserved."""
        cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            preserve_x_categories=True,
            title="preserve=True (middle only)",
        )
        lp = LinePlotter(sparse_middle, cfg)
        fig, ax = lp.plot()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for tp in _TIMEPOINTS:
            assert tp in tick_labels

    def test_preserve_true_last_only(self, sparse_last_only):
        """Last-only data should still show all timepoints when preserved."""
        cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            preserve_x_categories=True,
            title="preserve=True (last only)",
        )
        lp = LinePlotter(sparse_last_only, cfg)
        fig, ax = lp.plot()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for tp in _TIMEPOINTS:
            assert tp in tick_labels

    def test_preserve_true_twinx(self, full_df, sparse_first_only):
        """Twinx secondary with sparse data should preserve categories."""
        primary_cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            group_col="sensor",
            preserve_x_categories=True,
            title="Twinx: primary full",
        )
        secondary_cfg = LinePlotConfig(
            x="timepoint",
            y="measurement",
            label_col="sensor",
            group_col="sensor",
            preserve_x_categories=True,
            ylabel="Secondary Axis",
        )
        fig = generate_lineplot_twinx(
            df=full_df,
            twinx_data=sparse_first_only,
            primary_config=primary_cfg,
            secondary_config=secondary_cfg,
        )
        assert fig is not None
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        for tp in _TIMEPOINTS:
            assert tp in tick_labels


# ---------------------------------------------------------------------------
# PDF visual demo
# ---------------------------------------------------------------------------


def _make_config(preserve: bool, title: str) -> LinePlotConfig:
    return LinePlotConfig(
        x="timepoint",
        y="measurement",
        label_col="sensor",
        label_points=True,
        preserve_x_categories=preserve,
        title=title,
        ylim=(0, None),
    )


def generate_demo_pdf():
    """Create a multi-page PDF comparing preserve_x_categories True vs False."""
    import matplotlib.pyplot as plt

    pdf_path = TESTS_DIR / "preserve_x_categories_demo.pdf"

    full_df = pd.DataFrame(
        {
            "timepoint": pd.Categorical(_TIMEPOINTS * 2, categories=_TIMEPOINTS, ordered=True),
            "measurement": [10.0, 25.0, 18.0, 30.0, 5.0, 12.0, 8.0, 22.0],
            "sensor": ["Alpha"] * 4 + ["Beta"] * 4,
        }
    )

    sparse_first = pd.DataFrame(
        {
            "timepoint": pd.Categorical(["Week 0"], categories=_TIMEPOINTS, ordered=True),
            "measurement": [15.0],
            "sensor": ["Gamma"],
        }
    )

    sparse_middle = pd.DataFrame(
        {
            "timepoint": pd.Categorical(["Week 4", "Week 8"], categories=_TIMEPOINTS, ordered=True),
            "measurement": [20.0, 35.0],
            "sensor": ["Delta", "Delta"],
        }
    )

    sparse_last = pd.DataFrame(
        {
            "timepoint": pd.Categorical(["Week 12"], categories=_TIMEPOINTS, ordered=True),
            "measurement": [42.0],
            "sensor": ["Epsilon"],
        }
    )

    scenarios = [
        ("All timepoints populated", full_df),
        ("Only first timepoint", sparse_first),
        ("Only middle timepoints", sparse_middle),
        ("Only last timepoint", sparse_last),
    ]

    with PdfPages(pdf_path) as pdf:
        for scenario_name, df in scenarios:
            for preserve in (True, False):
                label = f"{scenario_name}\npreserve_x_categories={preserve}"
                cfg = _make_config(preserve=preserve, title=label)
                lp = LinePlotter(df.copy(), cfg)
                fig, ax = lp.plot()
                if fig is not None:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1)
                    plt.close(fig)

    print(f"Saved → {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    generate_demo_pdf()
