import importlib
import sys
from importlib.metadata import PackageNotFoundError, version


def matplotlib_available():
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


def test_top_level_names_present():
    sys.path.insert(0, str(importlib.util.find_spec("bioviz").origin).rsplit("/", 2)[0])
    import bioviz

    for name in ("plots", "utils", "oncoplot", "lineplot", "table", "plot_configs"):
        assert name in bioviz.__all__
    # We intentionally do not resolve plotting callables at package root.
    # Advanced users should import from `bioviz.plots` for direct access.


def test_package_version_matches_installed_metadata():
    import bioviz

    try:
        installed_version = version("bioviz-kit") or "0.4.2"
    except PackageNotFoundError:
        installed_version = "0.4.2"

    assert bioviz.__version__ == installed_version


def test_plot_lazy_exports_are_in_dunder_all():
    import bioviz.plots as plots

    expected_exports = {
        "VolcanoPlotter",
        "LinePlotter",
        "DistributionPlotter",
        "plot_distribution",
        "format_pvalue",
        "add_pvalue_annotation",
    }
    assert expected_exports.issubset(set(plots.__all__))
