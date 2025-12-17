import importlib
import sys


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
