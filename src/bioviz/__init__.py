"""bioviz package initializer.

Submodules and convenience plotting classes/functions are imported lazily to
avoid importing heavy dependencies (matplotlib, lifelines) at package import.

Main entry points:
- ``bioviz.plots`` - All plotters (KMPlotter, OncoPlotter, VolcanoPlotter, etc.)
- ``bioviz.configs`` - All configuration classes (KMPlotConfig, OncoplotConfig, etc.)

Example usage::

    from bioviz.plots import KMPlotter
    from bioviz.configs import KMPlotConfig

    config = KMPlotConfig(time_col="time", event_col="event", group_col="arm")
    plotter = KMPlotter(df, config)
    fig, ax, pval = plotter.plot()

Advanced users can import submodules directly:
- ``bioviz.plots.km`` - KM plot module
- ``bioviz.plots.volcano`` - Volcano plot module
- etc.
"""

__all__ = [
    "configs",
    "plots",
    "utils",
    # Convenience aliases for direct submodule access
    "oncoplot",
    "lineplot",
    "table",
    "plot_configs",
]

__version__ = "0.3.4"

# Mapping of convenience aliases to their actual module paths
_ALIAS_MAP = {
    "oncoplot": "bioviz.plots.oncoplot",
    "lineplot": "bioviz.plots.lineplot",
    "table": "bioviz.plots.table",
    "plot_configs": "bioviz.configs",
}


def __getattr__(name: str):
    """Lazy import submodules on attribute access.

    Args:
        name: Submodule name to import (one of the names exposed in `__all__`).

    Returns:
        The imported submodule object.

    Raises:
        AttributeError: If `name` is not a recognized submodule.
    """
    # Handle aliased submodules
    if name in _ALIAS_MAP:
        module = __import__(_ALIAS_MAP[name], fromlist=[name.split(".")[-1]])
        globals()[name] = module
        return module

    # Handle direct submodules
    if name in ("configs", "plots", "utils"):
        module = __import__(f"bioviz.{name}", fromlist=[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
