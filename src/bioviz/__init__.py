"""bioviz package initializer.

Submodules and a small set of convenience plotting callables are imported
lazily to avoid importing heavy plotting dependencies at package import time.

Available convenience callables (resolved on first access):
- ``oncoplot`` -> :mod:`bioviz.plots.oncoplot`.oncoplot
- ``lineplot`` -> :mod:`bioviz.plots.lineplot`.lineplot
- ``waterfall`` -> :mod:`bioviz.plots.waterfall`.waterfall
- ``table`` -> :mod:`bioviz.plots.table`.table
- ``volcano`` -> :mod:`bioviz.plots.volcano`.volcano

Advanced users can still import submodules directly from
``bioviz.plots.*`` if they need finer-grained control.
"""

__all__ = [
    "lineplot",
    "oncoplot",
    "table",
    "style",
    "utils",
    "plots",
]

__all__ = [
    "lineplot",
    "oncoplot",
    "table",
    "style",
    "plot_configs",
    "utils",
    "plots",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """
    Lazy import submodules on attribute access.

    Args:
        name: Submodule name to import (one of the names exposed in `__all__`).

    Returns:
        The imported submodule object.

    Raises:
        AttributeError: If `name` is not a recognized submodule.
    """

    # Lazily import submodules listed in __all__ on first access. We do not
    # provide legacy convenience callables at the package root to keep the
    # initializer lightweight and avoid exposing symbols unexpectedly.
    if name in __all__:
        module = __import__(f"bioviz.{name}", fromlist=[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__} has no attribute {name}")
