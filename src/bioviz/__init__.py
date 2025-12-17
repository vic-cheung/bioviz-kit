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

    # First handle the existing submodule imports
    if name in __all__ and name not in _PUBLIC_FUNCS:
        module = __import__(f"bioviz.{name}", fromlist=[name])
        globals()[name] = module
        return module

    # Then handle lazy exported convenience functions
    if name in _PUBLIC_FUNCS:
        mod_name, func_name = _PUBLIC_FUNCS[name]
        module = __import__(mod_name, fromlist=[func_name])
        func = getattr(module, func_name)
        globals()[name] = func
        return func
    raise AttributeError(f"module {__name__} has no attribute {name}")
