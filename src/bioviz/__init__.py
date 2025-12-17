"""bioviz package initializer.

Submodules are imported lazily to avoid importing heavy plotting
dependencies at package import time.
"""

__all__ = [
    "lineplot",
    "oncoplot",
    "table",
    "style",
    "plot_utils",
    "plot_configs",
    "plot_composites",
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

    if name in __all__:
        module = __import__(f"bioviz.{name}", fromlist=[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
