"""bioviz package initializer.

Submodules are imported lazily to avoid importing heavy plotting
dependencies at package import time.
"""

__all__ = ["lineplot", "spiderplot", "oncoplot", "table", "style", "plot_utils", "plot_configs"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in __all__:
        module = __import__(f"bioviz.{name}", fromlist=[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
