"""Plot composites subpackage for higher-level plotting utilities.

Attributes are loaded lazily to avoid importing optional heavy
dependencies (e.g. `statsmodels`) when the parent package is imported.
"""

__all__ = [
    "grouped",
    "waterfall",
    "volcano",
    "oncoplot",
    "lineplot",
    "table",
    # convenience functions / classes
    "plot_volcano",
    "resolve_labels",
    "plot_waterfall",
    "waterfall_with_distribution",
    "plot_grouped_boxplots",
    "plot_oncoplot",
    "OncoplotPlotter",
    "generate_lineplot",
    "generate_styled_table",
]

from importlib import import_module
from typing import List

# Map convenience/exported names -> (module_path, attribute_name)
_PUBLIC_FUNCS = {
    "plot_volcano": ("bioviz.plots.volcano", "plot_volcano"),
    "resolve_labels": ("bioviz.plots.volcano", "resolve_labels"),
    "plot_waterfall": ("bioviz.plots.waterfall", "plot_waterfall"),
    "waterfall_with_distribution": (
        "bioviz.plots.waterfall",
        "waterfall_with_distribution",
    ),
    "plot_grouped_boxplots": ("bioviz.plots.grouped", "plot_grouped_boxplots"),
    # oncoplot exports
    "plot_oncoplot": ("bioviz.plots.oncoplot", "plot_oncoplot"),
    "OncoplotPlotter": ("bioviz.plots.oncoplot", "OncoplotPlotter"),
    # volcano class wrapper
    "VolcanoPlotter": ("bioviz.plots.volcano", "VolcanoPlotter"),
    # lineplot / table helpers
    "generate_lineplot": ("bioviz.plots.lineplot", "generate_lineplot"),
    "generate_styled_table": ("bioviz.plots.table", "generate_styled_table"),
}


def __getattr__(name: str):
    # Resolve convenience callables first
    if name in _PUBLIC_FUNCS:
        mod_name, attr_name = _PUBLIC_FUNCS[name]
        mod = import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val

    # Then resolve submodules lazily
    if name in __all__:
        mod = import_module(f"bioviz.plots.{name}")
        globals()[name] = mod
        return mod

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> List[str]:
    names = set(globals().keys())
    names.update(_PUBLIC_FUNCS.keys())
    names.update(__all__)
    return sorted(names)
