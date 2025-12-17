"""Plot composites subpackage for higher-level plotting utilities.

Attributes are loaded lazily to avoid importing optional heavy
dependencies (e.g. `statsmodels`) when the parent package is imported.
"""

__all__ = ["plot_grouped_boxplots", "plot_waterfall", "plot_volcano", "waterfall_with_distribution", "resolve_labels"]


def __getattr__(name: str):
    if name == "plot_grouped_boxplots":
        from .grouped import plot_grouped_boxplots as _obj

        globals()[name] = _obj
        return _obj
    if name == "plot_waterfall":
        from .waterfall import plot_waterfall as _obj

        globals()[name] = _obj
        return _obj
    if name == "plot_volcano":
        from .volcano import plot_volcano as _obj

        globals()[name] = _obj
        return _obj
    if name == "resolve_labels":
        from .volcano import resolve_labels as _obj

        globals()[name] = _obj
        return _obj
    if name == "waterfall_with_distribution":
        from .waterfall import waterfall_with_distribution as _obj

        globals()[name] = _obj
        return _obj
    raise AttributeError(f"module {__name__} has no attribute {name}")
