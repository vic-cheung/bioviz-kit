"""Compatibility shim for plotting configuration models.

This module re-exports per-plot config classes from :mod:`bioviz.configs` so
imports like ``from bioviz.plot_configs import LinePlotConfig`` continue to work
while the codebase migrates to the new per-plot modules.
"""

from .configs import LinePlotConfig, StyledTableConfig, OncoplotConfig
from .configs.base_cfg import BasePlotConfig

# Re-export the annotation types from the oncoplot config module for
# backwards-compatibility with `from bioviz.plot_configs import TopAnnotationConfig`.
from .configs.oncoplot_cfg import TopAnnotationConfig, HeatmapAnnotationConfig
from .configs.oncoplot_annotations_cfg import make_annotation_config


__all__ = [
    "BasePlotConfig",
    "LinePlotConfig",
    "StyledTableConfig",
    "OncoplotConfig",
    "TopAnnotationConfig",
    "HeatmapAnnotationConfig",
    "make_annotation_config",
]
