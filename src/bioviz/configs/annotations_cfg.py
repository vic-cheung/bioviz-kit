"""Deprecated shim; use oncoplot_annotations_cfg for oncoplot annotation configs."""

from .oncoplot_annotations_cfg import (  # noqa: F401
    HeatmapAnnotationConfig,
    TopAnnotationConfig,
    make_annotation_config,
)

__all__ = [
    "TopAnnotationConfig",
    "HeatmapAnnotationConfig",
    "make_annotation_config",
]
