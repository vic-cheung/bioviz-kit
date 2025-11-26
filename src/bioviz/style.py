"""Style interfaces and a neutral default style for bioviz.

Define a small `StyleBase` protocol that plotters can rely on. Plotters
should accept an optional `style` argument and fall back to `DefaultStyle`.
"""

from __future__ import annotations
from typing import Protocol, Optional, Mapping

import matplotlib as mpl


class StyleBase(Protocol):
    """Minimal style protocol used by bioviz plotters."""

    palette: Mapping[str, str]
    font_family: str
    base_fontsize: int

    def apply_theme(self, rc_overrides: Optional[dict] = None) -> None: ...


class DefaultStyle:
    """
    Neutral, minimal style used when no external style is provided.
    """

    def __init__(self) -> None:
        self.font_family = "DejaVu Sans"
        self.base_fontsize = 12
        # a small, neutral palette; consumers can override fully
        self.palette = {
            "mutant": "#1f77b4",
            "wildtype": "#d3d3d3",
            "background": "#ffffff",
        }

    def apply_theme(self, rc_overrides: Optional[dict] = None) -> None:
        rc = {
            "font.family": self.font_family,
            "font.size": self.base_fontsize,
            "axes.titlesize": int(self.base_fontsize * 1.1),
            "axes.labelsize": self.base_fontsize,
            "xtick.labelsize": int(self.base_fontsize * 0.9),
            "ytick.labelsize": int(self.base_fontsize * 0.9),
            "figure.facecolor": self.palette.get("background", "white"),
        }
        if rc_overrides:
            rc.update(rc_overrides)
        mpl.rcParams.update(rc)


__all__ = ["StyleBase", "DefaultStyle"]
