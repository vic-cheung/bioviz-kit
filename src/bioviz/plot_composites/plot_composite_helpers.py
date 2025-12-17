"""Helper utilities for composite plots.

This contains small wrappers for palettes, stat tests and annotator helpers used
by the composite plotting functions.
"""

from typing import Iterable, List, Tuple

import pandas as pd
import numpy as np

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - optional
    mannwhitneyu = None

try:
    from statannotations.Annotator import Annotator
except Exception:  # pragma: no cover - optional
    Annotator = None

import seaborn as sns


def get_default_palette(n: int, prefer: str = "Set2") -> List[str]:
    """Return a palette of length `n`.

    Prefer seaborn palettes (`Set2` or `Dark2`) for public package defaults.
    """
    if prefer == "Dark2":
        base = sns.color_palette("Dark2", n_colors=n)
    else:
        base = sns.color_palette(prefer, n_colors=n)
    return [sns.utils.rgb2hex(c) for c in base]


def compute_mwu_and_annot_pairs(
    df: pd.DataFrame, group_col: str, value_col: str, pairs: Iterable[Tuple[str, str]]
) -> List[Tuple[Tuple[str, str], float]]:
    results = []
    for a, b in pairs:
        try:
            va = df.loc[df[group_col] == a, value_col].dropna().values
            vb = df.loc[df[group_col] == b, value_col].dropna().values
            if len(va) == 0 or len(vb) == 0:
                p = np.nan
            else:
                if mannwhitneyu is None:
                    p = np.nan
                else:
                    p = mannwhitneyu(va, vb, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        results.append(((a, b), float(p) if not np.isnan(p) else np.nan))
    return results


def apply_statannotations(ax, data, x, y, pairs, test="Mann-Whitney", **kwargs):
    # If statannotations is available, use it for rich annotation placement.
    if Annotator is not None:
        annot = Annotator(ax, pairs, data=data, x=x, y=y)
        annot.configure(test=test, text_format="star", **kwargs)
        annot.apply_and_annotate()
        return annot

    # Lightweight fallback: compute p-values (if possible) and draw simple
    # bracket lines + text at the top of the axis. This keeps the package
    # usable without depending on `statannotations`.
    def _p_to_label(p):
        if p is None or np.isnan(p):
            return "n.s."
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    # compute p-values for pairs using available function (may return nan)
    computed = compute_mwu_and_annot_pairs(data, x, y, pairs)

    # Determine x positions: prefer axis tick positions, else assign by category order
    xticks = [t.get_text() for t in ax.get_xticklabels()]
    if len(xticks) and all(t != "" for t in xticks):
        tick_positions = dict(zip(xticks, ax.get_xticks()))
    else:
        cats = list(pd.Categorical(data[x]).categories)
        tick_positions = {str(cat): i for i, cat in enumerate(cats)}

    y0, y1 = ax.get_ylim()
    top = y1
    pad = (y1 - y0) * 0.05
    cur = 0
    for (a, b), p in computed:
        x1 = tick_positions.get(str(a), None)
        x2 = tick_positions.get(str(b), None)
        if x1 is None or x2 is None:
            continue
        y_coord = top + pad * (cur + 1)
        # Draw horizontal line
        ax.plot([x1, x2], [y_coord, y_coord], color="k", linewidth=1)
        # Small vertical ticks
        ax.plot([x1, x1], [y_coord - pad * 0.2, y_coord], color="k", linewidth=1)
        ax.plot([x2, x2], [y_coord - pad * 0.2, y_coord], color="k", linewidth=1)
        label = _p_to_label(p)
        ax.text((x1 + x2) / 2.0, y_coord + pad * 0.1, label, ha="center", va="bottom")
        cur += 1

    # adjust ylim to make room
    ax.set_ylim(y0, top + pad * (cur + 2))
    return None
