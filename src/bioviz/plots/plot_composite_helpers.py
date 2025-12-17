"""Feature-complete helpers for composite plots.

This module provides helpers used by `grouped.py` and `waterfall.py` with
behavior closer to the original implementations in tm_toolbox. It prefers
to use optional dependencies when available (`statannotations`) but falls
back to conservative implementations otherwise.
"""

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def get_default_palette(n: int, palette_name: str = "tab10") -> list[str]:
    """Return a list of hex color strings of length `n` using seaborn palettes.

    The function ensures a reasonable minimum number of colors and mirrors
    the tab-based palettes commonly used in prior code.
    """
    pal = sns.color_palette(palette_name, n_colors=max(3, n))
    from matplotlib import colors as mcolors

    return [mcolors.to_hex(c) for c in pal[:n]]


def compute_mwu_and_annot_pairs(
    df: pd.DataFrame,
    x: str,
    y: str,
    pairs: List[Tuple[str, str]],
    alternative: str = "two-sided",
    nan_policy: str = "omit",
) -> List[Tuple[Tuple[str, str], float, Optional[float]]]:
    """Compute Mann-Whitney U p-values and Cliff's delta effect sizes for pairs.

    Returns a list of tuples: ((a, b), pvalue, cliffs_delta)
    cliffs_delta is None on failure.
    """
    results: List[Tuple[Tuple[str, str], float, Optional[float]]] = []
    for a, b in pairs:
        va = df.loc[df[x] == a, y].dropna()
        vb = df.loc[df[x] == b, y].dropna()
        try:
            stat, p = stats.mannwhitneyu(va, vb, alternative=alternative)
        except Exception:
            p = float("nan")
        # Compute Cliff's delta (non-parametric effect size)
        try:
            # delta = (2 * U) / (n1*n2) - 1  where U is the Mann-Whitney U statistic
            n1 = len(va)
            n2 = len(vb)
            if n1 > 0 and n2 > 0:
                U = stat
                cliffs = (2.0 * U) / (n1 * n2) - 1.0
            else:
                cliffs = None
        except Exception:
            cliffs = None
        results.append(((a, b), p, cliffs))
    return results


def apply_statannotations(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    pairs: List[Tuple[str, str]],
    use_bh: bool = True,
    test: str = "Mann-Whitney",
    text_format: str = "star",
    loc: str = "inside",
    verbose: bool = False,
) -> None:
    """Annotate `ax` with statistical comparisons for `pairs`.

    - Attempts to use `statannotations`'s `Annotator` if installed (preferred).
    - Otherwise computes p-values, applies Benjamini-Hochberg correction
      if requested, and draws simple lines with star annotations.
    """
    # First try statannotations (preferred for nicer visuals)
    try:
        from statannotations.Annotator import Annotator

        annot = Annotator(ax, pairs, data=df, x=x, y=y)
        annot.configure(test=test, text_format=text_format, loc=loc)
        annot.apply_and_annotate()
        if verbose:
            print("Applied statannotations Annotator")
        return
    except Exception:
        if verbose:
            print("statannotations not available or failed; using fallback annotations")

    # Fallback path: compute p-values and draw annotations manually
    raw = compute_mwu_and_annot_pairs(df, x, y, pairs)
    pairs_only = [r[0] for r in raw]
    pvals = [r[1] for r in raw]

    # BH correction when requested
    if use_bh and len(pvals) > 0:
        try:
            mask = [not (p is None or (isinstance(p, float) and np.isnan(p))) for p in pvals]
            valid_pvals = [p for p, m in zip(pvals, mask) if m]
            if valid_pvals:
                rej, p_adj, _, _ = multipletests(valid_pvals, method="fdr_bh")
                # map back
                p_adj_full = []
                iter_adj = iter(p_adj)
                for m in mask:
                    if m:
                        p_adj_full.append(next(iter_adj))
                    else:
                        p_adj_full.append(float("nan"))
            else:
                p_adj_full = [float("nan")] * len(pvals)
        except Exception:
            p_adj_full = pvals
    else:
        p_adj_full = pvals

    # Simple drawing: determine x positions in the axis coordinate system
    cats = list(pd.Categorical(df[x]).categories)
    x_positions = {cat: i for i, cat in enumerate(cats)}
    y_std = float(df[y].std(skipna=True) or 1.0)
    offset = y_std * 0.1
    # Stack multiple annotations to avoid overlap
    stack_counts = {}
    for idx, ((a, b), raw_p, adj_p) in enumerate(zip(pairs_only, pvals, p_adj_full)):
        xa = x_positions.get(a, 0)
        xb = x_positions.get(b, 0)
        left = min(xa, xb)
        right = max(xa, xb)
        key = (left, right)
        stack_counts[key] = stack_counts.get(key, 0) + 1
        level = stack_counts[key]
        # compute y position above the higher group's max
        ya = (
            df.loc[df[x] == a, y].max(skipna=True)
            if not df.loc[df[x] == a, y].dropna().empty
            else 0
        )
        yb = (
            df.loc[df[x] == b, y].max(skipna=True)
            if not df.loc[df[x] == b, y].dropna().empty
            else 0
        )
        yline = max(ya, yb) + offset * level
        # Draw bracket
        ax.plot(
            [left, left, right, right],
            [yline - offset * 0.1, yline, yline, yline - offset * 0.1],
            color="k",
        )
        # Determine label (use adjusted p if available)
        p_for_label = (
            adj_p
            if (adj_p is not None and not (isinstance(adj_p, float) and np.isnan(adj_p)))
            else raw_p
        )
        label = "n.s."
        try:
            if p_for_label is not None and not (
                isinstance(p_for_label, float) and np.isnan(p_for_label)
            ):
                if p_for_label < 0.001:
                    label = "***"
                elif p_for_label < 0.01:
                    label = "**"
                elif p_for_label < 0.05:
                    label = "*"
                else:
                    label = f"p={p_for_label:.3g}"
        except Exception:
            label = "n.s."
        ax.text((left + right) / 2.0, yline + offset * 0.02, label, ha="center", va="bottom")
