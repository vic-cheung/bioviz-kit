"""Grouped plotting utilities (grouped boxplots with stat annotations).

This module provides `plot_grouped_boxplots` which mirrors the behaviour
used in PRG requests, including Mann-Whitney tests and BH correction.
"""

from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from .plot_composite_helpers import (
    get_default_palette,
    compute_mwu_and_annot_pairs,
    apply_statannotations,
)


def plot_grouped_boxplots(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[List[str]] = None,
    palette: Optional[Iterable[str]] = None,
    stat_pairs: Optional[List[Tuple[str, str]]] = None,
    ax=None,
    figsize: Optional[Tuple[float, float]] = (8, 6),
    show: bool = True,
    use_bh_correction: bool = True,
    pval_alpha: float = 0.05,
    show_pvalues: bool = False,
    strip_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw grouped boxplots with optional statistical annotations.

    Enhancements over the basic version:
    - Optional Benjamini-Hochberg correction of p-values computed by Mann-Whitney U tests
    - Optional textual p-value annotation above compared groups
    - Cleaner legend handling when `hue` is used (avoid duplicate legends)

    Returns (fig, ax) when `ax` is None, otherwise returns (ax.figure, ax).
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        try:
            # default to white face with transparent background
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.0)
        except Exception:
            pass
        created_fig = True
    else:
        fig = ax.figure

    if order is None:
        order = list(df[x].dropna().unique())

    if palette is None:
        palette = get_default_palette(len(order))

    # Draw boxplot
    sns.boxplot(data=df, x=x, y=y, hue=hue, order=order, palette=palette, ax=ax)

    # Default stripplot kwargs
    if strip_kwargs is None:
        strip_kwargs = dict(dodge=True, color="k", size=3, alpha=0.8)
    sns.stripplot(data=df, x=x, y=y, hue=hue, order=order, ax=ax, **strip_kwargs)

    # Remove duplicate legend entries if hue is present
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        # legend shows twice due to boxplot + stripplot; keep unique labels preserving order
        seen = set()
        new_handles = []
        new_labels = []
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen.add(lbl)
                new_handles.append(h)
                new_labels.append(lbl)
        ax.legend(new_handles, new_labels, title=hue)

    # Statistical annotations: compute p-values, optionally correct, then annotate
    if stat_pairs:
        p_results = compute_mwu_and_annot_pairs(df, x, y, stat_pairs)
        pairs = [pr[0] for pr in p_results]
        pvals = [pr[1] for pr in p_results]

        # Apply BH correction if requested
        try:
            if use_bh_correction and len(pvals) > 0:
                valid_idx = [i for i, pv in enumerate(pvals) if not np.isnan(pv)]
                # multipletests requires array of pvals without nans; handle mapping
                pv_for_correction = [pvals[i] for i in valid_idx]
                if len(pv_for_correction) > 0:
                    _, p_adj, _, _ = multipletests(pv_for_correction, method="fdr_bh")
                    # map adjusted back
                    p_adj_full = [np.nan] * len(pvals)
                    for j, idx in enumerate(valid_idx):
                        p_adj_full[idx] = p_adj[j]
                else:
                    p_adj_full = [np.nan] * len(pvals)
            else:
                p_adj_full = pvals
        except Exception:
            p_adj_full = pvals

        # Use statannotations to draw significance bars/stars (best-effort)
        try:
            apply_statannotations(ax, df, x, y, pairs)
        except Exception:
            # fallback: our helper will either use statannotations or draw simple annotations
            try:
                apply_statannotations(ax, df, x, y, pairs)
            except Exception:
                pass

        # Optionally display adjusted p-values as text labels above the pair
        if show_pvalues:
            # Determine x positions
            x_positions = {cat: i for i, cat in enumerate(order)}
            # If hue is present, annotation placement is trickier; place above max of the pair
            offset = (df[y].std(skipna=True) or 1.0) * 0.1
            # count stacked annotations to avoid overlap
            stack_counts = {}
            for idx, ((a, b), pv) in enumerate(zip(pairs, p_adj_full)):
                try:
                    # compute y coordinate above the higher group
                    ya = df.loc[df[x] == a, y].dropna()
                    yb = df.loc[df[x] == b, y].dropna()
                    ytop = max(ya.max() if not ya.empty else 0, yb.max() if not yb.empty else 0)
                    key = (
                        min(x_positions.get(a, 0), x_positions.get(b, 0)),
                        max(x_positions.get(a, 0), x_positions.get(b, 0)),
                    )
                    stack_counts[key] = stack_counts.get(key, 0) + 1
                    ytext = ytop + offset * stack_counts[key]
                    label = "n.s."
                    if pv is not None and not np.isnan(pv):
                        label = f"p={pv:.3g}"
                    ax.text(
                        (x_positions.get(a, 0) + x_positions.get(b, 0)) / 2.0,
                        ytext,
                        label,
                        ha="center",
                        va="bottom",
                    )
                except Exception:
                    continue

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if created_fig and show:
        plt.show()
    return fig, ax
