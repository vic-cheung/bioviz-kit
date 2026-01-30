bioviz-kit Documentation
========================

**bioviz-kit** is a framework-agnostic visualization library for publication-ready
clinical and biological data plots.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+


Features
--------

- **Publication-ready styling** - Clean, professional visualizations out of the box
- **Framework-agnostic** - Works with any data pipeline or analysis framework
- **Pydantic configurations** - Type-safe, validated configuration objects
- **Clinical & bioinformatics focused** - Specialized plot types:

  - Kaplan-Meier survival curves with risk tables
  - Volcano plots for differential expression/enrichment
  - Oncoplots (mutation landscapes)
  - Forest plots for hazard ratios
  - Waterfall plots for tumor response
  - Grouped bar charts with confidence intervals
  - Distribution plots (histogram + boxplot)
  - Styled tables


Quick Start
-----------

.. code-block:: python

   from bioviz.configs import KMPlotConfig
   from bioviz.plots import KMPlotter

   config = KMPlotConfig(
       time_col="time",
       event_col="event",
       group_col="arm",
       title="Overall Survival",
       show_risktable=True,
   )

   plotter = KMPlotter(df, config)
   fig, ax, pval = plotter.plot()


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
