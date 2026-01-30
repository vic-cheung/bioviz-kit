Usage Guide
===========

This guide covers the main plot types available in bioviz-kit with working examples
drawn from the actual codebase.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install bioviz-kit

Or install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/bioviz-kit.git
   cd bioviz-kit
   pip install -e .


Core Concepts
-------------

bioviz-kit follows a consistent pattern for all plot types:

1. **Config classes** - Pydantic models that define all plot parameters
2. **Plotter classes** - Take data + config, produce matplotlib figures

This design provides:

- Type safety and validation via Pydantic
- IDE autocompletion for all parameters
- Sensible defaults for publication-ready output
- Easy serialization/deserialization of configurations


Kaplan-Meier Survival Plots
---------------------------

KM plots are commonly used in clinical trials to visualize time-to-event data.

.. code-block:: python

   import pandas as pd
   from bioviz.configs import KMPlotConfig
   from bioviz.plots import KMPlotter

   # Prepare your survival data
   df = pd.DataFrame({
       "time": [5, 10, 15, 8, 12, 20],
       "event": [1, 0, 1, 1, 0, 1],
       "arm": ["Treatment", "Treatment", "Treatment",
               "Control", "Control", "Control"],
   })

   # Configure the plot
   config = KMPlotConfig(
       time_col="time",
       event_col="event",
       group_col="arm",
       title="Overall Survival",
       show_risktable=True,
       show_pvalue=True,
   )

   # Generate
   plotter = KMPlotter(df, config)
   fig, ax, pval = plotter.plot()


Key configuration options:

- ``show_risktable`` - Display number at risk below the plot
- ``show_pvalue`` - Show log-rank test p-value
- ``show_ci`` - Display confidence intervals
- ``legend_loc`` - Legend position: "bottom", "right", or "inside"
- ``color_dict`` - Map group names to colors


Volcano Plots
-------------

Volcano plots display statistical significance vs effect size, commonly used
for differential expression analysis.

.. code-block:: python

   import pandas as pd
   from bioviz.configs import VolcanoConfig
   from bioviz.plots import VolcanoPlotter

   df = pd.DataFrame({
       "label": ["A", "B", "C", "D", "E"],
       "log2_or": [3.2, -2.5, 0.5, -4.2, 1.1],
       "p_adj": [0.01, 0.04, 0.5, 0.001, 0.2],
   })

   config = VolcanoConfig(
       x_col="log2_or",
       y_col="p_adj",
       label_col="label",
       log_transform_ycol=True,        # -log10 transform p-values
       label_mode="sig_and_thresh",    # label points meeting both thresholds
       color_mode="sig_and_thresh",    # color points meeting both thresholds
       y_col_thresh=0.05,              # significance threshold
       abs_x_thresh=2.0,               # effect size threshold (|x| >= 2)
   )

   plotter = VolcanoPlotter(df, config)
   fig, ax = plotter.plot()
   plotter.save("volcano.png")


Key configuration options:

- ``label_mode`` - Controls which points get labeled: "auto", "sig", "sig_and_thresh", "thresh", "sig_or_thresh", "all"
- ``color_mode`` - Controls which points get colored (same options)
- ``force_label_side_by_point_sign`` - Push labels outward based on point sign
- ``use_adjust_text`` - Enable adjustText library for label placement


Oncoplots
---------

Oncoplots (mutation landscapes) show the mutation status of genes across samples.
The plotter requires column mappings (``x_col``, ``y_col``, ``value_col``, ``row_group_col``)
plus optional row_groups DataFrame and top annotations.

.. code-block:: python

   import pandas as pd
   from bioviz.configs import (
       OncoplotConfig,
       HeatmapAnnotationConfig,
       TopAnnotationConfig,
   )
   from bioviz.plots import OncoPlotter

   # Main mutation data
   df = pd.DataFrame({
       "Patient_ID": ["p1", "p1", "p2", "p2", "p1", "p2", "p2"],
       "Gene": ["TP53", "KRAS", "KRAS", "TP53", "PIK3CA", "PIK3CA", "PIK3CA"],
       "Variant_type": ["SNV", "SNV", "CNV", "Fusion", "SNV", "SNV", "CNV"],
       "Cohort": ["A", "A", "B", "B", "A", "B", "B"],
       "Dose": ["100 mg", "100 mg", "200 mg", "200 mg", "100 mg", "200 mg", "200 mg"],
   })

   # Row groups (pathway annotations) - index matches y_col values
   row_groups = pd.DataFrame({
       "Pathway": {
           "TP53": "Tumor Suppressor",
           "KRAS": "RAS Signaling",
           "PIK3CA": "PI3K/AKT",
       }
   }).rename_axis("Gene")

   # Pathway bar colors
   row_groups_color_dict = {
       "Tumor Suppressor": "#000000",
       "RAS Signaling": "#000000",
       "PI3K/AKT": "#000000",
   }

   # Top annotation series (must be indexed by x_col values)
   cohort_series = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Cohort"]
   dose_series = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["Dose"]

   # Mutation type colors
   colors = {"SNV": "#1f77b4", "CNV": "#ff7f0e", "Fusion": "#2ca02c"}

   # Configure top annotations
   top_ann = TopAnnotationConfig(
       values=cohort_series,
       colors={"A": "#003975", "B": "#9d0ca2"},
       legend_title="Cohort",
       legend_value_order=["A", "B"],
       merge_labels=False,
       show_category_labels=False,
   )

   dose_ann = TopAnnotationConfig(
       values=dose_series,
       colors={"100 mg": "#007352", "200 mg": "#860F0F"},
       legend_title="Dose",
       legend_value_order=["100 mg", "200 mg"],
       merge_labels=False,
       show_category_labels=False,
   )

   # Configure heatmap cell rendering
   heat_ann = HeatmapAnnotationConfig(
       values="Variant_type",  # column name or pd.Series
       colors=colors,
       bottom_left_triangle_values=["SNV"],   # render as bottom-left triangles
       upper_right_triangle_values=["CNV"],   # render as top-right triangles
       legend_title="Mutation Type",
       legend_value_order=["SNV", "CNV", "Fusion"],
   )

   # Main config with column mappings
   onc_cfg = OncoplotConfig(
       x_col="Patient_ID",
       y_col="Gene",
       value_col="Variant_type",
       row_group_col="Pathway",
       heatmap_annotation=heat_ann,
       top_annotations={"Cohort": top_ann, "Dose": dose_ann},
       top_annotation_order=["Cohort", "Dose"],
       legend_category_order=["Dose", "Cohort", "Mutation Type"],
   )

   # Create plotter with data, config, and row groups
   plotter = OncoPlotter(
       df,
       onc_cfg,
       row_groups=row_groups,
       row_groups_color_dict=row_groups_color_dict,
   )
   fig = plotter.plot()
   fig.savefig("oncoplot.pdf", bbox_inches="tight", pad_inches=0.1)


Key configuration options:

- ``x_col``, ``y_col``, ``value_col``, ``row_group_col`` - Required column mappings
- ``heatmap_annotation`` - Controls cell rendering (colors, triangles)
- ``top_annotations`` - Dict of annotation name → TopAnnotationConfig
- ``col_split_by`` / ``col_split_order`` - Split columns by a categorical variable
- ``row_group_order`` - Custom ordering for pathway/row-group bars


Forest Plots
------------

Forest plots visualize hazard ratios with confidence intervals from survival analysis.

.. code-block:: python

   import pandas as pd
   from bioviz.configs import ForestPlotConfig
   from bioviz.plots import ForestPlotter

   df = pd.DataFrame({
       "comparator": ["Age ≥65", "Age <65", "Male", "Female"],
       "hr": [1.2, 0.85, 1.1, 0.9],
       "ci_lower": [0.9, 0.6, 0.8, 0.65],
       "ci_upper": [1.6, 1.2, 1.5, 1.25],
       "p_value": [0.21, 0.35, 0.42, 0.48],
       "reference": ["<65", "<65", "Female", "Male"],
   })

   config = ForestPlotConfig(
       hr_col="hr",
       ci_lower_col="ci_lower",
       ci_upper_col="ci_upper",
       label_col="comparator",
       pvalue_col="p_value",
       reference_col="reference",
       show_reference_line=True,
       show_stats_table=True,
       xlabel="Hazard Ratio (95% CI)",
       log_scale=True,  # Standard for HR visualization
   )

   plotter = ForestPlotter(df, config)
   fig, ax = plotter.plot()


Key configuration options:

- ``log_scale`` - Use log scale for x-axis (standard for HR plots)
- ``show_stats_table`` - Show HR/CI/p-value table on right side
- ``show_reference_line`` - Vertical line at HR=1
- ``color_significant`` / ``color_nonsignificant`` - Colors by p-value
- ``variable_col`` - Group rows by variable for multi-section plots


Grouped Bar Charts
------------------

Grouped bar charts with automatic confidence interval calculation.

.. code-block:: python

   import pandas as pd
   from bioviz.configs import GroupedBarConfig
   from bioviz.plots import GroupedBarPlotter

   # Pre-computed data with CIs
   df = pd.DataFrame({
       "Category": ["Gene A", "Gene A", "Gene B", "Gene B"],
       "Group": ["Baseline", "Progression", "Baseline", "Progression"],
       "value": [0.15, 0.35, 0.20, 0.45],
       "ci_low": [0.08, 0.25, 0.12, 0.35],
       "ci_high": [0.25, 0.48, 0.32, 0.58],
   })

   config = GroupedBarConfig(
       category_col="Category",
       group_col="Group",
       value_col="value",
       ci_low_col="ci_low",
       ci_high_col="ci_high",
       orientation="horizontal",  # or "vertical"
   )

   plotter = GroupedBarPlotter(df, config)
   fig, ax = plotter.plot()


For proportion data with automatic CI computation:

.. code-block:: python

   config = GroupedBarConfig(
       category_col="Category",
       group_col="Group",
       value_col="value",
       k_col="count",      # numerator column
       n_col="total",      # denominator column
       ci_method="clopper-pearson",  # or "bootstrap"
   )


Key configuration options:

- ``orientation`` - "horizontal" (barh) or "vertical" (bar)
- ``ci_method`` - "clopper-pearson", "bootstrap", or "none"
- ``k_col`` / ``n_col`` - Columns for proportion CI computation
- ``alpha`` - Significance level for CI (0.05 = 95% CI)


Styling and Themes
------------------

Font sizes in bioviz-kit default to ``None``, which means they inherit from
matplotlib's rcParams. This makes it easy to apply global themes:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set global style
   plt.rcParams.update({
       "font.size": 12,
       "axes.labelsize": 14,
       "axes.titlesize": 16,
       "legend.fontsize": 11,
   })

   # All bioviz plots will now use these sizes


Saving Figures
--------------

Figures can be saved directly via the plotter or manually:

.. code-block:: python

   # Some plotters have a save() method
   plotter.save("figure.pdf")

   # Or save manually with custom settings
   fig.savefig("figure.pdf", dpi=300, bbox_inches="tight")


Examples
--------

See the ``examples/`` directory for complete, runnable examples:

- ``km_survival_example.py`` - Kaplan-Meier survival analysis with multiple variants
- ``volcano_smoke.py`` - Volcano plot variations (forced labels, adjust_text)
- ``oncoplot_example.py`` - Detailed oncoplot with pathway bars and top annotations
- ``minimal_bioviz_smoke.py`` - Line plots, tables, and oncoplots in one file
- ``distribution_examples.py`` - Histogram + boxplot combinations
