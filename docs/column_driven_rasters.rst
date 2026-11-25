Column-driven prevalence rasters
================================

:class:`~bioviz.plots.OncoPrevalenceRasterPlotter` uses the exact values of
``OncoplotConfig.y_col`` as row labels. Prepare that column to control which
records share a row. For example, ``EGFR`` records share one row and
``EML4::ALK``, ``A/B``, and ``A;B`` each retain their full label. If you supply
row groups, assign each row to one group.

Each cohort column has one horizontal slot per patient. By default the plot
shows all patients. Use ``group_by`` to derive cohorts from data columns or
pass :class:`~bioviz.plots.ColumnSpec` objects through ``column_specs`` to
define explicit cohorts, including overlapping ones. Each column uses its
own patient denominator.

In the default ``event_band_mode="global"``, every slot reserves the same
positions for event categories. In ``"per_sample"`` mode, the events observed
for a patient on a particular row divide that slot's height. Repeated records
count once per patient, row, and event. The prevalence numerator counts each
patient with at least one event once per row.

Switch event categories by choosing a column
--------------------------------------------

This example uses a placeholder for patient P4 to keep that patient in the
denominator. The EGFR row shows 75% prevalence in both figures because three
of four patients have an EGFR event.

.. code-block:: python

   import pandas as pd

   from bioviz.configs import OncoplotConfig
   from bioviz.plots import OncoPrevalenceRasterPlotter

   frame = pd.DataFrame(
       [
           ("P1", "EGFR", "SV", None),
           ("P1", "EGFR", "CNV", "Amp"),
           ("P1", "EGFR", "CNV", "Del"),
           ("P2", "EGFR", "CNV", "Amp"),
           ("P3", "EGFR", "CNV", "Del"),
           ("P1", "EML4::ALK", "Fusion", None),
           ("P4", "EGFR", None, None),
       ],
       columns=["patient", "display_row", "event", "subtype"],
   )
   frame["detailed_event"] = frame["subtype"].fillna(frame["event"])

   colors = {
       "SV": "#d62728",
       "CNV": "#1f77b4",
       "Fusion": "#ff7f0e",
       "Amp": "#9467bd",
       "Del": "#2ca02c",
   }
   parent_config = OncoplotConfig(
       x_col="patient",
       y_col="display_row",
       value_col="event",
       row_values_color_dict=colors,
   )
   parent_fig = OncoPrevalenceRasterPlotter(
       frame, parent_config, event_band_mode="per_sample"
   ).plot()

   detailed_config = parent_config.model_copy(update={"value_col": "detailed_event"})
   detailed_fig = OncoPrevalenceRasterPlotter(
       frame, detailed_config, event_band_mode="per_sample"
   ).plot()

When a heatmap annotation supplies an event column in ``values``, that column
takes precedence over ``value_col``. Keep both fields aligned when changing
event columns on a config that already has a heatmap annotation.

Nest subtype bands within a parent event
----------------------------------------

Reuse ``parent_config`` to keep ``CNV`` as the parent event. Supply a subtype
column, a top-to-bottom band order for each parent you want to subdivide,
and a color for each subtype.

.. code-block:: python

   nested_fig = OncoPrevalenceRasterPlotter(
       frame,
       parent_config,
       event_band_mode="per_sample",
       event_subtype_col="subtype",
       event_subtype_bands={"CNV": ["Amp", "Del"]},
       event_subtype_colors={"Amp": colors["Amp"], "Del": colors["Del"]},
   ).plot()
   nested_fig.savefig("prevalence_raster.pdf", bbox_inches="tight")

For P1, ``SV`` and ``CNV`` each get half the EGFR slot. ``Amp`` and ``Del``
each fill a quarter of the slot within the CNV half. For P2 and P3, CNV gets
the full slot. ``Amp`` fills the top half for P2 and ``Del`` fills the bottom
half for P3. P4 has an empty EGFR slot. The prevalence label remains
``75% (3/4)``.

Missing subtype values use the parent color. When both missing and known
subtypes occur in one slot, the known bands overlay the parent color. A
nonmissing subtype outside the configured band order raises an error. The
legend shows subtype colors and positions. Nested bands work in both
``"global"`` and ``"per_sample"`` modes.
