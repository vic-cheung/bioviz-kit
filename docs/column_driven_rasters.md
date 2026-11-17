# Column-driven raster rows and events

`OncoPrevalenceRasterPlotter` uses exact `OncoplotConfig.y_col` values as rows.
It never splits or collapses names using punctuation or biological categories.
Provide a prepared row column to control which records share a row. For example,
`EGFR` records share a row while `EML4::ALK`, `A/B`, and `A;B` each retain their
complete, distinct row labels. Assign each row to only one row group.

Each patient has one horizontal slot. In `event_band_mode="per_sample"`, only
that patient's observed events on that row divide its height. Repeated records
are deduplicated; overall prevalence counts unique patients, not event types.
The existing default `"global"` mode reserves all configured event positions.

## Switch categories by passing a column

The caller owns event preparation and semantic colors:

```python
# Source-specific normalization belongs in the caller, not the renderer.
frame["detailed_event"] = frame["subtype"].fillna(frame["event"])

config = OncoplotConfig(
    x_col="patient",
    y_col="display_row",
    value_col="event",             # e.g. SV/CNV/FUSION
    row_values_color_dict=colors,
)
fig = OncoPrevalenceRasterPlotter(frame, config, event_band_mode="per_sample").plot()

# The same rows and patient denominators, with finer event categories:
config = config.model_copy(update={"value_col": "detailed_event"})
fig = OncoPrevalenceRasterPlotter(frame, config, event_band_mode="per_sample").plot()
```

If supplying `heatmap_annotation`, its `values` field selects the event column
and takes precedence over `value_col`. Update both together when switching a
config that already contains a heatmap annotation. No exclusion list is needed:
records with a distinct row value cannot divide another row's patient slot.

## Optional nested subtype bands

Keep the parent event column and pass the subtype column and a caller-defined
band order. The renderer contains no gene, fusion, CNV, Amp, or Del rules.

```python
fig = OncoPrevalenceRasterPlotter(
    frame, config, event_band_mode="per_sample",
    event_subtype_col="subtype",
    event_subtype_bands={"CNV": ["Amp", "Del"]},
    event_subtype_colors={"Amp": amp_color, "Del": del_color},
).plot()
```

Restore `config.value_col="event"` before this nested example if using the
previous detailed-event config. Bands run top to bottom inside the parent's
allocated region. Amp only colors the upper half; Del only colors the lower
half; both color both. For SV plus CNV, SV gets half the row and each CNV subtype
gets a quarter. Other rows with only one unsplit event fill the height.

Missing subtype values retain the generic parent color. Known subtype bands
are overlaid if both known and missing evidence exists. Nonmissing values absent
from the configured band order raise an error instead of silently disappearing.
Subtype colors and top/bottom positions appear in the legend. Main-event counts
remain deduplicated by patient, row, and parent even with multiple subtypes.
Both global and per-sample outer band modes support nested subtypes.
