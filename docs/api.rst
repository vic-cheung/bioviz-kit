API Reference
=============

This section documents the public API of bioviz-kit.


Configuration Classes
---------------------

All configuration classes are Pydantic models that validate inputs and provide
sensible defaults for publication-ready output.

.. automodule:: bioviz.configs
   :members:
   :undoc-members:
   :show-inheritance:


KMPlotConfig
~~~~~~~~~~~~

.. autoclass:: bioviz.configs.KMPlotConfig
   :members:
   :undoc-members:
   :show-inheritance:


VolcanoConfig
~~~~~~~~~~~~~

.. autoclass:: bioviz.configs.VolcanoConfig
   :members:
   :undoc-members:
   :show-inheritance:


OncoplotConfig
~~~~~~~~~~~~~~

.. autoclass:: bioviz.configs.OncoplotConfig
   :members:
   :undoc-members:
   :show-inheritance:


ForestPlotConfig
~~~~~~~~~~~~~~~~

.. autoclass:: bioviz.configs.ForestPlotConfig
   :members:
   :undoc-members:
   :show-inheritance:


GroupedBarConfig
~~~~~~~~~~~~~~~~

.. autoclass:: bioviz.configs.GroupedBarConfig
   :members:
   :undoc-members:
   :show-inheritance:


Plotter Classes
---------------

Plotters take a DataFrame and a configuration object, then produce matplotlib figures.

KMPlotter
~~~~~~~~~

.. autoclass:: bioviz.plots.KMPlotter
   :members:
   :undoc-members:
   :show-inheritance:


VolcanoPlotter
~~~~~~~~~~~~~~

.. autoclass:: bioviz.plots.VolcanoPlotter
   :members:
   :undoc-members:
   :show-inheritance:


OncoPlotter
~~~~~~~~~~~

.. autoclass:: bioviz.plots.OncoPlotter
   :members:
   :undoc-members:
   :show-inheritance:


ForestPlotter
~~~~~~~~~~~~~

.. autoclass:: bioviz.plots.ForestPlotter
   :members:
   :undoc-members:
   :show-inheritance:


GroupedBarPlotter
~~~~~~~~~~~~~~~~~

.. autoclass:: bioviz.plots.GroupedBarPlotter
   :members:
   :undoc-members:
   :show-inheritance:


TablePlotter
~~~~~~~~~~~~

.. autoclass:: bioviz.plots.TablePlotter
   :members:
   :undoc-members:
   :show-inheritance:


Utility Functions
-----------------

.. automodule:: bioviz.plots.grouped_bar
   :members: clopper_pearson_ci, compute_proportion_summary
   :undoc-members:


.. automodule:: bioviz.plots.km
   :members: format_pvalue, add_pvalue_annotation
   :undoc-members:
