bioviz
======

A small, framework-agnostic plotting utilities package for plotting common
clinical bioinformatics visualizations with a clean, publication-ready style.

Installation
------------

Install from PyPI using the package name `bioviz-kit` and import as `bioviz`:

```bash
pip install bioviz-kit
python -c "import bioviz; print(bioviz)"
```

During development:

```bash
python -m pip install -e .
```

Requirements
------------

This package depends on common plotting libraries which will be installed by pip:

- `pandas`
- `matplotlib`
- `seaborn`
- `adjustText`

Usage
-----

Import top-level plotting modules:

```python
from bioviz import lineplot, oncoplot

# Use the functions provided by the modules, e.g.:
# fig = lineplot.generate_styled_lineplot(df, config)
```

Design Notes
------------

- The package is intentionally agnostic to any company-specific styling. A lightweight
	`DefaultStyle` is provided.


Licensing
---------

- `bioviz-kit` is released under the MIT License © 2025 Victoria Cheung.

### Thanks

This package was spun out of internal tooling at Revolution Medicines.
Many thanks to the team there for allowing the code to be open sourced.
