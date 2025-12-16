bioviz-kit
==========

Framework-agnostic visualization library for publication-ready clinical and biological data plots.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Publication-ready styling** – Clean, professional visualizations out of the box
- **Framework-agnostic** – Works with any data pipeline or analysis framework
- **Customizable configurations** – Extensive theming and layout options
- **Clinical & bioinformatics focused** – Specialized plot types for common analyses

## Installation

bioviz-kit
==========

Framework-agnostic visualization library for publication-ready clinical and biological data plots.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Publication-ready styling** – Clean, professional visualizations out of the box
- **Framework-agnostic** – Works with any data pipeline or analysis framework
- **Customizable configurations** – Extensive theming and layout options
- **Clinical & bioinformatics focused** – Specialized plot types for common analyses

## Installation

Install from PyPI:

```bash
pip install bioviz-kit
```

Or install in development mode:

```bash
pip install -e .
```

## Requirements

- Python 3.11+
- pandas
- matplotlib
- seaborn
- adjustText

## Usage

```python
from bioviz import lineplot, oncoplot

# Generate styled plots with minimal configuration
fig = lineplot.generate_styled_lineplot(df, config)
fig = oncoplot.generate_styled_oncoplot(df, config)
```

See [examples/](examples/) for complete usage examples.

## Examples

- Example files live in the `examples/` directory. Recommended practices:
    - Keep runnable `.py` scripts for quick CLI usage and reproducible examples.
    - Provide companion `.ipynb` notebooks for narrative tutorials and figures. Convert with `jupytext` or `nbconvert` if needed:

```bash
pip install jupytext
jupytext --to notebook examples/my_example.py -o examples/my_example.ipynb
```

    - Name files with short, descriptive snake_case (optionally numeric prefixes for ordered tutorials, e.g. `01_quickstart.ipynb`).
    - Add a short header comment or top-level README in `examples/` describing each example's purpose and required inputs.

## Documentation and ReadTheDocs

TBD

## Design Philosophy

- Lightweight `DefaultStyle` provided; easily extended with custom themes


## Licensing

- `bioviz-kit` is released under the MIT License © 2025 Victoria Cheung.

### Thanks

This package was spun out of internal tooling I wrote at Revolution Medicines.
Many thanks to the team there for allowing the code to be open sourced.
