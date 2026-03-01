# pygui

Interactive tool for analyzing and transforming high-frequency time series data (10M+ points) with GPU acceleration support.

## Features

- **Data Loading**: Load Parquet files with automatic datetime parsing
- **Data Transformations**: Filter, resample, rolling windows, outlier detection, missing value imputation, differencing, lags, custom aggregation
- **Descriptive Statistics**: Basic stats, time-based aggregation, correlation matrix, outlier statistics, distributions
- **Interactive Dashboard**: Panel-based UI with time series plots, histograms, and statistics panels
- **GPU Support**: CUDA detection and GPU acceleration via cuDF (RAPIDS)
- **Export**: Save plots as images and transformed data as Parquet

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

```bash
# Install uv (if not already installed)
pip install uv

# Clone the repository
git clone <repository-url>
cd pygui

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

### Optional: CUDA/GPU Support

For GPU acceleration with cuDF:

```bash
# Install all GPU dependencies at once
uv sync --extra gpu

# Or install specific CUDA dependencies individually
uv add --extra gpu cupy-cuda12x  # For CUDA 12.x
uv add --extra gpu cudf-cu12        # For CUDA 12.x
```

## Usage

### HDF5 to Parquet Conversion

If you have HDF5 files, convert them to Parquet format first:

```bash
uv run python scripts/convert_hdf5_to_parquet.py your_file.h5 --output your_file.parquet
```

For HDF5 files with multiple tables, specify the table key:

```bash
uv run python scripts/convert_hdf5_to_parquet.py your_file.h5 --table-key table_name --output your_file.parquet
```

### Launching the Dashboard

```bash
uv run pygui
```

Or directly:

```bash
uv run python -m pygui.app
```

The dashboard will open in your default browser.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
# Check formatting
uv run ruff check .

# Fix formatting issues
uv run ruff check --fix .
```

## Project Structure

```
pygui/
├── src/pygui/
│   ├── app.py                 # Entry point
│   ├── utils/
│   │   └── cuda.py           # CUDA detection
│   ├── data/
│   │   ├── loader.py         # Parquet loading
│   │   └── transformer.py    # Data transformations
│   ├── analysis/
│   │   └── statistics.py     # Descriptive statistics
│   └── viz/
│       └── dashboard.py      # Panel UI
├── tests/                     # Test suite
├── scripts/
│   └── convert_hdf5_to_parquet.py  # HDF5 converter
└── data/                      # Sample data directory
```

## License

[Specify your license here]
