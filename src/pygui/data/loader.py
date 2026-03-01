"""Data loading utilities for Parquet files."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_parquet(filepath: str) -> pd.DataFrame:
    """Load Parquet file and return DataFrame with datetime index.

    Args:
        filepath: Path to Parquet file

    Returns:
        DataFrame with datetime index

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If no datetime column found
    """
    filepath_obj = Path(filepath)

    if not filepath_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    df = pd.read_parquet(filepath, engine="pyarrow")

    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty: {filepath}")

    datetime_col = _detect_datetime_column(df)

    if datetime_col:
        df = df.set_index(datetime_col)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(
            "No datetime column found in Parquet file. "
            "Please ensure one column contains datetime data."
        )

    return df


def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Detect the datetime column in a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Name of datetime column or None
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

        if pd.api.types.is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col].head(100))
                if pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df[col].head(100))):
                    return col
            except (ValueError, TypeError):
                continue

    return None


def get_data_info(df: pd.DataFrame) -> dict:
    """Get information about the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with data information
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    if isinstance(df.index, pd.DatetimeIndex):
        info["time_range"] = {
            "start": df.index.min(),
            "end": df.index.max(),
            "duration": df.index.max() - df.index.min(),
            "frequency": df.index.freq,
        }

    return info


def get_parquet_schema(filepath: str) -> pa.Schema:
    """Get the schema of a Parquet file without loading the data.

    Args:
        filepath: Path to Parquet file

    Returns:
        PyArrow schema
    """
    parquet_file = pq.ParquetFile(filepath)
    return parquet_file.schema_arrow
