"""Tests for data loader module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pygui.data.loader import (
    _detect_datetime_column,
    get_data_info,
    get_parquet_schema,
    load_parquet,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with datetime index."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "value1": range(1000),
            "value2": range(1000, 2000),
            "category": ["A", "B", "C"] * 333 + ["A"],
        }
    )
    return df


@pytest.fixture
def sample_parquet_file(sample_dataframe):
    """Create a temporary Parquet file with sample data."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name

    sample_dataframe.to_parquet(tmp_path, engine="pyarrow")
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


def test_load_parquet_success(sample_parquet_file):
    """Test successful loading of Parquet file."""
    df = load_parquet(sample_parquet_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "value1" in df.columns
    assert "value2" in df.columns


def test_load_parquet_file_not_found():
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_parquet("nonexistent.parquet")


def test_detect_datetime_column_with_datetime_col():
    """Test datetime column detection with explicit datetime column."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1, 2],
        }
    )

    result = _detect_datetime_column(df)
    assert result == "timestamp"


def test_detect_datetime_column_no_datetime():
    """Test datetime column detection with no datetime column."""
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )

    result = _detect_datetime_column(df)
    assert result is None


def test_get_data_info(sample_dataframe):
    """Test getting data information."""
    info = get_data_info(sample_dataframe)

    assert "shape" in info
    assert info["shape"] == (1000, 4)
    assert "columns" in info
    assert len(info["columns"]) == 4
    assert "memory_usage_mb" in info


def test_get_data_info_with_datetime_index(sample_dataframe):
    """Test data info with datetime index."""
    df = sample_dataframe.set_index("timestamp")
    info = get_data_info(df)

    assert "time_range" in info
    assert "start" in info["time_range"]
    assert "end" in info["time_range"]
    assert "duration" in info["time_range"]


def test_get_parquet_schema(sample_parquet_file):
    """Test getting Parquet schema."""
    schema = get_parquet_schema(sample_parquet_file)

    assert schema is not None
    assert len(schema.names) > 0
