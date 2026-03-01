"""Tests for statistics module."""

import numpy as np
import pandas as pd
import pytest

from pygui.analysis.statistics import (
    basic_stats,
    correlation_matrix,
    distributions,
    outlier_stats,
    time_based_stats,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "value3": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        }
    )
    return df


@pytest.fixture
def sample_time_series():
    """Create a sample time series DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "value1": np.random.randn(100),
            "value2": np.random.randn(100) * 2 + 5,
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_with_outliers():
    """Create a sample DataFrame with outliers."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
            "value2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    return df


def test_basic_stats(sample_dataframe):
    """Test basic statistics calculation."""
    result = basic_stats(sample_dataframe)

    assert "count" in result.index
    assert "mean" in result.index
    assert "std" in result.index
    assert "min" in result.index
    assert "max" in result.index
    assert "q25" in result.index
    assert "q50" in result.index
    assert "q75" in result.index

    assert result.loc["count", "value1"] == 10
    assert result.loc["mean", "value1"] == 5.5
    assert result.loc["max", "value1"] == 10


def test_time_based_stats(sample_time_series):
    """Test time-based statistics aggregation."""
    result = time_based_stats(sample_time_series, freq="6H")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 17  # 100 hours / 6 ≈ 17 bins
    assert result.shape[1] == 6  # 2 columns * 3 aggregations (mean, std, count)


def test_correlation_matrix(sample_dataframe):
    """Test correlation matrix calculation."""
    result = correlation_matrix(sample_dataframe)

    assert result.shape == (3, 3)
    assert result.loc["value1", "value1"] == 1.0
    assert result.loc["value2", "value2"] == 1.0

    assert result.loc["value1", "value2"] > 0.9  # Should be highly correlated


def test_correlation_matrix_mixed_types():
    """Test correlation matrix with mixed types."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, 3, 4, 5],
            "value2": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "B", "A"],
        }
    )

    result = correlation_matrix(df)

    assert result.shape == (2, 2)  # Only numeric columns
    assert "value1" in result.columns
    assert "value2" in result.columns


def test_outlier_stats(sample_with_outliers):
    """Test outlier statistics calculation."""
    result = outlier_stats(sample_with_outliers)

    assert "value1" in result
    assert "value2" in result

    assert "count" in result["value1"]
    assert "percentage" in result["value1"]

    assert result["value1"]["count"] >= 1  # Should detect at least one outlier


def test_outlier_stats_with_nan():
    """Test outlier stats with NaN values."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, np.nan, 4, 5, 100, 7, 8, 9, 10],
        }
    )

    result = outlier_stats(df)

    assert "value1" in result
    assert isinstance(result["value1"]["count"], (int, np.integer))


def test_distributions(sample_dataframe):
    """Test distribution calculation."""
    result = distributions(sample_dataframe, bins=5)

    assert "value1" in result
    assert "value2" in result
    assert "value3" in result

    assert "bins" in result["value1"]
    assert "counts" in result["value1"]

    assert len(result["value1"]["bins"]) == 5
    assert len(result["value1"]["counts"]) == 5


def test_distributions_custom_columns(sample_dataframe):
    """Test distribution calculation with specific columns."""
    result = distributions(sample_dataframe, columns=["value1"], bins=10)

    assert "value1" in result
    assert "value2" not in result
    assert "value3" not in result

    assert len(result["value1"]["bins"]) == 10


def test_basic_stats_with_nan():
    """Test basic statistics with NaN values."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, np.nan, 4, 5],
            "value2": [10, 20, 30, np.nan, 50],
        }
    )

    result = basic_stats(df)

    assert result.loc["count", "value1"] == 4
    assert result.loc["count", "value2"] == 4
