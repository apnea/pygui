"""Tests for data transformer module."""

import numpy as np
import pandas as pd
import pytest

from pygui.data.transformer import (
    create_lags,
    custom_aggregate,
    detect_outliers,
    difference,
    filter_by_time_range,
    impute_missing,
    resample,
    rolling_window,
)


@pytest.fixture
def sample_time_series():
    """Create a sample time series DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    df = pd.DataFrame(
        {
            "value1": range(100),
            "value2": range(100, 200),
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_with_nan():
    """Create a sample DataFrame with NaN values."""
    df = pd.DataFrame(
        {
            "value1": [1, 2, np.nan, 4, np.nan, 6],
            "value2": [10, np.nan, 30, 40, np.nan, 60],
        }
    )
    return df


def test_filter_by_time_range(sample_time_series):
    """Test filtering by time range."""
    result = filter_by_time_range(sample_time_series, "2024-01-01", "2024-01-02")

    assert len(result) == 25  # 24 hours + 1 for inclusive end
    assert result.index.min() == pd.Timestamp("2024-01-01")
    assert result.index.max() <= pd.Timestamp("2024-01-02")


def test_resample(sample_time_series):
    """Test resampling data."""
    result = resample(sample_time_series, freq="6H", agg_func="mean")

    assert len(result) == 17  # 100 hours / 6 ≈ 17 bins
    assert isinstance(result.index, pd.DatetimeIndex)


def test_rolling_window(sample_time_series):
    """Test rolling window calculation."""
    result = rolling_window(sample_time_series, window=10, agg_func="mean")

    assert result.shape == sample_time_series.shape
    assert result.iloc[:9].isna().all().all()  # First 9 rows should be NaN


def test_detect_outliers_zscore(sample_time_series):
    """Test outlier detection using Z-score method."""
    result = detect_outliers(sample_time_series, method="zscore", threshold=2.5)

    assert result.shape == sample_time_series.shape
    assert result.dtypes.apply(str).str.startswith("bool").all()


def test_detect_outliers_iqr(sample_time_series):
    """Test outlier detection using IQR method."""
    result = detect_outliers(sample_time_series, method="iqr", threshold=1.5)

    assert result.shape == sample_time_series.shape
    assert result.dtypes.apply(str).str.startswith("bool").all()


def test_impute_missing_ffill(sample_with_nan):
    """Test forward fill imputation."""
    result = impute_missing(sample_with_nan, method="ffill")

    assert result["value1"].iloc[2] == 2  # Forward filled
    assert result["value2"].iloc[1] == 10  # Forward filled


def test_impute_missing_bfill(sample_with_nan):
    """Test backward fill imputation."""
    result = impute_missing(sample_with_nan, method="bfill")

    assert result["value1"].iloc[4] == 6  # Backward filled
    assert result["value2"].iloc[4] == 60  # Backward filled


def test_impute_missing_interpolate(sample_with_nan):
    """Test interpolation imputation."""
    result = impute_missing(sample_with_nan, method="interpolate")

    assert not result.isna().any().any()  # No NaN values should remain


def test_difference(sample_time_series):
    """Test differencing."""
    result = difference(sample_time_series, periods=1)

    assert result.shape == sample_time_series.shape
    assert pd.isna(result.iloc[0, 0])  # First row should be NaN


def test_create_lags(sample_time_series):
    """Test creating lagged features."""
    result = create_lags(sample_time_series, lags=[1, 2])

    assert "value1_lag_1" in result.columns
    assert "value1_lag_2" in result.columns
    assert "value2_lag_1" in result.columns
    assert "value2_lag_2" in result.columns


def test_create_lags_single_lag(sample_time_series):
    """Test creating single lagged feature."""
    result = create_lags(sample_time_series, lags=3)

    assert "value1_lag_3" in result.columns
    assert "value2_lag_3" in result.columns


def test_custom_aggregate():
    """Test custom aggregation."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C"],
            "value1": [1, 2, 3, 4, 5, 6],
            "value2": [10, 20, 30, 40, 50, 60],
        }
    )

    result = custom_aggregate(df, groupby="group", agg_dict={"value1": "mean", "value2": "sum"})

    assert len(result) == 3
    assert "A" in result.index
    assert "B" in result.index
    assert "C" in result.index


def test_custom_aggregate_multiple_functions():
    """Test custom aggregation with multiple functions."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1, 3, 5, 7],
        }
    )

    result = custom_aggregate(df, groupby="group", agg_dict={"value": ["mean", "std"]})

    assert len(result) == 2
    assert ("value", "mean") in result.columns
    assert ("value", "std") in result.columns
