"""Data transformation utilities for time series."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def filter_by_time_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter DataFrame by time range.

    Args:
        df: Input DataFrame with DatetimeIndex
        start: Start time (ISO format string)
        end: End time (ISO format string)

    Returns:
        Filtered DataFrame
    """
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)
    return df.loc[start_time:end_time]


def resample(df: pd.DataFrame, freq: str, agg_func: str | dict | list = "mean") -> pd.DataFrame:
    """Resample DataFrame to different frequency.

    Args:
        df: Input DataFrame with DatetimeIndex
        freq: Resampling frequency (e.g., 'H', 'D', '15T', '1S')
        agg_func: Aggregation function(s) to apply

    Returns:
        Resampled DataFrame
    """
    return df.resample(freq).agg(agg_func).dropna()


def rolling_window(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int | str = 10,
    agg_func: str | Callable = "mean",
) -> pd.DataFrame:
    """Calculate rolling window statistics.

    Args:
        df: Input DataFrame
        columns: Columns to process (default: all numeric)
        window: Window size (int for number of periods, str for time offset)
        agg_func: Aggregation function

    Returns:
        DataFrame with rolling statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df[columns].rolling(window).agg(agg_func)
    return result


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Detect outliers in DataFrame.

    Args:
        df: Input DataFrame
        columns: Columns to check (default: all numeric)
        method: Detection method ('zscore' or 'iqr')
        threshold: Threshold for outlier detection

    Returns:
        Boolean DataFrame indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
        elif method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers[col] = (df[col] < q1 - threshold * iqr) | (df[col] > q3 + threshold * iqr)
        else:
            raise ValueError(f"Unknown method: {method}")

    return outliers


def impute_missing(
    df: pd.DataFrame,
    method: str = "ffill",
    limit: int | None = None,
) -> pd.DataFrame:
    """Impute missing values in DataFrame.

    Args:
        df: Input DataFrame
        method: Imputation method ('ffill', 'bfill', 'interpolate')
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with imputed values
    """
    if method == "ffill":
        return df.ffill(limit=limit)
    elif method == "bfill":
        return df.bfill(limit=limit)
    elif method == "interpolate":
        if isinstance(df.index, pd.DatetimeIndex):
            return df.interpolate(method="time", limit=limit)
        else:
            return df.interpolate(method="linear", limit=limit)
    else:
        raise ValueError(f"Unknown method: {method}")


def difference(
    df: pd.DataFrame, columns: list[str] | None = None, periods: int = 1
) -> pd.DataFrame:
    """Calculate differences for stationarity.

    Args:
        df: Input DataFrame
        columns: Columns to difference (default: all numeric)
        periods: Number of periods to shift for differencing

    Returns:
        DataFrame with differenced values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    for col in columns:
        result[col] = df[col].diff(periods=periods)

    return result


def create_lags(
    df: pd.DataFrame, columns: list[str] | None = None, lags: int | list[int] = 1
) -> pd.DataFrame:
    """Create lagged features.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for (default: all numeric)
        lags: Lag value(s) to create

    Returns:
        DataFrame with lagged columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if isinstance(lags, int):
        lags = [lags]

    result = df.copy()
    for col in columns:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return result


def custom_aggregate(
    df: pd.DataFrame,
    groupby: str | list[str],
    agg_dict: dict[str, str | list[str] | Callable],
) -> pd.DataFrame:
    """Perform custom groupby aggregation.

    Args:
        df: Input DataFrame
        groupby: Column(s) to group by
        agg_dict: Dictionary mapping columns to aggregation functions

    Returns:
        Aggregated DataFrame
    """
    return df.groupby(groupby).agg(agg_dict)
