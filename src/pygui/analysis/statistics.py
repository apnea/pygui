"""Descriptive statistics for time series data."""

import numpy as np
import pandas as pd


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic descriptive statistics.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with count, mean, std, min, max, q25, q50, q75 for each column
    """
    stats = pd.DataFrame(
        {
            "count": df.count(),
            "mean": df.mean(),
            "std": df.std(),
            "min": df.min(),
            "q25": df.quantile(0.25),
            "q50": df.quantile(0.50),
            "q75": df.quantile(0.75),
            "max": df.max(),
        }
    )

    return stats.T


def time_based_stats(df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    """Calculate statistics aggregated by time buckets.

    Args:
        df: Input DataFrame with DatetimeIndex
        freq: Aggregation frequency ('H' for hourly, 'D' for daily, etc.)

    Returns:
        DataFrame with aggregated statistics
    """
    stats_by_time = df.resample(freq).agg(["mean", "std", "count"])

    stats_by_time.columns = [f"{col}_{stat}" for col, stat in stats_by_time.columns]

    return stats_by_time


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix between numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        Correlation matrix
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()


def outlier_stats(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict]:
    """Calculate outlier statistics.

    Args:
        df: Input DataFrame
        columns: Columns to analyze (default: all numeric)

    Returns:
        Dictionary with outlier count and percentage per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in columns:
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

        stats[col] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(col_data)) * 100,
        }

    return stats


def distributions(
    df: pd.DataFrame, columns: list[str] | None = None, bins: int = 50
) -> dict[str, dict]:
    """Calculate histogram data for distributions.

    Args:
        df: Input DataFrame
        columns: Columns to analyze (default: all numeric)
        bins: Number of histogram bins

    Returns:
        Dictionary with histogram bins and counts for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    dist_data = {}

    for col in columns:
        col_data = df[col].dropna()
        counts, bin_edges = np.histogram(col_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        dist_data[col] = {
            "bins": bin_centers.tolist(),
            "counts": counts.tolist(),
        }

    return dist_data
