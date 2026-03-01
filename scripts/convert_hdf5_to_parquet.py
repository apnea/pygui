#!/usr/bin/env python
"""Utility to convert HDF5 files to Parquet format."""

import argparse
from pathlib import Path

import pandas as pd


def convert_hdf5_to_parquet(
    hdf5_path: str, parquet_path: str | None = None, table_key: str | None = None
) -> None:
    """Convert HDF5 file to Parquet format.

    Args:
        hdf5_path: Path to input HDF5 file
        parquet_path: Path to output Parquet file (default: same as HDF5 but .parquet)
        table_key: Key of table in HDF5 file (required if multiple tables)
    """
    hdf5_path_obj = Path(hdf5_path)

    if not hdf5_path_obj.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path_obj}")

    if parquet_path is None:
        parquet_path_obj = hdf5_path_obj.with_suffix(".parquet")
    else:
        parquet_path_obj = Path(parquet_path)

    print(f"Reading HDF5 file: {hdf5_path_obj}")

    with pd.HDFStore(str(hdf5_path_obj), mode="r") as store:
        keys = store.keys()

        if table_key is None:
            if len(keys) == 1:
                table_key = keys[0].strip("/")
                print(f"Auto-detected table key: {table_key}")
            else:
                raise ValueError(
                    f"Multiple tables found in HDF5 file: {keys}. Please specify --table-key"
                )
        else:
            if f"/{table_key}" not in keys:
                raise ValueError(f"Table key '{table_key}' not found. Available: {keys}")

        df = store[table_key]

    print(f"Converting table '{table_key}' with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    parquet_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(parquet_path_obj), engine="pyarrow")
    print(f"Successfully saved to: {parquet_path_obj}")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 files to Parquet format")
    parser.add_argument("hdf5_path", help="Path to input HDF5 file")
    parser.add_argument("--output", "-o", help="Path to output Parquet file (optional)")
    parser.add_argument(
        "--table-key", "-t", help="Table key in HDF5 file (optional if single table)"
    )

    args = parser.parse_args()

    convert_hdf5_to_parquet(args.hdf5_path, args.output, args.table_key)


if __name__ == "__main__":
    main()
