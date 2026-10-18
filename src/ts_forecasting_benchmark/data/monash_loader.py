"""Monash Time Series Forecasting Archive Dataset Loader.

Parses .tsf (Time Series Forecasting) format files into pandas DataFrames.
Compatible with Python 3.12+ (no distutils dependency).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _strtobool(val: str) -> bool:
    """Convert a string representation of truth to boolean."""
    val = val.lower().strip()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value: {val}")


def convert_tsf_to_dataframe(
    full_file_path_and_name: str | Path,
    replace_missing_vals_with: Any = np.nan,
    value_column_name: str = "series_value",
    encoding: str = "cp1252",
) -> Tuple[pd.DataFrame, Optional[str], Optional[int], Optional[bool], Optional[bool]]:
    """Convert the contents of a .tsf file into a pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name : str or Path
        Path to the .tsf file.
    replace_missing_vals_with : Any, default np.nan
        Placeholder to replace missing values ('?') in the series.
    value_column_name : str, default 'series_value'
        Name of the column containing the series values in the returned DataFrame.
    encoding : str, default 'cp1252'
        File encoding (Monash files are usually encoded in cp1252 or utf-8).

    Returns
    -------
    loaded_data : pd.DataFrame
        DataFrame where each row corresponds to a time series.
    frequency : Optional[str]
        Series frequency ('daily', 'hourly', 'monthly', etc.).
    forecast_horizon : Optional[int]
        Recommended forecast horizon for the dataset.
    contain_missing_values : Optional[bool]
        Whether the series contain missing values.
    contain_equal_length : Optional[bool]
        Whether all series have the same length.
    """
    path = Path(full_file_path_and_name)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    col_names = []
    col_types = []
    all_data: Dict[str, list] = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False
    all_series = []

    with open(path, "r", encoding=encoding, errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith("@"):
                if not line.startswith("@data"):
                    line_content = line.split(" ")
                    if line.startswith("@attribute"):
                        if len(line_content) != 3:
                            raise ValueError(f"Invalid attribute specification: {line}")
                        col_names.append(line_content[1])
                        col_types.append(line_content[2])
                    else:
                        if len(line_content) != 2:
                            raise ValueError(f"Invalid meta-data specification: {line}")
                        tag = line_content[0].lower()
                        if tag == "@frequency":
                            frequency = line_content[1]
                        elif tag == "@horizon":
                            forecast_horizon = int(line_content[1])
                        elif tag == "@missing":
                            contain_missing_values = _strtobool(line_content[1])
                        elif tag == "@equallength":
                            contain_equal_length = _strtobool(line_content[1])
                else:
                    if len(col_names) == 0:
                        raise ValueError("Missing attribute section before @data.")
                    found_data_tag = True
            elif not line.startswith("#"):
                if len(col_names) == 0:
                    raise ValueError("Missing attribute section before @data.")
                elif not found_data_tag:
                    raise ValueError("Missing @data tag before data rows.")
                else:
                    if not started_reading_data_section:
                        started_reading_data_section = True
                        found_data_section = True
                        for col in col_names:
                            all_data[col] = []

                    full_info = line.split(":")
                    if len(full_info) != (len(col_names) + 1):
                        raise ValueError("Mismatch between attribute count and data columns.")

                    series_raw = full_info[-1].split(",")
                    if len(series_raw) == 0:
                        raise ValueError("Series contains no values.")

                    numeric_series = []
                    for val in series_raw:
                        val = val.strip()
                        if val == "?":
                            numeric_series.append(replace_missing_vals_with)
                        else:
                            numeric_series.append(float(val))

                    all_series.append(np.array(numeric_series, dtype=np.float64))

                    for i, col in enumerate(col_names):
                        att_type = col_types[i].lower()
                        raw_val = full_info[i].strip()
                        if att_type == "numeric":
                            att_val = int(raw_val) if raw_val.isdigit() else float(raw_val)
                        elif att_type == "string":
                            att_val = str(raw_val)
                        elif att_type == "date":
                            # Standard Monash formats
                            try:
                                att_val = datetime.strptime(raw_val, "%Y-%m-%d %H-%M-%S")
                            except ValueError:
                                att_val = datetime.strptime(raw_val, "%Y-%m-%d")
                        else:
                            att_val = raw_val
                        all_data[col].append(att_val)

            line_count += 1

    if line_count == 0:
        raise ValueError("File is empty.")
    if not found_data_section:
        raise ValueError("No series data found under @data section.")

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)

    return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length


def to_long_format(
    df: pd.DataFrame,
    series_id_col: str = "series_name",
    value_col: str = "series_value",
    start_timestamp_col: Optional[str] = "start_timestamp",
    frequency_str: str = "1h",
) -> pd.DataFrame:
    """Convert Monash DataFrame to long-format (unique_id, ds, y) for NeuralForecast / Nixtla.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by `convert_tsf_to_dataframe`.
    series_id_col : str
        Column containing series identifier.
    value_col : str
        Column containing 1D numpy array of values.
    start_timestamp_col : Optional[str]
        Optional start timestamp column.
    frequency_str : str
        Frequency string for date range generation if start_timestamp is present.

    Returns
    -------
    pd.DataFrame
        Long format dataframe with columns: unique_id, ds, y.
    """
    records = []
    has_start = start_timestamp_col and start_timestamp_col in df.columns

    for idx, row in df.iterrows():
        sid = row[series_id_col] if series_id_col in df.columns else f"T_{idx}"
        vals = row[value_col]
        n = len(vals)

        if has_start and pd.notnull(row[start_timestamp_col]):
            start_dt = row[start_timestamp_col]
            timestamps = pd.date_range(start=start_dt, periods=n, freq=frequency_str)
        else:
            timestamps = np.arange(n)

        for t, v in zip(timestamps, vals):
            records.append({"unique_id": str(sid), "ds": t, "y": v})

    return pd.DataFrame(records)

