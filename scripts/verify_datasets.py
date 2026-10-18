#!/usr/bin/env python3
"""Dataset verification script for LTSF benchmarks and Monash loader.

Usage:
    python scripts/verify_datasets.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

from ts_forecasting_benchmark.data import (
    FILENAME_MAP,
    TimeSeriesBenchmarkDataset,
    convert_tsf_to_dataframe,
    load_benchmark_dataframe,
    to_long_format,
)

project_root = Path(__file__).resolve().parent.parent


def verify_monash_loader() -> bool:
    """Verify Monash .tsf parser on a synthetic .tsf payload."""
    sample_tsf_content = """@relation test_dataset
@attribute series_name string
@attribute start_timestamp date
@frequency daily
@horizon 14
@missing false
@equallength true
@data
T1:2020-01-01 00-00-00:10.5,12.3,14.1,15.0,16.2
T2:2020-01-01 00-00-00:20.1,21.0,22.5,23.1,24.0
"""
    with tempfile.NamedTemporaryFile("w", suffix=".tsf", delete=False) as tmp:
        tmp.write(sample_tsf_content)
        tmp_path = Path(tmp.name)

    try:
        df, freq, horizon, missing, equal = convert_tsf_to_dataframe(tmp_path)
        assert len(df) == 2, f"Expected 2 series, got {len(df)}"
        assert freq == "daily", f"Expected 'daily', got {freq}"
        assert horizon == 14, f"Expected 14, got {horizon}"
        assert not missing, "Expected missing to be False"
        assert equal, "Expected equal length to be True"

        long_df = to_long_format(df, frequency_str="1D")
        assert len(long_df) == 10, f"Expected 10 total points, got {len(long_df)}"
        assert set(long_df.columns) == {"unique_id", "ds", "y"}
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def verify_benchmarks(data_dir: Path) -> pd.DataFrame:
    """Verify all benchmark CSVs and return a summary dataframe."""
    records = []
    print("\n" + "=" * 90)
    print(
        f"{'Dataset':<14} | {'Rows':<8} | {'Cols':<6} | {'Date Range':<32} | {'Missing':<8} | {'Status'}"
    )
    print("=" * 90)

    for name, filename in FILENAME_MAP.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"{name:<14} | {'MISSING':<8} | {'-':<6} | {'-':<32} | {'-':<8} | ❌ NOT FOUND")
            records.append({"Dataset": name, "Status": "MISSING", "File": str(filepath)})
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        try:
            df = load_benchmark_dataframe(filepath)
            date_col = next(
                (c for c in ["date", "Date", "timestamp"] if c in df.columns), df.columns[0]
            )
            num_rows, num_cols = df.shape
            start_date = str(df[date_col].iloc[0])[:19]
            end_date = str(df[date_col].iloc[-1])[:19]
            date_range = f"{start_date} -> {end_date}"
            missing_count = df.isnull().sum().sum()

            # Test PyTorch dataset generation
            ds = TimeSeriesBenchmarkDataset(df, seq_len=96, pred_len=96, split="train")
            x, y = ds[0]
            assert x.shape == (96, num_cols - 1), f"Unexpected x shape: {x.shape}"
            assert y.shape == (96, num_cols - 1), f"Unexpected y shape: {y.shape}"

            print(
                f"{name:<14} | {num_rows:<8} | {num_cols:<6} | {date_range:<32} | {missing_count:<8} | ✅ OK ({size_mb:.1f} MB)"
            )
            records.append(
                {
                    "Dataset": name,
                    "Filename": filename,
                    "Rows": num_rows,
                    "Columns": num_cols,
                    "Date_Start": start_date,
                    "Date_End": end_date,
                    "Missing_Values": int(missing_count),
                    "Size_MB": round(size_mb, 2),
                    "Status": "VERIFIED",
                }
            )
        except Exception as e:
            print(f"{name:<14} | {'ERROR':<8} | {'-':<6} | {str(e)[:32]:<32} | {'-':<8} | ❌ {e}")
            records.append({"Dataset": name, "Status": f"ERROR: {e}", "File": str(filepath)})

    print("=" * 90)
    return pd.DataFrame(records)


def main():
    data_dir = project_root / "data" / "raw"
    print(f"Verifying benchmark datasets in: {data_dir}")

    # 1. Monash loader test
    print("\n[1/2] Testing Monash .tsf dataset loader...")
    try:
        if verify_monash_loader():
            print("  ✅ Monash .tsf dataset loader verified successfully!")
    except Exception as e:
        print(f"  ❌ Monash loader failed: {e}")
        sys.exit(1)

    # 2. Benchmark datasets verification
    print("\n[2/2] Verifying LTSF benchmark datasets...")
    summary_df = verify_benchmarks(data_dir)

    all_verified = all(summary_df["Status"] == "VERIFIED") if not summary_df.empty else False
    if all_verified:
        print("\nAll 7 benchmark datasets are downloaded and verified successfully!")
        sys.exit(0)
    else:
        missing = summary_df[summary_df["Status"] != "VERIFIED"]["Dataset"].tolist()
        print(f"\nDatasets not yet verified or missing: {missing}")
        print("Run `python scripts/download_datasets.py --all` to download them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
