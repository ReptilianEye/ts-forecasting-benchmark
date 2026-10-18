#!/usr/bin/env python3
"""CLI script to download benchmark datasets for time series forecasting.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset ETTh1
"""

import argparse
from pathlib import Path

from ts_forecasting_benchmark.data.download import (
    BENCHMARK_URLS,
    download_all_benchmarks,
    download_dataset,
)

project_root = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Download LTSF benchmark datasets.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all benchmark datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset name to download or 'all'. Choices: {list(BENCHMARK_URLS.keys()) + ['all']}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "raw"),
        help="Target directory to store dataset CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if file exists.",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    print(f"Target directory: {out_dir}")
    if args.dataset.lower() == "all":
        print(f"Downloading all benchmark datasets: {list(BENCHMARK_URLS.keys())}")
        results = download_all_benchmarks(out_dir, force_redownload=args.force)
        print("\nDownload complete:")
        for name, path in results.items():
            print(f"  ✓ {name:12s} -> {path} ({path.stat().st_size / (1024 * 1024):.2f} MB)")
    else:
        path = download_dataset(args.dataset, out_dir, force_redownload=args.force)
        print(
            f"\nDownload complete: {args.dataset} -> {path} ({path.stat().st_size / (1024 * 1024):.2f} MB)"
        )


if __name__ == "__main__":
    main()
