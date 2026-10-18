"""Dataset download utilities for Long-Term Time Series Forecasting (LTSF) benchmarks.

Supported datasets:
- ETTh1, ETTh2 (1-hour Electricity Transformer Temperature)
- ETTm1, ETTm2 (15-minute Electricity Transformer Temperature)
- Weather (21 meteorological indicators from Max Planck Institute)
- Electricity (Hourly consumption of 321 clients)
- Traffic (San Francisco freeway occupancy rate from Caltrans)
- Monash repository datasets (via Zenodo/URL)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_URLS: Dict[str, List[str]] = {
    "ETTh1": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTh1.csv",
    ],
    "ETTh2": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTh2.csv",
    ],
    "ETTm1": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTm1.csv",
    ],
    "ETTm2": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm2.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/data/ETT/ETTm2.csv",
    ],
    "Weather": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
    ],
    "Electricity": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
    ],
    "Traffic": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv",
    ],
}

FILENAME_MAP: Dict[str, str] = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Weather": "weather.csv",
    "Electricity": "electricity.csv",
    "Traffic": "traffic.csv",
}


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file with a tqdm progress bar and resume/stream support."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TSBenchmarkDownloader/1.0)"}
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as response:
            if response.status_code != 200:
                logger.warning("HTTP %d for URL: %s", response.status_code, url)
                return False

            total_size = int(response.headers.get("content-length", 0))
            with open(temp_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                leave=True,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        temp_path.rename(dest_path)
        return True
    except Exception as exc:
        logger.error("Download failed for %s: %s", url, exc)
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_dataset(
    dataset_name: str,
    output_dir: str | Path,
    force_redownload: bool = False,
) -> Path:
    """Download a benchmark dataset by name.

    Parameters
    ----------
    dataset_name : str
        One of 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic'.
    output_dir : str or Path
        Target directory to save the dataset.
    force_redownload : bool, default False
        Whether to redownload if file already exists and is non-empty.

    Returns
    -------
    Path
        Path to the downloaded dataset file.
    """
    normalized_name = dataset_name.strip()
    match = None
    for k in BENCHMARK_URLS:
        if k.lower() == normalized_name.lower():
            match = k
            break

    if not match:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(BENCHMARK_URLS.keys())}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = FILENAME_MAP[match]
    dest_path = out_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 0 and not force_redownload:
        logger.info("Dataset '%s' already exists at %s (%d bytes). Skipping download.", match, dest_path, dest_path.stat().st_size)
        return dest_path

    urls = BENCHMARK_URLS[match]
    success = False
    for url in urls:
        logger.info("Attempting download of '%s' from %s", match, url)
        if download_file_with_progress(url, dest_path):
            success = True
            logger.info("Successfully downloaded '%s' to %s", match, dest_path)
            break

    if not success or not dest_path.exists():
        raise RuntimeError(f"Failed to download dataset '{match}' from all available mirrors.")

    return dest_path


def download_all_benchmarks(output_dir: str | Path, force_redownload: bool = False) -> Dict[str, Path]:
    """Download all 7 benchmark datasets."""
    results: Dict[str, Path] = {}
    for name in BENCHMARK_URLS:
        results[name] = download_dataset(name, output_dir, force_redownload=force_redownload)
    return results

