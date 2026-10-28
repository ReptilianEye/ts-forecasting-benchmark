"""Dataset utilities and PyTorch Dataset implementations for LTSF benchmarks.

Supports:
- Standard LTSF splits (Train / Val / Test)
- Data scaling (StandardScaler fit on train only)
- Univariate and Multivariate forecasting
- PyTorch Dataset windowing (lookback L, horizon H)
- Long-format conversion for NeuralForecast / Nixtla
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# Standard benchmark split ratios used in LTSF literature (Autoformer, Informer, PatchTST, DLinear)
SPLIT_CONFIGS = {
    "ETTh1": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "freq": "1h"},
    "ETTh2": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "freq": "1h"},
    "ETTm1": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "freq": "15min"},
    "ETTm2": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "freq": "15min"},
    "Weather": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "freq": "10min"},
    "Electricity": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "freq": "1h"},
    "Traffic": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "freq": "1h"},
}


def load_benchmark_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load a benchmark CSV file and parse date column."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    date_col = next((c for c in ["date", "Date", "timestamp", "datetime"] if c in df.columns), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


class TimeSeriesBenchmarkDataset(Dataset):
    """PyTorch Dataset for sliding-window time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series columns and date.
    seq_len : int
        Input lookback window length (L).
    pred_len : int
        Forecast horizon length (H).
    target_col : str, optional
        Target column for univariate forecasting. If None, all numeric columns are used (multivariate).
    split : {'train', 'val', 'test'}
        Which split to use.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.
    scale : bool
        Whether to normalize features using StandardScaler fitted on train split.
    scaler : StandardScaler, optional
        Pre-fitted scaler instance. If None and split == 'train', a new scaler is fitted.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 96,
        pred_len: int = 96,
        target_col: Optional[str] = None,
        split: Literal["train", "val", "test"] = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        scale: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.target_col = target_col

        date_col = next((c for c in ["date", "Date", "timestamp"] if c in df.columns), None)
        feature_cols = [c for c in df.columns if c != date_col]

        if target_col is not None:
            if target_col not in feature_cols:
                raise ValueError(f"Target '{target_col}' not found in columns: {feature_cols}")
            data_raw = df[[target_col]].values.astype(np.float32)
        else:
            data_raw = df[feature_cols].values.astype(np.float32)

        n_samples = len(data_raw)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        # Indices with lookback overlap for validation and test
        if split == "train":
            border1 = 0
            border2 = n_train
        elif split == "val":
            border1 = n_train - seq_len
            border2 = n_train + n_val
        elif split == "test":
            border1 = n_train + n_val - seq_len
            border2 = n_samples
        else:
            raise ValueError(f"Unknown split: {split}")

        # Scale data using train split only
        if scale:
            if scaler is None:
                train_data = data_raw[:n_train]
                self.scaler = StandardScaler()
                self.scaler.fit(train_data)
            else:
                self.scaler = scaler
            self.data = self.scaler.transform(data_raw[border1:border2])
        else:
            self.scaler = None
            self.data = data_raw[border1:border2]

        self.num_samples = len(self.data) - self.seq_len - self.pred_len + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


def to_neuralforecast_df(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert wide multivariate DataFrame into long format (unique_id, ds, y) for NeuralForecast."""
    if date_col is None:
        date_col = next((c for c in ["date", "Date", "timestamp"] if c in df.columns), df.columns[0])

    if target_col is not None:
        cols_to_melt = [target_col]
    else:
        cols_to_melt = [c for c in df.columns if c != date_col]

    melted = df.melt(id_vars=[date_col], value_vars=cols_to_melt, var_name="unique_id", value_name="y")
    melted = melted.rename(columns={date_col: "ds"})
    melted["ds"] = pd.to_datetime(melted["ds"])
    melted = melted.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return melted


@dataclass
class LongBenchmarkData:
    """Container for long-format benchmark splits and metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    full_df: pd.DataFrame
    freq: str
    target_col: Optional[str]
    scalers: Dict[str, Tuple[float, float]]  # {unique_id: (mean, std)}

    def inverse_transform(self, df: pd.DataFrame, col_name: str = "y") -> pd.DataFrame:
        """Denormalize a column using the training mean and std for each series."""
        out = df.copy()
        for uid, (mean, std) in self.scalers.items():
            mask = out["unique_id"] == uid
            if col_name in out.columns and mask.any():
                out.loc[mask, col_name] = out.loc[mask, col_name] * std + mean
        return out


def load_long_benchmark(
    dataset_name_or_path: str | Path,
    target_col: Optional[str] = "OT",
    scale: bool = True,
    data_dir: Optional[str | Path] = None,
) -> LongBenchmarkData:
    """Universal long-format loader for StatsForecast, NeuralForecast, and Chronos.

    Loads a benchmark dataset, melts it into `(unique_id, ds, y)`, splits chronologically
    into train, val, and test partitions, and optionally standardizes the series based on
    the training partition.

    Parameters
    ----------
    dataset_name_or_path : str or Path
        Name of dataset (e.g., 'ETTh1', 'Weather') or direct path to CSV.
    target_col : str, optional
        Target column for univariate forecasting (e.g. 'OT'). If None, all numeric columns
        are extracted as separate series with their column name as unique_id.
    scale : bool
        If True, standardizes values per series using (y - mean_train) / std_train.
    data_dir : str or Path, optional
        Directory where CSVs are stored. Defaults to 'data/raw' relative to project root.

    Returns
    -------
    LongBenchmarkData
        Dataclass containing train_df, val_df, test_df, full_df, freq, and scaler parameters.
    """
    path = Path(dataset_name_or_path)

    # Resolve dataset name and path
    dataset_name = path.stem
    if not path.exists():
        search_dirs = [
            Path(data_dir) if data_dir else None,
            Path("data/raw"),
            Path("project/data/raw"),
            Path(__file__).parents[3] / "data" / "raw",
        ]
        found_path = None
        for s_dir in search_dirs:
            if s_dir is None:
                continue
            candidates = [
                s_dir / f"{dataset_name}.csv",
                s_dir / f"{dataset_name.lower()}.csv",
                s_dir / f"{dataset_name.capitalize()}.csv",
            ]
            for cand in candidates:
                if cand.exists():
                    found_path = cand
                    break
            if found_path:
                break

        if found_path is None:
            raise FileNotFoundError(f"Could not find benchmark dataset '{dataset_name_or_path}'")
        path = found_path

    # Load wide dataframe
    wide_df = load_benchmark_dataframe(path)

    # Determine split configuration and frequency
    cfg = SPLIT_CONFIGS.get(dataset_name, SPLIT_CONFIGS.get(dataset_name.capitalize(), {}))
    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.1)
    freq = cfg.get("freq", "1h")

    # Melt to long format (unique_id, ds, y)
    long_df = to_neuralforecast_df(wide_df, target_col=target_col)

    # Calculate train / val / test borders per series
    # All series share the same timestamps and lengths
    n_total = len(wide_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_dates = wide_df["date"].iloc[:n_train]
    val_dates = wide_df["date"].iloc[n_train : n_train + n_val]
    test_dates = wide_df["date"].iloc[n_train + n_val :]

    train_mask = long_df["ds"].isin(train_dates)
    val_mask = long_df["ds"].isin(val_dates)
    test_mask = long_df["ds"].isin(test_dates)

    train_df = long_df[train_mask].copy().reset_index(drop=True)
    val_df = long_df[val_mask].copy().reset_index(drop=True)
    test_df = long_df[test_mask].copy().reset_index(drop=True)

    # Standardization per series using train statistics
    scalers: Dict[str, Tuple[float, float]] = {}
    if scale:
        for uid in long_df["unique_id"].unique():
            tr_series = train_df.loc[train_df["unique_id"] == uid, "y"]
            mean = float(tr_series.mean())
            std = float(tr_series.std())
            if std == 0 or np.isnan(std):
                std = 1.0
            scalers[str(uid)] = (mean, std)

            # Apply scaling
            train_df.loc[train_df["unique_id"] == uid, "y"] = (train_df.loc[train_df["unique_id"] == uid, "y"] - mean) / std
            val_df.loc[val_df["unique_id"] == uid, "y"] = (val_df.loc[val_df["unique_id"] == uid, "y"] - mean) / std
            test_df.loc[test_df["unique_id"] == uid, "y"] = (test_df.loc[test_df["unique_id"] == uid, "y"] - mean) / std
            long_df.loc[long_df["unique_id"] == uid, "y"] = (long_df.loc[long_df["unique_id"] == uid, "y"] - mean) / std

    return LongBenchmarkData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        full_df=long_df,
        freq=freq,
        target_col=target_col,
        scalers=scalers,
    )


