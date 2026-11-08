#!/usr/bin/env python3
"""Full LTSF Benchmark Runner (Issue #4).

Runs multi-dataset, multi-horizon rolling-window evaluation across four model paradigms:
1. Statistical Baselines (StatsForecast): SeasonalNaive, AutoARIMA
2. Deep Learning MLP / Linear (NeuralForecast): N-HiTS, DLinear, N-BEATS
3. Deep Learning Attention / Transformers (NeuralForecast): PatchTST, iTransformer
4. Foundation Model Zero-Shot (Chronos): Chronos-Bolt

Datasets: ETTh1, ETTm1, Weather, Electricity
Horizons: H in {96, 192, 336, 720}

Outputs saved to results/metrics/:
- full_benchmark_results.csv: Aggregated MSE, MAE, RMSE, std, fit/inference/total time per (Dataset, Horizon, Model)
- window_metrics.csv: Per-window MSE, MAE, RMSE for statistical tests (Wilcoxon, Diebold-Mariano)
- residuals/residuals_<Dataset>_H<Horizon>.csv: Raw step-level y_true, y_pred, and residuals
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

from ts_forecasting_benchmark.data import (
    SEASON_LENGTHS,
    SPLIT_CONFIGS,
    RollingWindow,
    generate_rolling_windows,
    load_long_benchmark,
)
from ts_forecasting_benchmark.evaluation import compute_metrics
from ts_forecasting_benchmark.models.neural import get_neural_models
from ts_forecasting_benchmark.utils import get_device, set_seed

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

CANONICAL_DATASETS = ["ETTh1", "ETTm1", "Weather", "Electricity"]
CANONICAL_HORIZONS = [96, 192, 336, 720]
ALL_MODELS = [
    "SeasonalNaive",
    "AutoARIMA",
    "N-HiTS",
    "DLinear",
    "N-BEATS",
    "PatchTST",
    "iTransformer",
    "Chronos-Bolt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full LTSF benchmark runner with rolling windows")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=CANONICAL_DATASETS,
        help="Benchmark datasets to evaluate",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=CANONICAL_HORIZONS,
        help="Forecast horizons H",
    )
    parser.add_argument("--target", type=str, default="OT", help="Target column name")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Models to include in the benchmark",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Rolling window step size (defaults to horizon H for non-overlapping windows)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum rolling windows per (dataset, horizon). Use 0 for all available windows.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
        help="Default lookback window (L) when not overridden by best_params.yaml",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Training steps per neural model",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=Path("configs/best_params.yaml"),
        help="Path to Optuna tuned hyperparameters YAML",
    )
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "mps", "cuda"],
        default=get_device().type,
        help="Hardware accelerator for neural and foundation models",
    )
    parser.add_argument(
        "--chronos-model",
        type=str,
        default="amazon/chronos-bolt-mini",
        help="HuggingFace model ID for Chronos-Bolt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metrics"),
        help="Directory to store benchmark metrics and residuals",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (dataset, horizon, model) combinations already present in full_benchmark_results.csv",
    )
    return parser.parse_args()


def _resolve_season_length(dataset_name: str, freq: str) -> int:
    cfg_lookup = {k.lower(): v for k, v in SPLIT_CONFIGS.items()}
    cfg = cfg_lookup.get(dataset_name.lower(), {})
    if "season_length" in cfg:
        return int(cfg["season_length"])
    return int(SEASON_LENGTHS.get(freq, 24))


def _evaluate_window_predictions(
    dataset: str,
    horizon: int,
    model_name: str,
    paradigm: str,
    zero_shot: bool,
    windows: Sequence[RollingWindow],
    window_preds: Sequence[np.ndarray],
    fit_time_s: float,
    infer_time_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute window-level metrics, step-level residuals, and overall summary row."""
    window_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    for win, y_pred in zip(windows, window_preds, strict=True):
        y_true = win.target_df["y"].to_numpy(dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)[:horizon]
        if len(pred) != len(y_true):
            raise ValueError(
                f"Prediction length mismatch for {model_name} on window {win.window_idx}: "
                f"expected {len(y_true)}, got {len(pred)}"
            )

        w_metrics = compute_metrics(y_true, pred)
        window_rows.append(
            {
                "Dataset": dataset,
                "Horizon": horizon,
                "Window_Idx": win.window_idx,
                "Cutoff_ds": str(win.cutoff_ds),
                "Model": model_name,
                "Paradigm": paradigm,
                "MSE": w_metrics["mse"],
                "MAE": w_metrics["mae"],
                "RMSE": w_metrics["rmse"],
            }
        )

        residuals = y_true - pred
        sq_errors = residuals**2
        abs_errors = np.abs(residuals)
        ds_values = win.target_df["ds"].astype(str).to_numpy()

        for step_idx in range(horizon):
            residual_rows.append(
                {
                    "Dataset": dataset,
                    "Horizon": horizon,
                    "Window_Idx": win.window_idx,
                    "Step": step_idx + 1,
                    "ds": ds_values[step_idx],
                    "Model": model_name,
                    "y_true": float(y_true[step_idx]),
                    "y_pred": float(pred[step_idx]),
                    "residual": float(residuals[step_idx]),
                    "sq_error": float(sq_errors[step_idx]),
                    "abs_error": float(abs_errors[step_idx]),
                }
            )

        all_y_true.append(y_true)
        all_y_pred.append(pred)

    concat_true = np.concatenate(all_y_true)
    concat_pred = np.concatenate(all_y_pred)
    overall = compute_metrics(concat_true, concat_pred)

    win_mses = [r["MSE"] for r in window_rows]
    win_maes = [r["MAE"] for r in window_rows]

    summary_row = {
        "Dataset": dataset,
        "Horizon": horizon,
        "Model": model_name,
        "Paradigm": paradigm,
        "Zero-Shot": zero_shot,
        "Num_Windows": len(windows),
        "MSE": overall["mse"],
        "MAE": overall["mae"],
        "RMSE": overall["rmse"],
        "MSE_Std": float(np.std(win_mses, ddof=1)) if len(win_mses) > 1 else 0.0,
        "MAE_Std": float(np.std(win_maes, ddof=1)) if len(win_maes) > 1 else 0.0,
        "Fit_Time_s": round(fit_time_s, 3),
        "Infer_Time_s": round(infer_time_s, 3),
        "Time (s)": round(fit_time_s + infer_time_s, 3),
    }
    return summary_row, window_rows, residual_rows


def run_statistical_models(
    dataset: str,
    horizon: int,
    freq: str,
    season_length: int,
    windows: Sequence[RollingWindow],
    requested_models: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, win_rows, res_rows = [], [], []

    # 1. SeasonalNaive across rolling windows (batched across all windows)
    if "seasonalnaive" in requested_models:
        t0 = time.time()
        ctx_limit = max(season_length * 3, 256)
        batched_dfs = []
        for win in windows:
            ctx = win.context_df.tail(ctx_limit).copy()
            ctx["unique_id"] = f"win_{win.window_idx:04d}"
            batched_dfs.append(ctx)
        batched_ctx = pd.concat(batched_dfs, ignore_index=True)
        sf = StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq=freq, n_jobs=1)
        sf.fit(df=batched_ctx)
        pred_df = sf.predict(h=horizon)
        sn_preds = [
            pred_df.loc[
                pred_df["unique_id"] == f"win_{win.window_idx:04d}", "SeasonalNaive"
            ].to_numpy()
            for win in windows
        ]
        sn_time = time.time() - t0
        s_row, w_r, r_r = _evaluate_window_predictions(
            dataset=dataset,
            horizon=horizon,
            model_name="SeasonalNaive",
            paradigm="Statistical",
            zero_shot=False,
            windows=windows,
            window_preds=sn_preds,
            fit_time_s=0.0,
            infer_time_s=sn_time,
        )
        summaries.append(s_row)
        win_rows.extend(w_r)
        res_rows.extend(r_r)
        print(
            f"    ✅ {'SeasonalNaive':15s} -> MSE: {s_row['MSE']:.4f} | MAE: {s_row['MAE']:.4f} ({sn_time:.2f}s)"
        )

    # 2. AutoARIMA across rolling windows (parallelized across windows via StatsForecast)
    if "autoarima" in requested_models:
        arima_season = season_length if season_length <= 24 else 1
        arima_ctx_limit = 512

        # Fit on first window context to measure single-series parameter estimation time
        t_fit0 = time.time()
        first_ctx = windows[0].context_df.tail(arima_ctx_limit)
        sf_first = StatsForecast(
            models=[
                AutoARIMA(
                    season_length=arima_season,
                    max_p=2,
                    max_q=2,
                    max_P=1,
                    max_Q=1,
                    approximation=True,
                    stepwise=True,
                )
            ],
            freq=freq,
            n_jobs=1,
        )
        sf_first.fit(df=first_ctx)
        first_pred = sf_first.predict(h=horizon)["AutoARIMA"].to_numpy()
        fit_time = time.time() - t_fit0

        t_inf0 = time.time()
        arima_preds: list[np.ndarray] = [first_pred]
        if len(windows) > 1:
            batched_dfs = []
            for win in windows[1:]:
                ctx = win.context_df.tail(arima_ctx_limit).copy()
                ctx["unique_id"] = f"win_{win.window_idx:04d}"
                batched_dfs.append(ctx)
            batched_ctx = pd.concat(batched_dfs, ignore_index=True)
            sf = StatsForecast(
                models=[
                    AutoARIMA(
                        season_length=arima_season,
                        max_p=2,
                        max_q=2,
                        max_P=1,
                        max_Q=1,
                        approximation=True,
                        stepwise=True,
                    )
                ],
                freq=freq,
                n_jobs=-1,
            )
            sf.fit(df=batched_ctx)
            pred_df = sf.predict(h=horizon)
            for win in windows[1:]:
                uid = f"win_{win.window_idx:04d}"
                arima_preds.append(pred_df.loc[pred_df["unique_id"] == uid, "AutoARIMA"].to_numpy())
        infer_time = time.time() - t_inf0

        s_row, w_r, r_r = _evaluate_window_predictions(
            dataset=dataset,
            horizon=horizon,
            model_name="AutoARIMA",
            paradigm="Statistical",
            zero_shot=False,
            windows=windows,
            window_preds=arima_preds,
            fit_time_s=fit_time,
            infer_time_s=infer_time,
        )
        summaries.append(s_row)
        win_rows.extend(w_r)
        res_rows.extend(r_r)
        print(
            f"    ✅ {'AutoARIMA':15s} -> MSE: {s_row['MSE']:.4f} | MAE: {s_row['MAE']:.4f} "
            f"(fit {fit_time:.2f}s + roll {infer_time:.2f}s)"
        )

    return summaries, win_rows, res_rows


def run_neural_models_for_horizon(
    dataset: str,
    horizon: int,
    freq: str,
    train_val_df: pd.DataFrame,
    windows: Sequence[RollingWindow],
    args: argparse.Namespace,
    requested_models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from neuralforecast import NeuralForecast

    summaries, win_rows, res_rows = [], [], []
    n_series = int(train_val_df["unique_id"].nunique())

    models = get_neural_models(
        h=horizon,
        input_size=args.input_size,
        max_steps=args.max_steps,
        n_series=n_series,
        accelerator=args.accelerator,
        best_params_path=args.best_params,
        model_names=requested_models,
        seed=args.seed,
        quiet=True,
    )

    for model in models:
        name = str(model)
        lookback = int(getattr(model, "input_size", args.input_size))
        paradigm = (
            "Deep Learning (Attention)"
            if name in {"PatchTST", "iTransformer"}
            else "Deep Learning (MLP / Linear)"
        )
        try:
            # 1. Fit once on train+val history before test split
            t_fit0 = time.time()
            nf = NeuralForecast(models=[model], freq=freq)
            nf.fit(df=train_val_df, val_size=horizon)
            fit_time = time.time() - t_fit0

            # 2. Rolling inference across test windows
            t_inf0 = time.time()
            win_preds: list[np.ndarray] = []

            if name != "iTransformer" and n_series == 1:
                # Batch all rolling windows as separate series IDs in a single predict() call
                batched_dfs = []
                for win in windows:
                    ctx_slice = win.context_df.tail(lookback).copy()
                    ctx_slice["unique_id"] = f"win_{win.window_idx:04d}"
                    batched_dfs.append(ctx_slice)
                batched_ctx = pd.concat(batched_dfs, ignore_index=True)
                preds_df = nf.predict(df=batched_ctx)
                for win in windows:
                    uid = f"win_{win.window_idx:04d}"
                    w_pred = preds_df.loc[preds_df["unique_id"] == uid, name].to_numpy()
                    win_preds.append(w_pred)
            else:
                for win in windows:
                    ctx_slice = win.context_df.tail(lookback)
                    preds_df = nf.predict(df=ctx_slice)
                    win_preds.append(preds_df[name].to_numpy())

            infer_time = time.time() - t_inf0

            s_row, w_r, r_r = _evaluate_window_predictions(
                dataset=dataset,
                horizon=horizon,
                model_name=name,
                paradigm=paradigm,
                zero_shot=False,
                windows=windows,
                window_preds=win_preds,
                fit_time_s=fit_time,
                infer_time_s=infer_time,
            )
            summaries.append(s_row)
            win_rows.extend(w_r)
            res_rows.extend(r_r)
            print(
                f"    ✅ {name:15s} -> MSE: {s_row['MSE']:.4f} | MAE: {s_row['MAE']:.4f} "
                f"(fit {fit_time:.2f}s + infer {infer_time:.2f}s, L={lookback})"
            )
        except Exception as exc:
            print(f"    ❌ Error running {name} on {dataset} H={horizon}: {exc}")

    return summaries, win_rows, res_rows


def run_chronos_for_horizon(
    pipeline: Any,
    dataset: str,
    horizon: int,
    windows: Sequence[RollingWindow],
    context_len: int = 512,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    t0 = time.time()
    min_ctx = min(len(w.context_df) for w in windows)
    eff_ctx = min(context_len, min_ctx)

    ctx_batch = np.stack(
        [w.context_df["y"].to_numpy(dtype=np.float32)[-eff_ctx:] for w in windows],
        axis=0,
    )
    ctx_tensor = torch.tensor(ctx_batch, dtype=torch.float32)

    pred_quantiles = pipeline.predict(ctx_tensor, prediction_length=horizon)
    if hasattr(pred_quantiles, "cpu"):
        pred_quantiles = pred_quantiles.cpu().numpy()
    else:
        pred_quantiles = np.asarray(pred_quantiles)

    # Index 4 corresponds to the 0.50 median quantile
    median_preds = [pred_quantiles[i, 4, :horizon] for i in range(len(windows))]
    infer_time = time.time() - t0

    s_row, w_r, r_r = _evaluate_window_predictions(
        dataset=dataset,
        horizon=horizon,
        model_name="Chronos-Bolt",
        paradigm="Foundation Model (LLM)",
        zero_shot=True,
        windows=windows,
        window_preds=median_preds,
        fit_time_s=0.0,
        infer_time_s=infer_time,
    )
    print(
        f"    ✅ {'Chronos-Bolt':15s} -> MSE: {s_row['MSE']:.4f} | MAE: {s_row['MAE']:.4f} "
        f"(zero-shot batch in {infer_time:.2f}s, L={eff_ctx})"
    )
    return s_row, w_r, r_r


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir: Path = args.output_dir
    residuals_dir = output_dir / "residuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "full_benchmark_results.csv"
    window_csv = output_dir / "window_metrics.csv"

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, int, str]] = set()

    if args.resume and summary_csv.exists():
        existing_df = pd.read_csv(summary_csv)
        summary_rows = existing_df.to_dict("records")
        for r in summary_rows:
            completed_keys.add((str(r["Dataset"]), int(r["Horizon"]), str(r["Model"]).lower()))
        if window_csv.exists():
            window_rows = pd.read_csv(window_csv).to_dict("records")
        print(
            f"Resuming benchmark with {len(completed_keys)} completed (dataset, horizon, model) entries."
        )

    requested_lower = {m.lower() for m in args.models}

    print("=" * 80)
    print("🚀 STARTING FULL LTSF ROLLING-WINDOW BENCHMARK (ISSUE #4)")
    print(f"Datasets:    {args.datasets}")
    print(f"Horizons:    {args.horizons}")
    print(f"Models:      {args.models}")
    print(
        f"Max Windows: {args.max_windows if args.max_windows > 0 else 'unlimited'} | Step Size: {args.step_size or 'H (non-overlapping)'}"
    )
    print(f"Best Params: {args.best_params} (exists={args.best_params.exists()})")
    print(f"Accelerator: {args.accelerator}")
    print("=" * 80)

    # Pre-load Chronos-Bolt pipeline once if requested
    chronos_pipeline = None
    if "chronos-bolt" in requested_lower:
        try:
            from chronos import ChronosBoltPipeline

            print(f"\n🔮 Pre-loading Chronos-Bolt ({args.chronos_model}) on {args.accelerator}...")
            chronos_pipeline = ChronosBoltPipeline.from_pretrained(
                args.chronos_model,
                device_map=args.accelerator,
                torch_dtype=torch.float32,
            )
        except Exception as exc:
            print(f"  ⚠️ Could not load Chronos-Bolt pipeline: {exc}")

    for dataset in args.datasets:
        print("\n" + "-" * 80)
        print(f"📂 DATASET: {dataset} (Target: {args.target})")
        data = load_long_benchmark(dataset, target_col=args.target, scale=True)
        season_len = _resolve_season_length(dataset, data.freq)
        train_val_df = (
            pd.concat([data.train_df, data.val_df], ignore_index=True)
            .sort_values("ds")
            .reset_index(drop=True)
        )
        print(
            f"   Train: {len(data.train_df):,} | Val: {len(data.val_df):,} | Test: {len(data.test_df):,} "
            f"| Freq: {data.freq} | Season: {season_len}"
        )

        for horizon in args.horizons:
            max_win = args.max_windows if args.max_windows and args.max_windows > 0 else None
            windows = generate_rolling_windows(
                data,
                horizon=horizon,
                step_size=args.step_size,
                max_windows=max_win,
            )
            print(
                f"\n  ⏱️  Horizon H={horizon} | Rolling Windows: K={len(windows)} (stride={args.step_size or horizon})"
            )

            pending_models = [
                m for m in args.models if (dataset, horizon, m.lower()) not in completed_keys
            ]
            if not pending_models:
                print(f"    ⏭️  All models already completed for {dataset} H={horizon}, skipping.")
                continue

            pending_lower = {m.lower() for m in pending_models}
            horizon_residuals: list[dict[str, Any]] = []
            res_file = residuals_dir / f"residuals_{dataset}_H{horizon}.csv"
            if args.resume and res_file.exists():
                horizon_residuals = pd.read_csv(res_file).to_dict("records")

            # 1. Statistical models
            stat_s, stat_w, stat_r = run_statistical_models(
                dataset=dataset,
                horizon=horizon,
                freq=data.freq,
                season_length=season_len,
                windows=windows,
                requested_models=pending_lower,
            )
            summary_rows.extend(stat_s)
            window_rows.extend(stat_w)
            horizon_residuals.extend(stat_r)

            # 2. NeuralForecast models
            s_n, w_n, r_n = run_neural_models_for_horizon(
                dataset=dataset,
                horizon=horizon,
                freq=data.freq,
                train_val_df=train_val_df,
                windows=windows,
                args=args,
                requested_models=pending_models,
            )
            summary_rows.extend(s_n)
            window_rows.extend(w_n)
            horizon_residuals.extend(r_n)

            # 3. Chronos-Bolt (Zero-Shot)
            if "chronos-bolt" in pending_lower and chronos_pipeline is not None:
                try:
                    c_s, c_w, c_r = run_chronos_for_horizon(
                        pipeline=chronos_pipeline,
                        dataset=dataset,
                        horizon=horizon,
                        windows=windows,
                    )
                    summary_rows.append(c_s)
                    window_rows.extend(c_w)
                    horizon_residuals.extend(c_r)
                except Exception as exc:
                    print(f"    ❌ Error running Chronos-Bolt on {dataset} H={horizon}: {exc}")

            # Checkpoint results after each (dataset, horizon)
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
            pd.DataFrame(window_rows).to_csv(window_csv, index=False)
            if horizon_residuals:
                pd.DataFrame(horizon_residuals).to_csv(res_file, index=False)

    print("\n" + "=" * 80)
    print("🏆 FULL LTSF BENCHMARK COMPLETED")
    print("=" * 80)
    final_df = pd.DataFrame(summary_rows)
    if not final_df.empty:
        display_cols = [
            "Dataset",
            "Horizon",
            "Model",
            "Paradigm",
            "Num_Windows",
            "MSE",
            "MAE",
            "Time (s)",
        ]
        print(final_df[display_cols].to_string(index=False))
    print("-" * 80)
    print(f"Saved aggregated metrics to: {summary_csv}")
    print(f"Saved window metrics to:     {window_csv}")
    print(f"Saved step residuals under:  {residuals_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
