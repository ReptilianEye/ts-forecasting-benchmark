#!/usr/bin/env python3
"""Pilot Benchmark Runner.

Runs a quick pilot evaluation comparing:
1. Statistical Baseline: SeasonalNaive / AutoARIMA (StatsForecast)
2. MLP / Basis Expansion: N-HiTS (NeuralForecast)
3. Foundation Model (Zero-Shot): Chronos-Bolt (Amazon)

Evaluates on ETTh1 (target: OT) for horizon H=96.
Saves metrics to results/metrics/pilot_benchmark_results.csv
and plots comparison to results/figures/pilot_forecasts.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
import torch

from ts_forecasting_benchmark.data import load_long_benchmark
from ts_forecasting_benchmark.evaluation import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot benchmark runner")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="Dataset name")
    parser.add_argument("--target", type=str, default="OT", help="Target column name")
    parser.add_argument("--horizon", type=int, default=96, help="Forecast horizon (H)")
    parser.add_argument("--input-size", type=int, default=96, help="Lookback window (L)")
    parser.add_argument("--nhits-steps", type=int, default=50, help="Max training steps for NHITS pilot")
    parser.add_argument(
        "--chronos-model",
        type=str,
        default="amazon/chronos-bolt-mini",
        help="HuggingFace model id for Chronos-Bolt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 STARTING PILOT TIME SERIES BENCHMARK")
    print(f"Dataset: {args.dataset} (Target: {args.target}) | Horizon H={args.horizon} | Lookback L={args.input_size}")
    print("=" * 70)

    # 1. Load long-format benchmark data
    print("\n[1/5] 📥 Loading dataset in long format (unique_id, ds, y)...")
    data = load_long_benchmark(args.dataset, target_col=args.target, scale=True)
    print(f"  Train samples: {len(data.train_df):,}")
    print(f"  Val samples:   {len(data.val_df):,}")
    print(f"  Test samples:  {len(data.test_df):,}")
    print(f"  Frequency:     {data.freq}")

    # For pilot evaluation, test on the first test window of length H
    # (or right after train + val)
    val_and_test = pd.concat([data.val_df, data.test_df]).sort_values("ds").reset_index(drop=True)
    # Target ground truth for the first horizon of the test split
    ground_truth = data.test_df.iloc[: args.horizon].copy()
    y_true = ground_truth["y"].values

    results = []
    forecasts: dict[str, np.ndarray] = {}

    # 2. Statistical Baselines (StatsForecast)
    print("\n[2/5] 📊 Running Statistical Baseline (SeasonalNaive & AutoARIMA)...")
    t0 = time.time()
    # Fast seasonal naive + AutoARIMA with bounded p,q for pilot speed
    stat_models = [
        SeasonalNaive(season_length=24),
        AutoARIMA(season_length=24, max_p=2, max_q=2, max_P=1, max_Q=1),
    ]
    sf = StatsForecast(models=stat_models, freq=data.freq, n_jobs=1)
    
    # Fit on train data
    sf.fit(df=data.train_df)
    stat_preds = sf.predict(h=args.horizon)
    stat_time = time.time() - t0

    for m_name in ["SeasonalNaive", "AutoARIMA"]:
        if m_name in stat_preds.columns:
            y_pred = stat_preds[m_name].values[: args.horizon]
            metrics = compute_metrics(y_true, y_pred)
            forecasts[m_name] = y_pred
            results.append({
                "Model": m_name,
                "Paradigm": "Statistical",
                "MSE": metrics["mse"],
                "MAE": metrics["mae"],
                "Time (s)": round(stat_time / 2, 2),
                "Zero-Shot": False,
            })
            print(f"  ✅ {m_name:15s} -> MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f}")

    # 3. Deep Learning MLP: N-HiTS (NeuralForecast)
    # TODO(models): Add DLinear, NBEATS, and PatchTST / iTransformer into the NeuralForecast ensemble
    print("\n[3/5] 🧠 Running Deep Learning MLP: N-HiTS (NeuralForecast)...")
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        t0 = time.time()
        # Train on train_df
        nhits_model = NHITS(
            h=args.horizon,
            input_size=args.input_size,
            max_steps=args.nhits_steps,
            batch_size=64,
            learning_rate=1e-3,
        )
        nf = NeuralForecast(models=[nhits_model], freq=data.freq)
        nf.fit(df=data.train_df, val_size=args.horizon)
        nhits_preds = nf.predict()
        nhits_time = time.time() - t0

        if "NHITS" in nhits_preds.columns:
            y_pred = nhits_preds["NHITS"].values[: args.horizon]
            metrics = compute_metrics(y_true, y_pred)
            forecasts["NHITS"] = y_pred
            results.append({
                "Model": "N-HiTS",
                "Paradigm": "Deep Learning (MLP)",
                "MSE": metrics["mse"],
                "MAE": metrics["mae"],
                "Time (s)": round(nhits_time, 2),
                "Zero-Shot": False,
            })
            print(f"  ✅ {'N-HiTS':15s} -> MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} (trained in {nhits_time:.1f}s)")
    except Exception as e:
        print(f"  ❌ Error running N-HiTS: {e}")

    # 4. Foundation Model Zero-Shot: Chronos-Bolt
    print("\n[4/5] 🔮 Running Foundation Model: Chronos-Bolt (Zero-Shot)...")
    try:
        from chronos import ChronosBoltPipeline

        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  Loading Chronos-Bolt pipeline ({args.chronos_model}) on {device}...")
        pipeline = ChronosBoltPipeline.from_pretrained(
            args.chronos_model,
            device_map=device,
            torch_dtype=torch.float32,
        )

        # Context: last `input_size` points from training/val before the test horizon
        context_series = data.train_df["y"].values[-args.input_size :]
        context_tensor = torch.tensor(context_series, dtype=torch.float32).unsqueeze(0)

        # Zero-shot prediction (no training!)
        pred_quantiles = pipeline.predict(context_tensor, prediction_length=args.horizon)
        # Median prediction corresponds to 50th percentile (index 4 out of 9 quantiles: 0.1 .. 0.9)
        if hasattr(pred_quantiles, "cpu"):
            pred_quantiles = pred_quantiles.cpu().numpy()
        else:
            pred_quantiles = np.asarray(pred_quantiles)
        
        chronos_median = pred_quantiles[0, 4, :args.horizon]
        chronos_time = time.time() - t0

        metrics = compute_metrics(y_true, chronos_median)
        forecasts["Chronos-Bolt"] = chronos_median
        results.append({
            "Model": "Chronos-Bolt",
            "Paradigm": "Foundation Model (LLM)",
            "MSE": metrics["mse"],
            "MAE": metrics["mae"],
            "Time (s)": round(chronos_time, 2),
            "Zero-Shot": True,
        })
        print(f"  ✅ {'Chronos-Bolt':15s} -> MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} (Zero-shot in {chronos_time:.1f}s)")
    except Exception as e:
        print(f"  ❌ Error running Chronos-Bolt: {e}")

    # 5. Summary and Plotting
    print("\n[5/5] 📈 Generating Summary Table and Figures...")
    results_df = pd.DataFrame(results)

    # Save metrics
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "pilot_benchmark_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    # Display comparison
    print("\n" + "=" * 70)
    print("🏆 PILOT BENCHMARK RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # Plot forecast comparison
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "pilot_forecasts.png"

    plt.figure(figsize=(12, 6))
    time_steps = np.arange(args.horizon)
    plt.plot(time_steps, y_true, label="Ground Truth (Actual)", color="black", linewidth=2.0)

    colors = {"SeasonalNaive": "#94a3b8", "AutoARIMA": "#3b82f6", "N-HiTS": "#10b981", "Chronos-Bolt": "#8b5cf6"}
    for name, pred in forecasts.items():
        plt.plot(
            time_steps,
            pred,
            label=f"{name}",
            color=colors.get(name, None),
            linestyle="--" if name != "Chronos-Bolt" else "-",
            linewidth=1.7,
        )

    plt.title(f"Pilot Forecast Comparison on {args.dataset} (Target: {args.target}, H={args.horizon})", fontsize=14, pad=12)
    plt.xlabel("Horizon Step (Hours)", fontsize=11)
    plt.ylabel("Standardized Value (z-score)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved forecast plot to: {fig_path}")


if __name__ == "__main__":
    main()
