from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NBEATS, NHITS, DLinear, PatchTST, iTransformer
from pytorch_lightning.callbacks import Callback

from ts_forecasting_benchmark.data import load_long_benchmark
from ts_forecasting_benchmark.utils import get_device, set_seed

MODELS = {
    "NHITS": NHITS,
    "DLinear": DLinear,
    "NBEATS": NBEATS,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
}


class ValidationPruner(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = float(trainer.callback_metrics["ptl/val_loss"])
        if not np.isfinite(value):
            raise optuna.TrialPruned("Non-finite validation MSE")
        self.trial.report(value, step=trainer.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def suggest_params(trial, model_name):
    params = {
        "input_size": trial.suggest_categorical("input_size", [96, 192, 336, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    params["windows_batch_size"] = params["batch_size"]
    if model_name in {"NHITS", "NBEATS"}:
        params["n_blocks"] = [trial.suggest_int("blocks_per_stack", 1, 3)] * 3
        if model_name == "NHITS":
            params["dropout_prob_theta"] = trial.suggest_float("dropout_prob_theta", 0.0, 0.3)
    elif model_name == "DLinear":
        params["moving_avg_window"] = trial.suggest_categorical("moving_avg_window", [13, 25, 49])
    elif model_name == "PatchTST":
        params["encoder_layers"] = trial.suggest_int("encoder_layers", 1, 3)
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    elif model_name == "iTransformer":
        params["n_series"] = 1
        params["e_layers"] = trial.suggest_int("e_layers", 1, 3)
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    return params


def objective(trial, model_name, frame, val_size, freq, args):
    params = suggest_params(trial, model_name)
    params.update(h=args.horizon, max_steps=args.max_steps, random_seed=args.seed)
    trial.set_user_attr("model_params", params)
    model = MODELS[model_name](
        **params,
        loss=MSE(),
        valid_loss=MSE(),
        val_check_steps=min(args.val_check_steps, args.max_steps),
        inference_windows_batch_size=128,
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.models[0].trainer_kwargs["callbacks"] = [ValidationPruner(trial)]
    nf.fit(df=frame, val_size=val_size)
    return float(nf.models[0].valid_trajectories[-1][1])


def parse_args():
    parser = argparse.ArgumentParser(description="Tune neural models on validation MSE with Optuna")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=["cpu", "mps", "cuda"], default=get_device().type)
    parser.add_argument("--output", type=Path, default=Path("configs/best_params.yaml"))
    args = parser.parse_args()
    if min(args.horizon, args.n_trials, args.max_steps, args.val_check_steps) < 1:
        parser.error("horizon, n-trials, max-steps and val-check-steps must be positive")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_long_benchmark(args.dataset, target_col=args.target, scale=True)
    if len(data.val_df) < args.horizon:
        raise ValueError("Validation split must contain at least one forecast horizon")
    if len(data.train_df) < 512 + args.horizon:
        raise ValueError("Training split is too short for the largest input window and horizon")
    frame = pd.concat([data.train_df, data.val_df], ignore_index=True)
    output = {
        "dataset": args.dataset,
        "target": args.target,
        "horizon": args.horizon,
        "loss": "MSE",
        "seed": args.seed,
        "models": {},
    }
    for model_name in dict.fromkeys(args.models):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=2 * min(args.val_check_steps, args.max_steps),
            ),
        )
        study.optimize(
            partial(
                objective,
                model_name=model_name,
                frame=frame,
                val_size=len(data.val_df),
                freq=data.freq,
                args=args,
            ),
            n_trials=args.n_trials,
            gc_after_trial=True,
        )
        completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if not completed:
            raise RuntimeError(f"No completed trials for {model_name}; no configuration selected")
        best = study.best_trial
        output["models"][model_name] = {
            "validation_mse": best.value,
            "params": best.user_attrs["model_params"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
        print(f"{model_name}: validation MSE={best.value:.6f}; saved to {args.output}")


if __name__ == "__main__":
    main()
