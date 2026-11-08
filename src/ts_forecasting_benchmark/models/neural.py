"""NeuralForecast models used in the benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from neuralforecast.models import NBEATS, NHITS, DLinear, PatchTST, iTransformer

NEURAL_MODEL_SPECS = [
    ("NHITS", "N-HiTS", NHITS),
    ("DLinear", "DLinear", DLinear),
    ("NBEATS", "N-BEATS", NBEATS),
    ("PatchTST", "PatchTST", PatchTST),
    ("iTransformer", "iTransformer", iTransformer),
]


def load_best_params(best_params_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load tuned model hyperparameters from a YAML file produced by tune_optuna.py."""
    if best_params_path is None:
        return {}
    path = Path(best_params_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models_section = raw.get("models", {})
    out: dict[str, dict[str, Any]] = {}
    for key, entry in models_section.items():
        if isinstance(entry, dict) and "params" in entry and isinstance(entry["params"], dict):
            out[key] = dict(entry["params"])
        elif isinstance(entry, dict):
            out[key] = dict(entry)
    return out


def get_neural_models(
    h: int = 96,
    input_size: int = 96,
    max_steps: int = 50,
    n_series: int = 1,
    accelerator: str = "cpu",
    best_params_path: str | Path | None = None,
    model_names: Sequence[str] | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> list[Any]:
    """Create linear, MLP, and attention models with shared or Optuna-tuned settings."""
    tuned_by_model = load_best_params(best_params_path)
    requested = (
        {m.lower() for m in model_names}
        if model_names is not None
        else None
    )

    models: list[Any] = []
    for key, alias, cls in NEURAL_MODEL_SPECS:
        if requested is not None and key.lower() not in requested and alias.lower() not in requested:
            continue

        params: dict[str, Any] = {
            "h": h,
            "input_size": input_size,
            "max_steps": max_steps,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "random_seed": seed,
            "accelerator": accelerator,
            "devices": 1,
        }

        # Apply Optuna-tuned hyperparameters if available for this model
        tuned = tuned_by_model.get(key) or tuned_by_model.get(alias) or {}
        for k, v in tuned.items():
            if k in {"h", "max_steps", "random_seed", "accelerator", "devices", "n_series"}:
                continue
            params[k] = v

        if quiet:
            params.update(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )

        if key == "iTransformer":
            params["n_series"] = n_series
        if alias != key:
            params["alias"] = alias

        models.append(cls(**params))

    return models

