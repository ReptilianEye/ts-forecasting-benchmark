"""NeuralForecast models used in the benchmark."""

from neuralforecast.models import NBEATS, NHITS, DLinear, PatchTST, iTransformer


def get_neural_models(
    h: int = 96,
    input_size: int = 96,
    max_steps: int = 50,
    n_series: int = 1,
    accelerator: str = "cpu",
):
    """Create linear, MLP, and attention models with shared pilot settings."""
    params = {
        "h": h,
        "input_size": input_size,
        "max_steps": max_steps,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "accelerator": accelerator,
        "devices": 1,
    }
    return [
        NHITS(alias="N-HiTS", **params),
        DLinear(**params),
        NBEATS(alias="N-BEATS", **params),
        PatchTST(**params),
        iTransformer(n_series=n_series, **params),
    ]
