"""Model factories and wrappers for benchmark models.

Paradigms:
1. Statistical Baselines: AutoARIMA, SeasonalNaive (statsforecast)
2. MLP / Linear (No Attention): NBEATS, NHITS, DLinear (neuralforecast)
3. Attention Transformers: PatchTST, iTransformer (neuralforecast)
4. Foundation Models: Chronos-Bolt (Zero-Shot via chronos)
"""

# TODO(models): Implement modular model registry/factory in src/ts_forecasting_benchmark/models/
#   - statistical.py: get_statistical_models(freq, season_length)
#   - neural.py: get_neural_models(h, input_size, accelerator)
#   - foundation.py: get_chronos_pipeline(model_id, device)
