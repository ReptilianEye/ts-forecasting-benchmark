"""Evaluation metrics and statistical testing."""

from ts_forecasting_benchmark.evaluation.metrics import (
    compute_metrics,
    mae,
    mse,
    rmse,
)

# TODO(evaluation): Implement statistical significance tests in statistical_tests.py:
#   - diebold_mariano_test(y_true, y_pred1, y_pred2, h)
#   - wilcoxon_test(errors_model1, errors_model2)
#   - plot_critical_difference(matrix_of_ranks, model_names)

__all__ = [
    "compute_metrics",
    "mse",
    "mae",
    "rmse",
]
