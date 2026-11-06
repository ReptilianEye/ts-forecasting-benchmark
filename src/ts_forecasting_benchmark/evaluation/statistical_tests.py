from __future__ import annotations

from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from dieboldmariano import ZeroVarianceException, dm_test
from matplotlib.axes import Axes
from scipy import stats


def _paired_samples(first, second) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or second.shape != first.shape or first.size < 2:
        raise ValueError("Expected two matching 1D samples with at least two observations")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Samples must contain only finite values")
    return first, second


def diebold_mariano_test(y_true, y_pred1, y_pred2, h: int = 1, power: int = 2) -> dict[str, float]:
    actual, pred1 = _paired_samples(y_true, y_pred1)
    _, pred2 = _paired_samples(actual, y_pred2)
    n = actual.size
    if isinstance(h, bool) or not isinstance(h, Integral) or not 1 <= h < n:
        raise ValueError("h must be an integer between 1 and sample size minus 1")
    if power not in (1, 2):
        raise ValueError("power must be 1 (absolute error) or 2 (squared error)")

    differences = np.abs(actual - pred1) ** power - np.abs(actual - pred2) ** power
    if np.all(differences == 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    try:
        statistic, pvalue = dm_test(
            actual,
            pred1,
            pred2,
            loss=lambda actual, predicted: abs(actual - predicted) ** power,
            h=h,
            one_sided=False,
            harvey_correction=True,
            variance_estimator="bartlett",
        )
    except ZeroVarianceException as exc:
        raise ValueError("DM test is undefined for zero loss-difference variance") from exc
    return {"statistic": float(statistic), "pvalue": float(pvalue)}


def wilcoxon_test(losses_model1, losses_model2) -> dict[str, float]:
    first, second = _paired_samples(losses_model1, losses_model2)
    if (first < 0).any() or (second < 0).any():
        raise ValueError("Pass nonnegative losses, such as absolute or squared errors")
    differences = first - second
    if np.all(differences == 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    result = stats.wilcoxon(differences, alternative="two-sided", method="auto")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def friedman_nemenyi_test(scores: pd.DataFrame) -> dict:
    if not isinstance(scores, pd.DataFrame) or scores.shape[0] < 2 or scores.shape[1] < 3:
        raise ValueError("scores must be a DataFrame with at least 2 tasks and 3 models")
    if not scores.columns.is_unique or not scores.index.is_unique:
        raise ValueError("Task and model labels must be unique")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    mean_ranks = scores.astype(float).rank(axis=1, method="average").mean()
    if np.all(values == values[:, :1]):
        statistic, pvalue = 0.0, 1.0
        posthoc = pd.DataFrame(1.0, index=scores.columns, columns=scores.columns)
    else:
        result = stats.friedmanchisquare(*values.T)
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        posthoc = sp.posthoc_nemenyi_friedman(values)
        posthoc.index = scores.columns
        posthoc.columns = scores.columns
    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "mean_ranks": mean_ranks,
        "nemenyi_pvalues": posthoc,
    }


def plot_critical_difference(scores: pd.DataFrame, alpha: float = 0.05, ax: Axes | None = None) -> Axes:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    result = friedman_nemenyi_test(scores)
    pvalues = result["nemenyi_pvalues"].copy()
    if result["pvalue"] >= alpha:
        pvalues.loc[:, :] = 1.0
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    sp.critical_difference_diagram(result["mean_ranks"], pvalues, alpha=alpha, ax=ax)
    return ax
