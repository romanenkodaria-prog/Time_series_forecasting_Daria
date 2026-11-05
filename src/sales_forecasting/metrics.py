"""Positional metrics; use evaluation for key-based table alignment."""

from math import fsum, hypot, isfinite, sqrt

import pandas as pd


def _pairs(y_true, y_pred) -> list[tuple[float, float]]:
    try:
        actual = pd.Series(y_true).reset_index(drop=True)
        predicted = pd.Series(y_pred).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Метрики ожидают одномерные последовательности чисел."
        ) from exc
    if actual.empty or len(actual) != len(predicted):
        raise ValueError("Факт и прогноз должны иметь одинаковую ненулевую длину.")
    result = []
    for a, p in zip(actual, predicted):
        if isinstance(a, (bool, complex)) or isinstance(p, (bool, complex)):
            raise ValueError("Метрики ожидают конечные числа.")
        try:
            a, p = float(a), float(p)
        except (TypeError, ValueError) as exc:
            raise ValueError("Метрики ожидают конечные числа.") from exc
        if not isfinite(a) or not isfinite(p):
            raise ValueError("Метрики ожидают конечные числа.")
        result.append((a, p))
    return result


def mae(y_true, y_pred) -> float:
    """Mean absolute error in sales units (positional alignment)."""
    pairs = _pairs(y_true, y_pred)
    return fsum(abs(a - p) / len(pairs) for a, p in pairs)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error in sales units."""
    pairs = _pairs(y_true, y_pred)
    return hypot(*(a - p for a, p in pairs)) / sqrt(len(pairs))


def mape(y_true, y_pred) -> float:
    """Percentage error excluding zero actuals; all-zero actuals raise ValueError.

    Zeros still contribute to MAE and RMSE. No epsilon or silent zero score
    is substituted for an undefined percentage error.
    """
    pairs = [(a, p) for a, p in _pairs(y_true, y_pred) if a != 0]
    if not pairs:
        raise ValueError("MAPE не определена: все фактические значения равны нулю.")
    return 100 * fsum(abs((a - p) / a) / len(pairs) for a, p in pairs)
