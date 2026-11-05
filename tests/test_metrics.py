"""Numerical definitions and invalid inputs for metrics."""

import math

import pandas as pd
import pytest

from sales_forecasting.metrics import mae, mape, rmse


def test_known_metrics():
    actual, predicted = [0, 10, 20], [3, 12, 16]
    assert mae(actual, predicted) == pytest.approx(3)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(29 / 3))
    assert mape(actual, predicted) == pytest.approx(20)


@pytest.mark.parametrize("metric", [mae, rmse, mape])
def test_perfect_prediction(metric):
    assert metric([0, 1, 5], [0, 1, 5]) == 0


def test_all_zero_actuals():
    assert mae([0, 0], [1, 3]) == 2
    assert rmse([0, 0], [1, 3]) == pytest.approx(math.sqrt(5))
    with pytest.raises(ValueError, match="все фактические значения равны нулю"):
        mape([0, 0], [1, 3])


def test_series_indices_are_positional():
    assert mae(pd.Series([1, 2], index=[3, 4]), pd.Series([1, 2], index=[5, 6])) == 0


@pytest.mark.parametrize("metric", [mae, rmse, mape])
@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        ([1], [1, 2]),
        ([float("nan")], [1]),
        ([1], [float("inf")]),
        ([None], [1]),
        ([True], [1]),
        ([1], [complex(1, 2)]),
        ([[1, 2]], [[1, 2]]),
        (["bad"], [1]),
    ],
)
def test_invalid_metric_input(metric, actual, predicted):
    with pytest.raises(ValueError):
        metric(actual, predicted)
