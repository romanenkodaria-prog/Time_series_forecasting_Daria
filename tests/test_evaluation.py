"""Forecast alignment must be complete and one-to-one."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sales_forecasting.evaluation import align_forecasts, evaluate_forecasts


@pytest.fixture
def tables():
    actual = pd.DataFrame(
        {
            "unique_id": ["a", "b", "a"],
            "ds": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "y": [0, 10, 20],
        }
    )
    forecast = actual.rename(columns={"y": "y_pred"}).copy()
    forecast["y_pred"] = [3, 12, 16]
    return actual, forecast


def test_alignment_ignores_row_order(tables):
    actual, forecast = tables
    original = forecast.copy(deep=True)
    score = evaluate_forecasts(actual, forecast.iloc[::-1])
    assert score == pytest.approx({"MAE": 3, "RMSE": (29 / 3) ** 0.5, "MAPE": 20})
    assert_frame_equal(forecast, original)
    assert align_forecasts(actual, forecast).columns.tolist() == [
        "unique_id",
        "ds",
        "y",
        "y_pred",
    ]


def test_missing_forecasts(tables):
    actual, forecast = tables
    with pytest.raises(ValueError, match="пропущено прогнозов 1, лишних 0"):
        evaluate_forecasts(actual, forecast.iloc[1:])


def test_extra_forecasts(tables):
    actual, forecast = tables
    extra = forecast.iloc[[0]].assign(unique_id="unknown")
    with pytest.raises(ValueError, match="пропущено прогнозов 0, лишних 1"):
        evaluate_forecasts(actual, pd.concat([forecast, extra]))


def test_wrong_keys_same_count(tables):
    actual, forecast = tables
    forecast.loc[0, "ds"] = "2024-01-03"
    with pytest.raises(ValueError, match="пропущено прогнозов 1, лишних 1"):
        evaluate_forecasts(actual, forecast)


@pytest.mark.parametrize("side", ["actual", "forecast"])
def test_duplicates(tables, side):
    actual, forecast = tables
    if side == "actual":
        actual = pd.concat([actual, actual.iloc[[0]]])
    else:
        forecast = pd.concat([forecast, forecast.iloc[[0]]])
    with pytest.raises(ValueError, match="Дубликаты"):
        evaluate_forecasts(actual, forecast)


@pytest.mark.parametrize(
    "column,value",
    [
        ("unique_id", None),
        ("ds", "bad"),
        ("y_pred", None),
        ("y_pred", float("inf")),
        ("y_pred", "bad"),
        ("y_pred", True),
    ],
)
def test_invalid_forecasts(tables, column, value):
    actual, forecast = tables
    forecast[column] = forecast[column].astype(object)
    forecast.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate_forecasts(actual, forecast)


def test_missing_forecast_column(tables):
    actual, forecast = tables
    with pytest.raises(ValueError, match="y_pred"):
        evaluate_forecasts(actual, forecast.drop(columns="y_pred"))


def test_zero_only_evaluation(tables):
    actual, forecast = tables
    actual["y"] = 0
    with pytest.raises(ValueError, match="MAPE не определена"):
        evaluate_forecasts(actual, forecast)


def test_negative_prediction_allowed(tables):
    actual, forecast = tables
    forecast["y_pred"] = -1
    assert evaluate_forecasts(actual, forecast)["MAE"] == 11


def test_normalized_dates_and_identifiers(tables):
    actual, forecast = tables
    actual["unique_id"] = [1, 2, 1]
    forecast["unique_id"] = ["1", "2", "1"]
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    assert len(align_forecasts(actual, forecast)) == 3
