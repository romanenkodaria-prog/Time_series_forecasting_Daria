"""Seasonal forecasts on small, known weekly patterns."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sales_forecasting.evaluation import evaluate_forecasts
from sales_forecasting.models.baseline import seasonal_naive


def make_history(days=14, uid="a", start="2024-01-01"):
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.date_range(start, periods=days),
            "y": list(range(days)),
        }
    )


def test_repeats_last_week_for_90_days():
    history = make_history().sample(frac=1, random_state=2)
    original = history.copy(deep=True)
    forecast = seasonal_naive(history)
    assert forecast.columns.tolist() == ["unique_id", "ds", "y_pred"]
    assert forecast["y_pred"].tolist() == [7 + i % 7 for i in range(90)]
    assert forecast["ds"].tolist() == pd.date_range("2024-01-15", periods=90).tolist()
    assert not forecast.duplicated(["unique_id", "ds"]).any()
    assert_frame_equal(history, original)


def test_multiple_series_with_different_end_dates():
    history = pd.concat([make_history(7, "a"), make_history(21, "b", "2024-02-01")])
    forecast = seasonal_naive(history)
    assert forecast.groupby("unique_id").size().to_dict() == {"a": 90, "b": 90}
    assert forecast.loc[forecast["unique_id"] == "a", "y_pred"].tolist() == [
        i % 7 for i in range(90)
    ]
    assert forecast.loc[forecast["unique_id"] == "b", "y_pred"].tolist() == [
        14 + i % 7 for i in range(90)
    ]
    assert forecast.groupby("unique_id")["ds"].min().to_dict() == {
        "a": pd.Timestamp("2024-01-08"),
        "b": pd.Timestamp("2024-02-22"),
    }


def test_forecasts_align_by_keys():
    forecast = seasonal_naive(make_history(7))
    actual = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": pd.date_range("2024-01-08", periods=90),
            "y": [i % 7 for i in range(90)],
        }
    )
    assert evaluate_forecasts(actual.sample(frac=1, random_state=2), forecast) == {
        "MAE": 0,
        "RMSE": 0,
        "MAPE": 0,
    }


def test_one_short_series_rejects_entire_forecast():
    with pytest.raises(ValueError, match="истории ряда 'b'"):
        seasonal_naive(pd.concat([make_history(7), make_history(6, "b")]))


@pytest.mark.parametrize("missing", [2, 10])
def test_missing_date_in_history(missing):
    with pytest.raises(ValueError, match="пропущенные календарные дни"):
        seasonal_naive(make_history(21).drop(index=missing))


def test_duplicate_date():
    history = make_history()
    with pytest.raises(ValueError, match="Дубликаты"):
        seasonal_naive(pd.concat([history, history.iloc[[0]]]))


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "90"])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="horizon"):
        seasonal_naive(make_history(), horizon=horizon)


def test_short_horizon_and_numeric_strings():
    history = make_history(7)
    history["y"] = history["y"].astype(str)
    assert seasonal_naive(history, horizon=3)["y_pred"].tolist() == [0, 1, 2]


def test_zero_sales():
    history = make_history(7).assign(y=0)
    assert seasonal_naive(history)["y_pred"].eq(0).all()
