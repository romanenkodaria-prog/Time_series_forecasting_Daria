"""End-to-end CSV evaluation and checks against holdout leakage."""

import json
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import sales_forecasting.pipeline as pipeline
from sales_forecasting.data import load_sales_csv
from sales_forecasting.preprocessing import prepare_sales_data
from sales_forecasting.splitting import split_series


@pytest.fixture
def sales_csv(tmp_path):
    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "store": store,
                    "item": 1,
                    "date": pd.date_range("2024-01-01", periods=194),
                    "sales": [offset + day % 7 for day in range(194)],
                }
            )
            for store, offset in [(1, 0), (2, 10)]
        ],
        ignore_index=True,
    )
    path = tmp_path / "sales.csv"
    raw.sample(frac=1, random_state=3).to_csv(path, index=False)
    return path


def test_end_to_end(sales_csv):
    original = sales_csv.read_bytes()
    result = pipeline.run_baseline_pipeline(sales_csv)
    assert result.validation_metrics == {"MAE": 0, "RMSE": 0, "MAPE": 0}
    assert len(result.train) == 28
    assert len(result.validation_forecasts) == len(result.validation) == 180
    assert result.validation_forecasts["ds"].max() < result.test["ds"].min()
    expected = split_series(
        prepare_sales_data(load_sales_csv(sales_csv)), min_train_days=7
    )
    for actual, wanted in zip((result.train, result.validation, result.test), expected):
        assert_frame_equal(actual, wanted)
    assert sales_csv.read_bytes() == original


def test_model_and_evaluator_receive_only_intended_splits(sales_csv, monkeypatch):
    train, validation, test = split_series(
        prepare_sales_data(load_sales_csv(sales_csv)), min_train_days=7
    )
    snapshots = [part.copy(deep=True) for part in (train, validation, test)]
    model = pipeline.seasonal_naive
    evaluator = pipeline.evaluate_forecasts
    calls = []

    def predict(history):
        assert_frame_equal(history, train)
        calls.append("model")
        return model(history)

    def evaluate(actual, forecasts):
        assert_frame_equal(actual, validation)
        calls.append("evaluation")
        return evaluator(actual, forecasts)

    monkeypatch.setattr(
        pipeline, "split_series", lambda *args, **kwargs: (train, validation, test)
    )
    monkeypatch.setattr(pipeline, "seasonal_naive", predict)
    monkeypatch.setattr(pipeline, "evaluate_forecasts", evaluate)
    result = pipeline.run_baseline_pipeline(sales_csv)
    assert calls == ["model", "evaluation"]
    assert result.test is test
    for part, snapshot in zip((train, validation, test), snapshots):
        assert_frame_equal(part, snapshot)


def test_holdout_values_cannot_change_forecast(sales_csv):
    original = pipeline.run_baseline_pipeline(sales_csv)
    raw = pd.read_csv(sales_csv)
    dates = pd.to_datetime(raw["date"])
    raw.loc[dates >= original.test["ds"].min(), "sales"] = 99999
    raw.to_csv(sales_csv, index=False)
    changed_test = pipeline.run_baseline_pipeline(sales_csv)
    assert_frame_equal(original.validation_forecasts, changed_test.validation_forecasts)
    assert original.validation_metrics == changed_test.validation_metrics
    raw.loc[
        (dates >= original.validation["ds"].min())
        & (dates < original.test["ds"].min()),
        "sales",
    ] = 200
    raw.to_csv(sales_csv, index=False)
    changed_validation = pipeline.run_baseline_pipeline(sales_csv)
    assert_frame_equal(
        original.validation_forecasts, changed_validation.validation_forecasts
    )
    assert changed_validation.validation_metrics["MAE"] > 0


@pytest.mark.parametrize("days", [180, 186])
def test_insufficient_train_history(sales_csv, days):
    raw = pd.read_csv(sales_csv).sort_values("date").groupby("store").tail(days)
    raw.to_csv(sales_csv, index=False)
    with pytest.raises(ValueError, match="Недостаточно истории"):
        pipeline.run_baseline_pipeline(sales_csv)


def test_exact_minimum_history(sales_csv):
    raw = pd.read_csv(sales_csv).sort_values("date").groupby("store").tail(187)
    raw.to_csv(sales_csv, index=False)
    assert len(pipeline.run_baseline_pipeline(sales_csv).train) == 14


def test_pipeline_rejects_missing_dates(sales_csv):
    raw = pd.read_csv(sales_csv)
    raw.loc[~((raw["store"] == 1) & (raw["date"] == "2024-01-03"))].to_csv(
        sales_csv, index=False
    )
    with pytest.raises(ValueError, match="пропущенные календарные дни"):
        pipeline.run_baseline_pipeline(sales_csv)


def test_all_zero_validation_follows_mape_policy(sales_csv):
    raw = pd.read_csv(sales_csv).assign(sales=0)
    raw.to_csv(sales_csv, index=False)
    with pytest.raises(ValueError, match="MAPE не определена"):
        pipeline.run_baseline_pipeline(sales_csv)


def test_cli(sales_csv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", str(sales_csv)])
    pipeline.main()
    assert json.loads(capsys.readouterr().out) == {"MAE": 0, "RMSE": 0, "MAPE": 0}


def test_cli_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2
    assert "CSV-файл не найден" in capsys.readouterr().err
