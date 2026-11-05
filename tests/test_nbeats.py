"""Fast contract tests; real two-step training is explicitly opt-in."""

import os
import sys
from dataclasses import replace
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import sales_forecasting.models.nbeats as nbeats


@pytest.fixture
def train():
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2024-01-01", periods=270),
                    "y": 10 + i,
                }
            )
            for i, uid in enumerate(["a", "b"])
        ],
        ignore_index=True,
    )


def test_research_config_is_separate_from_fast_config():
    config = nbeats.EXPERIMENT_CONFIG
    assert (config.h, config.input_size, config.max_steps) == (90, 180, 500)
    assert (
        config.learning_rate,
        config.batch_size,
        config.scaler_type,
        config.random_seed,
    ) == (0.001, 32, "standard", 42)
    assert nbeats.FAST_CONFIG.max_steps == 2
    assert config == replace(nbeats.FAST_CONFIG, max_steps=500, smoke=False)
    with pytest.raises(AttributeError):
        config.max_steps = 2


@pytest.mark.parametrize("config", [nbeats.EXPERIMENT_CONFIG, nbeats.FAST_CONFIG])
def test_factory_passes_settings_without_training(monkeypatch, config):
    package, models = ModuleType("neuralforecast"), ModuleType("neuralforecast.models")
    package.NeuralForecast = MagicMock()
    models.NBEATS = MagicMock()
    monkeypatch.setitem(sys.modules, "neuralforecast", package)
    monkeypatch.setitem(sys.modules, "neuralforecast.models", models)
    result = nbeats.create_nbeats(config)
    options = models.NBEATS.call_args.kwargs
    for name in (
        "h",
        "input_size",
        "max_steps",
        "learning_rate",
        "batch_size",
        "scaler_type",
        "random_seed",
    ):
        assert options[name] == getattr(config, name)
    assert options["accelerator"] == "cpu"
    assert options["dataloader_kwargs"] == {"num_workers": 0}
    assert not options["enable_checkpointing"]
    assert ("mlp_units" in options) == config.smoke
    package.NeuralForecast.assert_called_once_with(
        models=[models.NBEATS.return_value], freq="D"
    )
    assert result is package.NeuralForecast.return_value
    result.fit.assert_not_called()


def test_fit_uses_only_provided_train_and_predicts_future_keys(train, monkeypatch):
    original = train.copy(deep=True)
    forecaster = MagicMock()
    factory = MagicMock(return_value=forecaster)
    monkeypatch.setattr(nbeats, "create_nbeats", factory)
    fitted = nbeats.fit_nbeats(train.sample(frac=1, random_state=5), nbeats.FAST_CONFIG)
    factory.assert_called_once_with(nbeats.FAST_CONFIG)
    assert_frame_equal(forecaster.fit.call_args.kwargs["df"], original)
    assert forecaster.fit.call_args.kwargs["val_size"] == 0
    assert_frame_equal(train, original)
    assert fitted.future_keys.groupby("unique_id").size().to_dict() == {
        "a": 90,
        "b": 90,
    }
    assert fitted.future_keys["ds"].min() == train["ds"].max() + pd.Timedelta(days=1)
    forecaster.predict.return_value = fitted.future_keys.assign(NBEATS=12.0).iloc[::-1]
    result = nbeats.predict_nbeats(fitted)
    assert_frame_equal(result, fitted.future_keys.assign(y_pred=12.0))
    forecaster.predict.assert_called_once_with()


@pytest.mark.parametrize(
    "problem", ["short", "gap", "duplicate", "bad_date", "negative"]
)
def test_invalid_train_before_model_creation(train, monkeypatch, problem):
    factory = MagicMock()
    monkeypatch.setattr(nbeats, "create_nbeats", factory)
    if problem == "short":
        train = train.iloc[:-1]
    elif problem == "gap":
        extra = train.iloc[[0]].assign(ds=pd.Timestamp("2023-12-30"))
        train = pd.concat([train.drop(index=20), extra])
    elif problem == "duplicate":
        train = pd.concat([train, train.iloc[[0]]])
    elif problem == "bad_date":
        train["ds"] = "not-a-date"
    else:
        train.loc[0, "y"] = -1
    with pytest.raises(ValueError):
        nbeats.fit_nbeats(train, nbeats.FAST_CONFIG)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "problem", ["missing", "extra", "duplicate", "infinite", "wrong_date"]
)
def test_strict_prediction_contract(train, monkeypatch, problem):
    forecaster = MagicMock()
    monkeypatch.setattr(nbeats, "create_nbeats", lambda config: forecaster)
    fitted = nbeats.fit_nbeats(train, nbeats.FAST_CONFIG)
    forecast = fitted.future_keys.assign(NBEATS=1.0)
    if problem == "missing":
        forecast = forecast.iloc[1:]
    elif problem == "extra":
        forecast = pd.concat([forecast, forecast.iloc[[0]].assign(unique_id="extra")])
    elif problem == "duplicate":
        forecast = pd.concat([forecast, forecast.iloc[[0]]])
    elif problem == "infinite":
        forecast.loc[0, "NBEATS"] = float("inf")
    else:
        forecast.loc[0, "ds"] -= pd.Timedelta(days=1)
    forecaster.predict.return_value = forecast
    with pytest.raises(ValueError):
        nbeats.predict_nbeats(fitted)


@pytest.mark.parametrize(
    "changes",
    [
        {"h": 30},
        {"input_size": 10},
        {"max_steps": 0},
        {"batch_size": True},
        {"smoke": True, "max_steps": 6},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        nbeats.create_nbeats(replace(nbeats.EXPERIMENT_CONFIG, **changes))


@pytest.mark.skipif(
    os.environ.get("RUN_NBEATS_SMOKE") != "1",
    reason="Real training is opt-in: set RUN_NBEATS_SMOKE=1",
)
def test_real_two_step_training():
    from sales_forecasting.nbeats_smoke import run_smoke

    result = run_smoke()
    assert result["completed_steps"] == 2
    assert result["forecast_rows"] == 270
    assert set(result["dates_per_series"].values()) == {90}
    assert set(result["validation_metrics"]) == {"MAE", "RMSE", "MAPE"}
