"""Experiment orchestration tests without CSV downloads or neural training."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import sales_forecasting.real_data_experiment as experiment
from sales_forecasting.pipeline import BaselineResult


@pytest.fixture
def setup_experiment(tmp_path, monkeypatch):
    path = tmp_path / "train.csv"
    path.write_text("date,store,item,sales\n2024-01-01,1,1,10\n", encoding="utf-8")

    def panel(start, days):
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": pd.date_range(start, periods=days),
                        "y": 10,
                    }
                )
                for uid in (*experiment.DEFAULT_IDS, "2/1")
            ],
            ignore_index=True,
        )

    train = panel("2023-01-01", 270)
    validation = panel(train["ds"].max() + pd.Timedelta(days=1), 90)
    test = panel(validation["ds"].max() + pd.Timedelta(days=1), 90)
    full = BaselineResult(
        train,
        validation,
        test,
        validation.rename(columns={"y": "y_pred"}),
        {"MAE": 0, "RMSE": 0, "MAPE": 0},
    )
    loader = MagicMock(return_value=full)
    monkeypatch.setattr(experiment, "run_baseline_pipeline", loader)
    torch = ModuleType("torch")
    torch.__version__ = "test-backend"
    torch.get_num_threads = lambda: 4
    torch.set_num_threads = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    fitted = SimpleNamespace(
        fit_seconds=0.1,
        forecaster=SimpleNamespace(
            models=[SimpleNamespace(train_trajectories=list(range(5)))]
        ),
    )
    fit = MagicMock(return_value=fitted)
    monkeypatch.setattr(experiment, "fit_nbeats", fit)
    predict = MagicMock(
        return_value=validation.loc[
            validation["unique_id"].isin(experiment.DEFAULT_IDS), ["unique_id", "ds"]
        ].assign(y_pred=9)
    )
    monkeypatch.setattr(experiment, "predict_nbeats", predict)
    return path, full, fit, torch


def test_same_train_validation_and_no_test_leakage(setup_experiment, monkeypatch):
    path, full, fit, torch = setup_experiment
    test_before = full.test.copy(deep=True)
    selected = full.train.loc[full.train["unique_id"].isin(experiment.DEFAULT_IDS)]
    selected_validation = full.validation.loc[
        full.validation["unique_id"].isin(experiment.DEFAULT_IDS)
    ]
    evaluator = MagicMock(wraps=experiment.evaluate_forecasts)
    monkeypatch.setattr(experiment, "evaluate_forecasts", evaluator)
    result = experiment.run_experiment(path)
    assert_frame_equal(fit.call_args.args[0], selected)
    assert fit.call_args.args[1].max_steps == 5
    assert not fit.call_args.args[1].smoke
    assert evaluator.call_count == 2
    for call in evaluator.call_args_list:
        assert_frame_equal(call.args[0], selected_validation)
    assert_frame_equal(full.test, test_before)
    assert result["test_unchanged"]
    assert not result["test_used_for_fit_or_selection"]
    assert result["comparison"]["baseline"]["metrics"]["MAE"] == 0
    assert result["comparison"]["nbeats"]["metrics"]["MAE"] == 1
    assert result["comparison"]["nbeats"]["forecast_rows"] == 270
    assert torch.set_num_threads.call_args_list[-1].args == (4,)
    report = experiment.render_report(result)
    assert "Все 4 рядов" in report
    assert "нельзя считать оценкой качества на всём датасете" in report
    assert "1/1, 1/2, 1/3" in report


@pytest.mark.parametrize(
    "ids", [("1/1",), ("1/1", "1/1", "1/2"), tuple(str(i) for i in range(6))]
)
def test_series_limit_before_reading_data(tmp_path, ids):
    with pytest.raises(ValueError, match="от 3 до 5"):
        experiment.run_experiment(tmp_path / "absent.csv", unique_ids=ids)


@pytest.mark.parametrize("steps", [1, 21, 500, True, 2.5])
def test_step_limit_before_reading_data(tmp_path, steps):
    with pytest.raises(ValueError, match="от 2 до 20"):
        experiment.run_experiment(tmp_path / "absent.csv", max_steps=steps)


def test_unknown_series_never_trains(setup_experiment):
    path, _, fit, _ = setup_experiment
    with pytest.raises(ValueError, match="Неизвестные unique_id"):
        experiment.run_experiment(path, unique_ids=("1/1", "1/2", "unknown"))
    fit.assert_not_called()


def test_rejects_changed_source(setup_experiment):
    path, _, fit, _ = setup_experiment
    path.with_suffix(".source.json").write_text(
        '{"sha256": "mismatch"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        experiment.run_experiment(path)
    fit.assert_not_called()


def test_detects_test_mutation(setup_experiment):
    path, full, fit, _ = setup_experiment
    fitted = fit.return_value

    def mutate_test(*args):
        full.test.loc[0, "y"] = 999
        return fitted

    fit.side_effect = mutate_test
    with pytest.raises(RuntimeError, match="Test был изменён"):
        experiment.run_experiment(path)


def test_restores_threads_if_training_fails(setup_experiment):
    path, _, fit, torch = setup_experiment
    fit.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        experiment.run_experiment(path)
    assert torch.set_num_threads.call_args_list[-1].args == (4,)
