"""Explicit two-step synthetic N-BEATS check: python -m sales_forecasting.nbeats_smoke."""

import json
import platform
from importlib.metadata import version
from math import sin, tau
from time import perf_counter

import pandas as pd

from sales_forecasting.evaluation import evaluate_forecasts
from sales_forecasting.models.nbeats import FAST_CONFIG, fit_nbeats, predict_nbeats
from sales_forecasting.splitting import split_series


def run_smoke() -> dict:
    """Fit three small series on CPU; score Validation and never evaluate Test."""
    import torch

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": f"synthetic_{index}",
                    "ds": pd.date_range("2023-01-01", periods=450),
                    "y": [
                        20 + index * 5 + 3 * sin(tau * day / 7) + day * 0.01
                        for day in range(450)
                    ],
                }
            )
            for index in range(3)
        ],
        ignore_index=True,
    )
    train, validation, _test = split_series(data, min_train_days=270)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        start = perf_counter()
        fitted = fit_nbeats(train, FAST_CONFIG)
        setup_and_fit_seconds = perf_counter() - start
        # NeuralForecast 3.3 removes its Trainer after fit, but retains one
        # training trajectory entry per training step (no gradient accumulation).
        completed_steps = len(fitted.forecaster.models[0].train_trajectories)
        forecasts = predict_nbeats(fitted)
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "neuralforecast": version("neuralforecast"),
        "torch": torch.__version__,
        "pytorch_lightning": version("pytorch-lightning"),
        "accelerator": "cpu",
        "max_steps": FAST_CONFIG.max_steps,
        "completed_steps": completed_steps,
        "fit_seconds": fitted.fit_seconds,
        "setup_and_fit_seconds": setup_and_fit_seconds,
        "series": forecasts["unique_id"].nunique(),
        "forecast_rows": len(forecasts),
        "dates_per_series": forecasts.groupby("unique_id")["ds"].nunique().to_dict(),
        "validation_metrics": evaluate_forecasts(validation, forecasts),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, ensure_ascii=False))
