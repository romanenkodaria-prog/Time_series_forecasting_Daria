"""N-BEATS factories and explicit Train-only fitting via NeuralForecast."""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

import pandas as pd

from sales_forecasting.evaluation import align_forecasts
from sales_forecasting.splitting import _canonical

if TYPE_CHECKING:
    from neuralforecast import NeuralForecast


@dataclass(frozen=True)
class NBEATSConfig:
    """Research settings; smoke mode changes steps and network size separately."""

    h: int = 90
    input_size: int = 180
    max_steps: int = 500
    learning_rate: float = 0.001
    batch_size: int = 32
    scaler_type: str = "standard"
    random_seed: int = 42
    smoke: bool = False


EXPERIMENT_CONFIG = NBEATSConfig()
FAST_CONFIG = replace(EXPERIMENT_CONFIG, max_steps=2, smoke=True)


def _validate_config(config: NBEATSConfig) -> None:
    if config.h != 90 or config.input_size != 180:
        raise ValueError("N-BEATS требует h=90 и input_size=180.")
    for name in ("h", "input_size", "max_steps", "batch_size"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом.")
    if config.smoke and not 2 <= config.max_steps <= 5:
        raise ValueError("Быстрая конфигурация требует от 2 до 5 шагов.")


def create_nbeats(config: NBEATSConfig = EXPERIMENT_CONFIG) -> "NeuralForecast":
    """Create an unfitted daily model on CPU, without files or worker processes.

    Heavy dependencies are imported only here so ordinary data tests stay light.
    Research architecture uses Nixtla defaults. Smoke mode uses one identity
    block with 32-unit layers and 16 sampled windows; it is not a quality test.
    """
    _validate_config(config)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS

    options = {}
    if config.smoke:
        options.update(
            stack_types=["identity"],
            n_blocks=[1],
            mlp_units=[[32, 32]],
            windows_batch_size=16,
        )
    model = NBEATS(
        h=config.h,
        input_size=config.input_size,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        scaler_type=config.scaler_type,
        random_seed=config.random_seed,
        alias="NBEATS",
        start_padding_enabled=False,
        early_stop_patience_steps=-1,
        val_check_steps=min(100, config.max_steps),
        dataloader_kwargs={"num_workers": 0},
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **options,
    )
    return NeuralForecast(models=[model], freq="D")


@dataclass
class FittedNBEATS:
    """Trained forecaster and expected future keys derived only from Train."""

    forecaster: "NeuralForecast"
    future_keys: pd.DataFrame
    fit_seconds: float


def fit_nbeats(
    train: pd.DataFrame, config: NBEATSConfig = EXPERIMENT_CONFIG
) -> FittedNBEATS:
    """Fit only supplied Train, with no internal holdout or early stopping.

    Require at least input_size+h=270 continuous daily observations per series
    to build a complete training window without artificial left padding.
    Validation and Test must be separated by the caller before this function.
    """
    _validate_config(config)
    history = _canonical(train)[["unique_id", "ds", "y"]]
    history["y"] = pd.to_numeric(history["y"])
    future = []
    for unique_id, series in history.groupby("unique_id", sort=False):
        if len(series) < config.input_size + config.h:
            raise ValueError(
                f"Недостаточно истории ряда {unique_id!r}: нужно минимум 270 дней Train."
            )
        if not series["ds"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Ряд {unique_id!r} содержит пропущенные календарные дни.")
        future.append(
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        series["ds"].iloc[-1] + pd.Timedelta(days=1),
                        periods=config.h,
                        freq="D",
                    ),
                }
            )
        )
    forecaster = create_nbeats(config)
    start = perf_counter()
    forecaster.fit(df=history, val_size=0)
    elapsed = perf_counter() - start
    return FittedNBEATS(forecaster, pd.concat(future, ignore_index=True), elapsed)


def predict_nbeats(fitted: FittedNBEATS) -> pd.DataFrame:
    """Predict and strictly check 90 future daily keys per fitted series.

    Return unique_id/ds/y_pred. Negative predictions are preserved and scored
    as-is. Missing/extra keys, duplicates and nonfinite predictions are errors.
    """
    forecast = fitted.forecaster.predict().rename(columns={"NBEATS": "y_pred"})
    # Zero placeholders are used only to reuse key validation, never for training.
    aligned = align_forecasts(fitted.future_keys.assign(y=0), forecast)
    return aligned.drop(columns="y")
