"""Strict one-to-one alignment of forecasts and actual observations."""

from math import isfinite

import pandas as pd

from sales_forecasting.metrics import mae, mape, rmse
from sales_forecasting.splitting import _canonical
from sales_forecasting.validation import _validate_table, parse_dates


def align_forecasts(actual: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return unique_id/ds/y/y_pred; require identical key sets.

    Predictions may be negative but must be finite. Dates and identifiers
    are normalized like actuals; row order and DataFrame indices are ignored.
    """
    truth = _canonical(actual)[["unique_id", "ds", "y"]]
    truth["y"] = pd.to_numeric(truth["y"])
    _validate_table(forecasts, ("unique_id", "ds", "y_pred"))
    predicted = forecasts[["unique_id", "ds", "y_pred"]].copy()
    predicted["unique_id"] = predicted["unique_id"].astype(str)
    predicted["ds"] = parse_dates(predicted["ds"])
    if predicted.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Дубликаты прогнозов по ключам (unique_id, ds).")
    values = predicted["y_pred"]
    numbers = pd.to_numeric(values, errors="coerce")
    if (
        values.map(lambda value: isinstance(value, (bool, complex))).any()
        or numbers.isna().any()
        or not numbers.map(isfinite).all()
    ):
        raise ValueError("Прогноз y_pred должен содержать конечные числа.")
    predicted["y_pred"] = numbers
    merged = truth.merge(
        predicted,
        on=["unique_id", "ds"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged["_merge"].eq("left_only").sum()
    extra = merged["_merge"].eq("right_only").sum()
    if missing or extra:
        raise ValueError(
            f"Несовпадение ключей: пропущено прогнозов {missing}, лишних {extra}."
        )
    return (
        merged.drop(columns="_merge")
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def evaluate_forecasts(
    actual: pd.DataFrame, forecasts: pd.DataFrame
) -> dict[str, float]:
    """Compute pooled MAE/RMSE/MAPE after strict key alignment.

    Every observation has equal weight. If all actuals are zero MAPE raises
    ValueError; MAE and RMSE remain available through their standalone functions.
    """
    aligned = align_forecasts(actual, forecasts)
    a, p = aligned["y"], aligned["y_pred"]
    return {"MAE": mae(a, p), "RMSE": rmse(a, p), "MAPE": mape(a, p)}
