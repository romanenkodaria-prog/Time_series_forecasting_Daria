"""Calendar-based splits with shared cutoffs across all series."""

import pandas as pd

from sales_forecasting.validation import parse_dates, validate_forecasting_data


def _canonical(data: pd.DataFrame) -> pd.DataFrame:
    validate_forecasting_data(data)
    result = data.copy()
    result["unique_id"] = result["unique_id"].astype(str)
    result["ds"] = parse_dates(result["ds"])
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def validate_splits(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Reject missing series, overlapping periods and reversed chronology globally."""
    parts = [_canonical(part) for part in (train, validation, test)]
    ids = [set(part["unique_id"]) for part in parts]
    if not ids[0] == ids[1] == ids[2]:
        raise ValueError("Train, Validation и Test должны содержать одни и те же ряды.")
    for earlier, later in zip(parts, parts[1:]):
        if earlier["ds"].max() >= later["ds"].min():
            raise ValueError("Пересечение периодов или нарушение хронологии разделов.")


def split_series(
    data: pd.DataFrame,
    *,
    validation_days: int = 90,
    test_days: int = 90,
    min_train_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Train, Validation, Test using common calendar cutoffs.

    Every series must have uninterrupted daily history ending at the global
    last date and at least min_train_days before the two holdout windows.
    No missing dates are filled and the input is never modified.
    """
    for name, value in (
        ("validation_days", validation_days),
        ("test_days", test_days),
        ("min_train_days", min_train_days),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом.")
    frame = _canonical(data)
    end = frame["ds"].max()
    test_start = end - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    for unique_id, series in frame.groupby("unique_id", sort=False):
        dates = series["ds"]
        if dates.max() != end or (dates < validation_start).sum() < min_train_days:
            raise ValueError(f"Недостаточно истории ряда {unique_id!r} для разделения.")
        if not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Ряд {unique_id!r} содержит пропущенные календарные дни.")
    train = frame.loc[frame["ds"] < validation_start].reset_index(drop=True)
    validation = frame.loc[
        (frame["ds"] >= validation_start) & (frame["ds"] < test_start)
    ].reset_index(drop=True)
    test = frame.loc[frame["ds"] >= test_start].reset_index(drop=True)
    validate_splits(train, validation, test)
    return train, validation, test
