"""Validation of raw sales and forecasting tables."""

from math import isfinite

import pandas as pd


def _validate_table(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Ожидается pandas.DataFrame.")
    if not data.columns.is_unique:
        raise ValueError("Названия колонок не должны повторяться.")
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing)}.")
    if data.empty:
        raise ValueError("Таблица данных пуста.")
    for column in columns:
        values = data[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Колонка {column} содержит пропуски.")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO calendar dates or timezone-naive midnight timestamps."""
    if pd.api.types.is_datetime64_any_dtype(values):
        dates = values
    else:
        if not values.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("Некорректные даты: ожидается формат YYYY-MM-DD.")
        dates = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Некорректные даты: обнаружена несуществующая дата.")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Даты должны быть без часового пояса и времени суток.")
    return dates


def _validate_sales(values: pd.Series) -> None:
    numbers = pd.to_numeric(values, errors="coerce")
    if (
        values.map(lambda value: isinstance(value, (bool, complex))).any()
        or numbers.isna().any()
        or not numbers.map(isfinite).all()
    ):
        raise ValueError(f"Колонка {values.name} должна содержать конечные числа.")
    if numbers.lt(0).any():
        raise ValueError(
            f"Колонка {values.name}: продажи должны быть неотрицательными."
        )


def validate_raw_data(data: pd.DataFrame) -> None:
    """Raise ValueError for invalid store/item/date/sales observations."""
    _validate_table(data, ("store", "item", "date", "sales"))
    dates = parse_dates(data["date"])
    _validate_sales(data["sales"])
    keys = data[["store", "item"]].astype(str).assign(date=dates)
    if keys.duplicated().any():
        raise ValueError("Дубликаты пары магазин–товар и даты (unique_id, ds).")


def validate_forecasting_data(data: pd.DataFrame) -> None:
    """Raise ValueError for invalid unique_id/ds/y observations."""
    _validate_table(data, ("unique_id", "ds", "y"))
    dates = parse_dates(data["ds"])
    _validate_sales(data["y"])
    keys = data[["unique_id"]].astype(str).assign(ds=dates)
    if keys.duplicated().any():
        raise ValueError("Дубликаты (unique_id, ds).")
