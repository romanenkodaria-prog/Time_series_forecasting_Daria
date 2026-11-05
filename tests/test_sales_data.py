"""Small synthetic examples for sales loading and validation."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sales_forecasting.data import load_sales_csv
from sales_forecasting.preprocessing import prepare_sales_data
from sales_forecasting.validation import (
    validate_forecasting_data,
    validate_raw_data,
)


@pytest.fixture
def raw():
    return pd.DataFrame(
        {
            "store": [1, 1, 2],
            "item": [3, 3, 4],
            "date": ["2024-01-02", "2024-01-01", "2024-02-29"],
            "sales": [5, 0, 8],
        }
    )


def test_prepare(raw):
    original = raw.copy(deep=True)
    result = prepare_sales_data(raw)
    expected = pd.DataFrame(
        {
            "unique_id": ["1/3", "1/3", "2/4"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-29"]),
            "y": [0, 5, 8],
        }
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(raw, original)
    validate_forecasting_data(result)


def test_load_csv(tmp_path, raw):
    path = tmp_path / "sales.csv"
    raw.to_csv(path, index=False)
    loaded = load_sales_csv(path)
    assert loaded["store"].tolist() == ["1", "1", "2"]
    assert_frame_equal(prepare_sales_data(loaded), prepare_sales_data(raw))


def test_load_preserves_identifier_zeros(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("store,item,date,sales\n001,02,2024-01-01,0\n", encoding="utf-8")
    assert prepare_sales_data(load_sales_csv(path))["unique_id"].tolist() == ["001/02"]


@pytest.mark.parametrize("column", ["store", "item", "date", "sales"])
def test_missing_columns(raw, column):
    with pytest.raises(ValueError, match="обязательные колонки"):
        prepare_sales_data(raw.drop(columns=column))


@pytest.mark.parametrize("column", ["store", "item", "date", "sales"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_values(raw, column, value):
    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match="пропуски"):
        prepare_sales_data(raw)


@pytest.mark.parametrize(
    "value",
    ["bad", "2023-02-29", "2024-13-01", 20240101, "01/02/2024", "2024-01-01T12:00:00"],
)
def test_invalid_dates(raw, value):
    raw.loc[0, "date"] = value
    with pytest.raises(ValueError, match="даты"):
        prepare_sales_data(raw)


@pytest.mark.parametrize("value", [-1, "bad", float("inf"), float("-inf"), True])
def test_invalid_sales(raw, value):
    raw["sales"] = raw["sales"].astype(object)
    raw.loc[0, "sales"] = value
    with pytest.raises(ValueError, match="продажи|числа"):
        prepare_sales_data(raw)


def test_raw_duplicates(raw):
    with pytest.raises(ValueError, match="Дубликаты"):
        prepare_sales_data(pd.concat([raw, raw.iloc[[0]]]))


def test_prepared_duplicates(raw):
    prepared = prepare_sales_data(raw)
    with pytest.raises(ValueError, match="Дубликаты"):
        validate_forecasting_data(pd.concat([prepared, prepared.iloc[[0]]]))


@pytest.mark.parametrize(
    "column,value", [("unique_id", None), ("ds", "bad"), ("y", -1)]
)
def test_invalid_prepared_data(raw, column, value):
    prepared = prepare_sales_data(raw).astype({"ds": object})
    prepared.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_forecasting_data(prepared)


def test_prepared_missing_column(raw):
    with pytest.raises(ValueError, match="обязательные колонки"):
        validate_forecasting_data(prepare_sales_data(raw).drop(columns="y"))


def test_empty_table(raw):
    with pytest.raises(ValueError, match="пуста"):
        validate_raw_data(raw.iloc[:0])


def test_wrong_input_type():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_raw_data([])


def test_duplicate_column_names(raw):
    with pytest.raises(ValueError, match="Названия колонок"):
        validate_raw_data(pd.concat([raw, raw[["sales"]]], axis=1))


def test_identifier_separator_collision():
    raw = pd.DataFrame(
        {
            "store": ["a/b", "a"],
            "item": ["c", "b/c"],
            "date": ["2024-01-01"] * 2,
            "sales": [1, 2],
        }
    )
    assert prepare_sales_data(raw)["unique_id"].nunique() == 2


def test_datetime_input(raw):
    raw["date"] = pd.to_datetime(raw["date"])
    validate_raw_data(raw)


@pytest.mark.parametrize(
    "date", [pd.Timestamp("2024-01-01T12:00:00"), pd.Timestamp("2024-01-01", tz="UTC")]
)
def test_datetime_with_time_or_timezone(raw, date):
    raw["date"] = date
    with pytest.raises(ValueError, match="времени суток"):
        validate_raw_data(raw)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "store,item,date,sales\n",
        'store,item,date,sales\n"unterminated',
        "wrong,columns\n1,2\n",
    ],
)
def test_invalid_csv(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="CSV"):
        load_sales_csv(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_sales_csv(tmp_path / "missing.csv")


def test_invalid_encoding(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(ValueError, match="CSV"):
        load_sales_csv(path)


def test_unreadable_file(tmp_path, monkeypatch):
    def deny_access(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr(pd, "read_csv", deny_access)
    with pytest.raises(OSError, match="открыть CSV"):
        load_sales_csv(tmp_path / "private.csv")
