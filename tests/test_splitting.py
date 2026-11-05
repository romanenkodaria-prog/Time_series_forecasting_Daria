"""Calendar splits on synthetic daily series."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sales_forecasting.splitting import split_series, validate_splits


@pytest.fixture
def history():
    return pd.concat(
        [
            pd.DataFrame(
                {"unique_id": uid, "ds": pd.date_range(start, "2024-08-01"), "y": 0}
            )
            for uid, start in [("a", "2024-01-01"), ("b", "2024-01-10")]
        ],
        ignore_index=True,
    )


def test_common_calendar_windows(history):
    shuffled = history.sample(frac=1, random_state=7)
    original = shuffled.copy(deep=True)
    train, validation, test = split_series(shuffled)
    assert_frame_equal(shuffled, original)
    for part in (validation, test):
        assert part.groupby("unique_id").size().to_dict() == {"a": 90, "b": 90}
        assert (part["ds"].max() - part["ds"].min()).days == 89
    assert train["ds"].max() < validation["ds"].min()
    assert validation["ds"].max() < test["ds"].min()
    assert len(train) + len(validation) + len(test) == len(history)
    combined = pd.concat([train, validation, test]).sort_values(["unique_id", "ds"])
    assert_frame_equal(combined.reset_index(drop=True), history)


def test_minimum_and_custom_windows():
    frame = pd.DataFrame(
        {"unique_id": "a", "ds": pd.date_range("2024-01-01", periods=181), "y": 1}
    )
    assert [len(part) for part in split_series(frame)] == [1, 90, 90]
    assert [
        len(part)
        for part in split_series(
            frame, validation_days=2, test_days=3, min_train_days=176
        )
    ] == [176, 2, 3]
    with pytest.raises(ValueError, match="Недостаточно истории"):
        split_series(frame.iloc[1:])


def test_one_short_series(history):
    history = history.loc[
        ~((history["unique_id"] == "b") & (history["ds"] < "2024-04-01"))
    ]
    with pytest.raises(ValueError, match="истории ряда 'b'"):
        split_series(history)


def test_missing_day(history):
    with pytest.raises(ValueError, match="пропущенные календарные дни"):
        split_series(history.drop(index=100))


def test_different_end_dates(history):
    with pytest.raises(ValueError, match="Недостаточно истории"):
        split_series(history.iloc[:-1])


@pytest.mark.parametrize("name", ["validation_days", "test_days", "min_train_days"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "90"])
def test_invalid_lengths(history, name, value):
    with pytest.raises(ValueError, match="положительным целым"):
        split_series(history, **{name: value})


def test_overlap_and_reversed_periods(history):
    train, validation, test = split_series(history)
    with pytest.raises(ValueError, match="Пересечение"):
        validate_splits(pd.concat([train, validation.iloc[[0]]]), validation, test)
    with pytest.raises(ValueError, match="хронологии"):
        validate_splits(test, validation, train)


def test_cross_series_period_overlap(history):
    train, validation, test = split_series(history)
    train.loc[train["unique_id"] == "b", "ds"] += pd.Timedelta(days=40)
    with pytest.raises(ValueError, match="Пересечение"):
        validate_splits(train, validation, test)


def test_missing_series(history):
    train, validation, test = split_series(history)
    with pytest.raises(ValueError, match="одни и те же ряды"):
        validate_splits(train, validation, test.loc[test["unique_id"] == "a"])


def test_duplicate_input(history):
    with pytest.raises(ValueError, match="Дубликаты"):
        split_series(pd.concat([history, history.iloc[[0]]]))
