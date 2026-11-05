"""Weekly Seasonal Naive forecasts using only the supplied history."""

import pandas as pd

from sales_forecasting.splitting import _canonical

SEASON_LENGTH = 7


def seasonal_naive(train: pd.DataFrame, *, horizon: int = 90) -> pd.DataFrame:
    """Repeat each series' last seven daily observations for the next horizon days.

    Input uses unique_id/ds/y; output uses unique_id/ds/y_pred. Each series
    needs at least seven observations and uninterrupted daily history.
    Series may end on different dates when this function is used standalone.
    The input is not modified and no future observations are accepted separately.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon должен быть положительным целым числом.")
    history = _canonical(train)
    forecasts = []
    for unique_id, series in history.groupby("unique_id", sort=False):
        if len(series) < SEASON_LENGTH:
            raise ValueError(
                f"Недостаточно истории ряда {unique_id!r}: нужно минимум 7 дней."
            )
        if not series["ds"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Ряд {unique_id!r} содержит пропущенные календарные дни.")
        week = pd.to_numeric(series["y"].tail(SEASON_LENGTH)).tolist()
        forecasts.append(
            pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        series["ds"].iloc[-1] + pd.Timedelta(days=1),
                        periods=horizon,
                        freq="D",
                    ),
                    "y_pred": [week[step % SEASON_LENGTH] for step in range(horizon)],
                }
            )
        )
    return pd.concat(forecasts, ignore_index=True)
