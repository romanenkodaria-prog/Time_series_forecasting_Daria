"""Load sales and evaluate a weekly baseline on Validation only."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sales_forecasting.data import load_sales_csv
from sales_forecasting.evaluation import evaluate_forecasts
from sales_forecasting.models.baseline import SEASON_LENGTH, seasonal_naive
from sales_forecasting.preprocessing import prepare_sales_data
from sales_forecasting.splitting import split_series


@dataclass
class BaselineResult:
    """Splits, Validation forecasts and pooled Validation metrics.

    Test is returned unchanged from splitting for future final evaluation.
    It is never passed to the model or the evaluator by this pipeline.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_forecasts: pd.DataFrame
    validation_metrics: dict[str, float]


def run_baseline_pipeline(
    path: str | Path, *, encoding: str = "utf-8"
) -> BaselineResult:
    """Read CSV, split into Train/90-day Validation/90-day Test and score baseline.

    Every series needs at least 187 continuous daily observations with a common
    end date. Only Train is supplied to Seasonal Naive. Validation is used only
    for scoring. Existing MAPE policy applies: all-zero Validation raises
    ValueError; individual zero actuals are excluded from MAPE only.
    No files are written and no Test forecasts or Test scores are computed.
    """
    prepared = prepare_sales_data(load_sales_csv(path, encoding=encoding))
    train, validation, test = split_series(prepared, min_train_days=SEASON_LENGTH)
    forecasts = seasonal_naive(train)
    scores = evaluate_forecasts(validation, forecasts)
    return BaselineResult(train, validation, test, forecasts, scores)


def main() -> None:
    """Print Validation metrics from a local sales CSV."""
    parser = argparse.ArgumentParser(
        description="Seasonal Naive: оценка на Validation."
    )
    parser.add_argument(
        "csv_path", type=Path, help="CSV с колонками store,item,date,sales"
    )
    parser.add_argument("--encoding", default="utf-8", help="Кодировка CSV (utf-8)")
    args = parser.parse_args()
    try:
        result = run_baseline_pipeline(args.csv_path, encoding=args.encoding)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(json.dumps(result.validation_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
