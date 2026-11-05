"""Run the baseline end to end on temporary synthetic sales, without downloads."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from sales_forecasting.pipeline import run_baseline_pipeline


def main() -> None:
    dates = pd.date_range("2024-01-01", periods=194)
    sales = pd.concat(
        [
            pd.DataFrame(
                {
                    "store": store,
                    "item": 1,
                    "date": dates,
                    "sales": [10 * store + day % 7 for day in range(len(dates))],
                }
            )
            for store in (1, 2, 3)
        ],
        ignore_index=True,
    )
    with TemporaryDirectory(prefix="sales-baseline-") as directory:
        path = Path(directory) / "sales.csv"
        sales.to_csv(path, index=False)
        result = run_baseline_pipeline(path)
    print(json.dumps(result.validation_metrics, indent=2))


if __name__ == "__main__":
    main()
