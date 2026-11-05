"""Convert sales observations to forecasting format."""

from urllib.parse import quote

import pandas as pd

from sales_forecasting.validation import (
    parse_dates,
    validate_forecasting_data,
    validate_raw_data,
)


def prepare_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, convert and sort sales without modifying the input table."""
    validate_raw_data(data)
    store = data["store"].map(lambda value: quote(str(value), safe=""))
    item = data["item"].map(lambda value: quote(str(value), safe=""))
    result = pd.DataFrame(
        {
            "unique_id": store + "/" + item,
            "ds": parse_dates(data["date"]),
            "y": pd.to_numeric(data["sales"]),
        }
    )
    validate_forecasting_data(result)
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
