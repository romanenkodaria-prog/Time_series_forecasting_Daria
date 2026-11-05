"""Read and validate local sales CSV files."""

from pathlib import Path

import pandas as pd

from sales_forecasting.validation import validate_raw_data


def load_sales_csv(path: str | Path, *, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a comma-separated CSV and validate its raw sales schema."""
    source = Path(path)
    try:
        data = pd.read_csv(source, encoding=encoding, dtype={"store": str, "item": str})
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV-файл не найден: {source}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Не удалось прочитать CSV-файл {source}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Не удалось открыть CSV-файл {source}: {exc}") from exc
    try:
        validate_raw_data(data)
    except ValueError as exc:
        raise ValueError(f"Некорректные данные в CSV {source}: {exc}") from exc
    return data
