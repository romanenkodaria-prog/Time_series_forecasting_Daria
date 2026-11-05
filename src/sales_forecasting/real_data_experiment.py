"""Bounded real-data comparison; no tuning and no Test evaluation."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

from sales_forecasting.evaluation import evaluate_forecasts
from sales_forecasting.models.baseline import seasonal_naive
from sales_forecasting.models.nbeats import (
    EXPERIMENT_CONFIG,
    fit_nbeats,
    predict_nbeats,
)
from sales_forecasting.pipeline import run_baseline_pipeline

DEFAULT_IDS = ("1/1", "1/2", "1/3")


def _fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def _period(frame: pd.DataFrame) -> dict:
    return {
        "start": str(frame["ds"].min().date()),
        "end": str(frame["ds"].max().date()),
        "rows": len(frame),
        "series": frame["unique_id"].nunique(),
    }


def run_experiment(
    path: str | Path, *, unique_ids: tuple[str, ...] = DEFAULT_IDS, max_steps: int = 5
) -> dict:
    """Score the existing full baseline pipeline, then compare on 3–5 fixed series.

    Selection is explicit, never based on sales or Validation/Test scores.
    Calendar boundaries are inherited from the full-data split. Only the selected
    Train is passed to N-BEATS; the Test frame is checked for mutation only.
    Research architecture is preserved, but steps are capped at 20 on CPU.
    """
    if not 3 <= len(unique_ids) <= 5 or len(set(unique_ids)) != len(unique_ids):
        raise ValueError("Выберите от 3 до 5 различных unique_id.")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 2 <= max_steps <= 20
    ):
        raise ValueError("Ограниченный эксперимент допускает от 2 до 20 шагов.")
    source = Path(path)
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    provenance_path = source.with_suffix(".source.json")
    provenance = {
        "url": "Локальный CSV; происхождение не подтверждено",
        "sha256": digest,
    }
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        if provenance.get("sha256") != digest:
            raise ValueError("SHA-256 CSV не совпадает с файлом происхождения данных.")

    start = perf_counter()
    full = run_baseline_pipeline(source)
    full_seconds = perf_counter() - start
    available = set(full.train["unique_id"])
    if not set(unique_ids) <= available:
        raise ValueError(
            f"Неизвестные unique_id: {sorted(set(unique_ids) - available)}"
        )
    test_before = _fingerprint(full.test)
    selected_train = full.train.loc[full.train["unique_id"].isin(unique_ids)].copy()
    selected_validation = full.validation.loc[
        full.validation["unique_id"].isin(unique_ids)
    ].copy()
    # Both models get the very same Train; the same Validation is scored twice.
    start = perf_counter()
    baseline = seasonal_naive(selected_train)
    baseline_scores = evaluate_forecasts(selected_validation, baseline)
    baseline_seconds = perf_counter() - start

    import torch

    config = replace(EXPERIMENT_CONFIG, max_steps=max_steps)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        start = perf_counter()
        fitted = fit_nbeats(selected_train, config)
        setup_and_fit_seconds = perf_counter() - start
        completed_steps = len(fitted.forecaster.models[0].train_trajectories)
        start = perf_counter()
        forecast = predict_nbeats(fitted)
        nbeats_scores = evaluate_forecasts(selected_validation, forecast)
        prediction_and_score_seconds = perf_counter() - start
    finally:
        torch.set_num_threads(previous_threads)
    if _fingerprint(full.test) != test_before:
        raise RuntimeError("Test был изменён во время эксперимента.")
    sizes = (
        pd.concat(
            [
                full.train[["unique_id"]],
                full.validation[["unique_id"]],
                full.test[["unique_id"]],
            ]
        )
        .groupby("unique_id")
        .size()
    )
    return {
        "executed_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "source": provenance,
        "dataset": {
            "path": source.as_posix(),
            "columns": columns,
            "rows": len(full.train) + len(full.validation) + len(full.test),
            "series": len(available),
            "start": _period(full.train)["start"],
            "end": _period(full.test)["end"],
            "min_rows_per_series": int(sizes.min()),
            "max_rows_per_series": int(sizes.max()),
            "stores": len({uid.split("/")[0] for uid in available}),
            "items": len({uid.split("/")[1] for uid in available}),
        },
        "periods": {
            name: _period(part)
            for name, part in [
                ("Train", full.train),
                ("Validation", full.validation),
                ("Test", full.test),
            ]
        },
        "full_baseline": {
            "metrics": full.validation_metrics,
            "pipeline_seconds": full_seconds,
        },
        "comparison": {
            "unique_ids": list(unique_ids),
            "train_rows": len(selected_train),
            "validation_rows": len(selected_validation),
            "baseline": {
                "metrics": baseline_scores,
                "predict_and_score_seconds": baseline_seconds,
            },
            "nbeats": {
                "config": asdict(config),
                "completed_steps": completed_steps,
                "metrics": nbeats_scores,
                "fit_seconds": fitted.fit_seconds,
                "setup_and_fit_seconds": setup_and_fit_seconds,
                "predict_and_score_seconds": prediction_and_score_seconds,
                "forecast_rows": len(forecast),
                "dates_per_series": forecast.groupby("unique_id")["ds"]
                .nunique()
                .to_dict(),
            },
        },
        "test_unchanged": True,
        "test_used_for_fit_or_selection": False,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "neuralforecast": version("neuralforecast"),
            "torch": torch.__version__,
            "pytorch_lightning": version("pytorch-lightning"),
            "pandas": pd.__version__,
            "accelerator": "cpu",
            "threads": 2,
        },
    }


def render_report(result: dict) -> str:
    """Render measured results without equating subset and full-panel scores."""
    data, comparison = result["dataset"], result["comparison"]
    neural = comparison["nbeats"]
    lines = [
        "# Реальный датасет: baseline и ограниченный N-BEATS",
        "",
        f"Запуск: {result['executed_at']} (Europe/Moscow).",
        "",
        "## Данные",
        "",
        f"Источник: {result['source']['url']}",
        f"SHA-256: `{result['source']['sha256']}`.",
        f"Локальный файл: `{data['path']}`; исключён из Git правилом `/data/`.",
        f"Колонки: {', '.join(data['columns'])}.",
        f"Строк: {data['rows']:,}; рядов: {data['series']}; магазинов: {data['stores']}; товаров: {data['items']}.",
        f"Даты: {data['start']} — {data['end']}; наблюдений на ряд: {data['min_rows_per_series']}–{data['max_rows_per_series']}.",
        "Валидация схемы, пропусков, дат, продаж, дубликатов и непрерывности ежедневной истории пройдена.",
        "",
        "## Общие календарные границы",
        "",
        "| Раздел | Начало | Конец | Строк во всей панели |",
        "| --- | --- | --- | --- |",
    ]
    for name, period in result["periods"].items():
        lines.append(
            f"| {name} | {period['start']} | {period['end']} | {period['rows']} |"
        )
    lines += [
        "",
        "## Validation: метрики",
        "",
        "| Область оценки | Модель | MAE | RMSE | MAPE, % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scope, model, scores in [
        (
            f"Все {data['series']} рядов",
            "Seasonal Naive (7)",
            result["full_baseline"]["metrics"],
        ),
        ("Выбранные ряды", "Seasonal Naive (7)", comparison["baseline"]["metrics"]),
        ("Те же выбранные ряды", "N-BEATS", neural["metrics"]),
    ]:
        lines.append(
            f"| {scope} | {model} | {scores['MAE']:.6f} | {scores['RMSE']:.6f} | {scores['MAPE']:.6f} |"
        )
    lines += [
        "",
        f"Ряды для сравнения, заданные до обучения: {', '.join(comparison['unique_ids'])}.",
        f"Обе модели: {comparison['train_rows']} строк Train и {comparison['validation_rows']} строк одной и той же Validation.",
        "Метрики усредняются по наблюдениям. MAPE исключает нулевой факт; полностью нулевой факт вызывает ошибку.",
        "",
        "## Параметры и время",
        "",
        "```json",
        json.dumps(neural["config"], indent=2, ensure_ascii=False),
        "```",
        "",
        "N-BEATS: стандартная архитектура Nixtla (identity/trend/seasonality, по одному блоку, слои 512/512); CPU, два потока, num_workers=0, без early stopping и внутреннего holdout.",
        f"Выполнено шагов: {neural['completed_steps']}. Горизонт проверен по ключам: {neural['forecast_rows']} строк, {neural['dates_per_series']}.",
        f"Baseline, весь pipeline на всех рядах (включая чтение и подготовку): {result['full_baseline']['pipeline_seconds']:.3f} с.",
        f"Baseline на выбранных рядах, прогноз и оценка: {comparison['baseline']['predict_and_score_seconds']:.3f} с.",
        f"N-BEATS, только fit: {neural['fit_seconds']:.3f} с; создание/первый импорт и fit: {neural['setup_and_fit_seconds']:.3f} с; прогноз и оценка: {neural['predict_and_score_seconds']:.3f} с.",
        "Время измерено perf_counter; эти интервалы включают разные операции и не являются сравнением скорости моделей.",
        "",
        "## Защита Test и ограничения",
        "",
        "Test участвует только в исходной проверке и выделении календарных границ. Он не передаётся модели или оценщику, его метрики не вычисляются; контрольный отпечаток после эксперимента не изменился.",
        "Ряды выбраны явно по идентификаторам, без отбора по продажам или метрикам. Выполнен один запуск с заранее ограниченным числом шагов; подбора параметров и выбора победителя нет.",
        "Результат N-BEATS на 3–5 рядах нельзя считать оценкой качества на всём датасете. Он не доказывает сходимость, превосходство модели или качество на Test. Честное сравнение моделей здесь относится только к выбранным рядам.",
        "Основная конфигурация max_steps=500 не менялась и не запускалась на 500 рядах.",
        "",
        "## Окружение",
        "",
        "```json",
        json.dumps(result["environment"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ограниченный эксперимент на реальных продажах."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--unique-ids", nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--report", type=Path, default=Path("docs/experiments/real_sales.md")
    )
    args = parser.parse_args()
    result = run_experiment(
        args.csv_path, unique_ids=tuple(args.unique_ids), max_steps=args.max_steps
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
