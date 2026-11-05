# ДЗ №1: инженерная подготовка проекта

Сопоставление составлено по требованиям, перечисленным в этапе 8 пользователем.
Отдельного исходного текста или критериев оценивания ДЗ в проекте нет;
соответствие дополнительным пунктам преподавателя здесь не утверждается.

| Требование | Подтверждение |
| --- | --- |
| Управление зависимостями через Poetry | [pyproject.toml](../pyproject.toml): стандартные метаданные `[project]`, Python `>=3.11,<3.12`, рабочие и dev-зависимости; [poetry.lock](../poetry.lock): точные версии и хеши |
| Изолированное виртуальное окружение | Команды `poetry env use 3.11`, `poetry install --with dev`, `poetry run ...` в [README](../README.md); локально Python 3.11.9 запускается из virtualenv Poetry |
| Линтер и форматирование Ruff | `[tool.ruff]` в `pyproject.toml`: Python 3.11, длина строки 88, проверки E4/E7/E9/F/I; `ruff check .`, `ruff format --check .` |
| pre-commit и Git hook | [.pre-commit-config.yaml](../.pre-commit-config.yaml): два локальных hook Ruff, проверяющих весь проект; hook установлен командой `poetry run pre-commit install` |
| Автоматические тесты | [tests/](../tests/): загрузка/подготовка данных, ошибки, календарные разделы, метрики, ключи прогнозов, baseline, pipeline и обвязка N-BEATS |
| Рефакторинг исследовательского кода в пакет | [src/sales_forecasting/](../src/sales_forecasting/): независимые модули данных, валидации, разделения, метрик, оценки и моделей; параметры, обучение и прогнозирование разделены |
| Сквозной воспроизводимый запуск | [pipeline.py](../src/sales_forecasting/pipeline.py) и [examples/baseline_demo.py](../examples/baseline_demo.py); синтетический пример не требует скачивания данных |
| CI при push и Pull Request | [.github/workflows/ci.yml](../.github/workflows/ci.yml): Windows, Python 3.11 x64, установка по lock-файлу, строгая проверка Poetry, pytest, Ruff и pre-commit |
| CI без больших CSV, GPU и длительного обучения | Синтетические данные в тестах; `RUN_NBEATS_SMOKE=0` в workflow; реальные эксперименты запускаются отдельными командами |
| Исключение данных и артефактов | [.gitignore](../.gitignore): `data/`, модели/веса, `.venv`, кеши, временные файлы и логи; `data/raw/train.csv` не отслеживается Git |
| Документация и реальный эксперимент | [README](../README.md), [отчёт](experiments/real_sales.md): установка, команды, параметры, метрики, время и ограничения эксперимента |

Вычислительная логика оформлена функциями и dataclass, без обучения при импорте.
CSV и ошибки данных проверяются явно; хронологические границы общие для рядов.
Метрики используют строгое сопоставление прогнозов с фактом по ключам. Test
не передаётся обучению или подбору; тесты проверяют отсутствие утечки данных.
Основная конфигурация N-BEATS отделена от быстрых проверок.

## Как проверить

Из корня проекта после установки Python 3.11 и Poetry 2.5.1:

```powershell
poetry env use 3.11
poetry install --with dev
poetry run pre-commit install
poetry run python examples/baseline_demo.py
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run pre-commit run --all-files
poetry check --lock --strict
```

Синтетический пример baseline должен вывести нулевые MAE, RMSE и MAPE.
Реальный обучающий smoke-тест N-BEATS по умолчанию пропускается.

## Результаты локальной проверки

5 ноября 2026, Windows x64, Python 3.11.9, Poetry 2.5.1:

- Установка через `poetry install --with dev` прошла без обновления версий пакетов.
- `poetry run pytest`: **182 passed, 1 skipped** (обучающий smoke-тест).
- `poetry run ruff check .` и `poetry run ruff format --check .`: успешно.
- `poetry run pre-commit run --all-files`: оба hook прошли, не пропущены.
- `poetry check` и `poetry check --lock --strict`: успешно, без предупреждений.
- `poetry build`: успешно созданы wheel и sdist в исключённой из Git `dist/`.
- Пример baseline вывел MAE=0, RMSE=0, MAPE=0; CLI обоих сценариев проверены
  командой `--help`, без повторного обучения.
- `git check-ignore` подтвердил исключение данных, окружения, кешей, весов,
  временных файлов и сборок; эти артефакты не отслеживаются Git.

Полное обучение N-BEATS при этой проверке не запускалось. YAML workflow и его
команды проверены локально; запуск на GitHub требует отдельной публикации.

Git hook локален и не входит в коммит: его устанавливают заново в каждом клоне.
Все файлы пакета, тесты, настройки, workflow и документация должны быть включены
в будущий коммит; данные и веса — исключены. На этом этапе индекс Git не менялся,
push, merge и создание PR не выполнялись. Удалённый запуск GitHub Actions
ещё не подтверждён; workflow подготовлен и проверен локально.
