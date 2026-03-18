"""Утилиты для сохранения результатов аналитики по этапам пайплайна."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Гарантирует наличие каталога перед записью файла."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Сохраняет структуру данных в json с читаемым форматированием."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет DataFrame в csv или parquet, в зависимости от расширения файла."""
    ensure_parent_dir(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        msg = f"Неизвестный формат файла {path}"
        raise ValueError(msg)
