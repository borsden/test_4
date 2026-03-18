"""Загрузка сырых котировок из csv с нестандартным заголовком."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _extract_header(csv_path: Path) -> List[str]:
    """Читает первую строку файла и возвращает очищенный список колонок."""
    with csv_path.open("r", encoding="utf-8") as file:
        header_line = file.readline().strip()
    clean_line = header_line.replace('"', "")
    columns = [col for col in clean_line.split(";") if col]
    if not columns:
        msg = f"Не удалось распарсить заголовок в файле {csv_path}"
        raise ValueError(msg)
    return columns


def load_raw_quotes(csv_path: Path) -> pd.DataFrame:
    """Возвращает DataFrame с котировками, не выполняя очистку и фильтрацию."""
    columns = _extract_header(csv_path)
    df = pd.read_csv(
        csv_path,
        sep=";",
        skiprows=1,
        names=columns,
        dtype={
            "ts": str,
            "symbol": str,
            "bid_price": float,
            "bid_qty": float,
            "ask_price": float,
            "ask_qty": float,
        },
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df
