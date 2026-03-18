"""Утилиты для загрузки котировок, очистки и расчёта edge-серий."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

RAW_COLUMNS = ["ts", "symbol", "bid_price", "bid_qty", "ask_price", "ask_qty"]
DEFAULT_PRIMARY = "GLDG26"
DEFAULT_SECONDARY = "GOLD-3.26"
# Вход: продаём primary (GLDG26 на B3) и покупаем secondary (GOLD-3.26 на MOEX)
ENTRY_EDGE_COL = "entry_edge"
# Выход: обратная нога — откупаем primary и продаём secondary
EXIT_EDGE_COL = "exit_edge"
ENTRY_QTY_COL = "entry_qty"
EXIT_QTY_COL = "exit_qty"


def _extract_header(csv_path: Path | str) -> list[str]:
    """Возвращает список колонок из первой строки csv (разделитель `;`)."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as file:
        header_line = file.readline().strip()
    clean_line = header_line.replace('"', "")
    columns = [part for part in clean_line.split(";") if part]
    if not columns:
        msg = f"Не удалось прочитать заголовок в файле {csv_path}"
        raise ValueError(msg)
    return columns


def load_raw_quotes(csv_path: Path | str) -> pd.DataFrame:
    """Читает исходный csv с котировками и приводит таймстемпы к UTC."""
    path = Path(csv_path)
    columns = _extract_header(path)
    df = pd.read_csv(
        path,
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


def clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет мусорные записи и оставляет только консистентные L1 котировки."""
    working = df.copy()

    null_ts_mask = working["ts"].isna()
    working = working[~null_ts_mask]

    bid_broken = (working["bid_price"] <= 0) | (working["bid_qty"] <= 0)
    ask_broken = (working["ask_price"] <= 0) | (working["ask_qty"] <= 0)

    fully_broken = bid_broken & ask_broken
    working = working[~fully_broken].copy()

    working.loc[bid_broken, ["bid_price", "bid_qty"]] = pd.NA
    working.loc[ask_broken, ["ask_price", "ask_qty"]] = pd.NA

    bid_valid = working["bid_price"].notna() & (working["bid_price"] > 0)
    ask_valid = working["ask_price"].notna() & (working["ask_price"] > 0)
    crossed = bid_valid & ask_valid & (working["ask_price"] < working["bid_price"])
    working = working[~crossed]

    working = working.sort_values(["symbol", "ts"])
    duplicate_mask = working.duplicated(subset=["ts", "symbol"], keep="last")
    working = working[~duplicate_mask]
    working = working.reset_index(drop=True)

    return working


def _symbol_frame(clean_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Возвращает отсортированную по времени таблицу котировок одного тикера."""
    subset = clean_df[clean_df["symbol"] == symbol].copy()
    if subset.empty:
        msg = f"В очищенных данных отсутствует тикер {symbol}"
        raise ValueError(msg)
    subset = subset.sort_values("ts")
    subset = subset.drop_duplicates("ts", keep="last")
    subset = subset.set_index("ts")
    return subset[RAW_COLUMNS[2:]]


def _align_pair(clean_df: pd.DataFrame, primary: str, secondary: str) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """Пересекает две временные серии котировок и ffill-ит пропуски по общей сетке."""
    primary_frame = _symbol_frame(clean_df, primary)
    secondary_frame = _symbol_frame(clean_df, secondary)
    union_index = primary_frame.index.union(secondary_frame.index)
    primary_aligned = primary_frame.reindex(union_index).ffill()
    secondary_aligned = secondary_frame.reindex(union_index).ffill()
    return union_index, primary_aligned, secondary_aligned


def build_edge_frame(
    clean_df: pd.DataFrame,
    primary: str = DEFAULT_PRIMARY,
    secondary: str = DEFAULT_SECONDARY,
) -> pd.DataFrame:
    """Строит серию entry/exit edge'ов между двумя тикерами.

    entry_edge = bid(primary) - ask(secondary)  → sell primary / buy secondary.
    exit_edge  = bid(secondary) - ask(primary) → buy back primary / sell secondary.
    """
    timeline, primary_frame, secondary_frame = _align_pair(clean_df, primary, secondary)

    bid_primary = primary_frame["bid_price"]
    ask_primary = primary_frame["ask_price"]
    bid_secondary = secondary_frame["bid_price"]
    ask_secondary = secondary_frame["ask_price"]

    qty_bid_primary = primary_frame["bid_qty"]
    qty_ask_primary = primary_frame["ask_qty"]
    qty_bid_secondary = secondary_frame["bid_qty"]
    qty_ask_secondary = secondary_frame["ask_qty"]

    entry_edge_series = bid_primary - ask_secondary
    exit_edge_series = bid_secondary - ask_primary

    entry_qty = np.minimum(qty_bid_primary, qty_ask_secondary)
    exit_qty = np.minimum(qty_bid_secondary, qty_ask_primary)

    df = pd.DataFrame(
        index=timeline,
        data={
            ENTRY_EDGE_COL: entry_edge_series,
            EXIT_EDGE_COL: exit_edge_series,
            ENTRY_QTY_COL: entry_qty,
            EXIT_QTY_COL: exit_qty,
        },
    )
    df.index.name = "ts"
    df = df.sort_index()
    df = df.dropna(how="all")
    return df


def prepare_edges(
    csv_path: Path | str,
    primary: str = DEFAULT_PRIMARY,
    secondary: str = DEFAULT_SECONDARY,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Возвращает DataFrame edge'ов (DatetimeIndex) после загрузки и очистки.

    Если frequency указана, дополнительно агрегируем данные методом `.resample().last()`
    и выбрасываем строки, где обе ноги отсутствуют.
    """
    raw = load_raw_quotes(csv_path)
    cleaned = clean_quotes(raw)
    edges = build_edge_frame(cleaned, primary=primary, secondary=secondary)

    if frequency:
        resampled = edges.resample(frequency).last()
        resampled.dropna(subset=[ENTRY_EDGE_COL, EXIT_EDGE_COL], inplace=True, how="any")
        edges = resampled

    return edges
