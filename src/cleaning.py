"""Очистка котировок и сбор диагностической статистики."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import pandas as pd


@dataclass
class CleaningStats:
    """Собирает агрегаты по этапам очистки котировок."""

    initial_rows: int
    null_ts_rows: int
    broken_bid_rows: int
    broken_ask_rows: int
    fully_broken_rows: int
    crossed_book_rows: int
    duplicate_rows: int
    final_rows: int
    symbol_rows: Dict[str, int]

    def to_dict(self) -> Dict[str, int]:
        """Возвращает статистику в виде словаря для сериализации."""
        mapping = asdict(self)
        mapping["symbol_rows"] = {
            str(symbol): int(count) for symbol, count in mapping["symbol_rows"].items()
        }
        return mapping


def clean_quotes(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Удаляет мусорные строки и возвращает DataFrame с базовыми фильтрами."""
    working = df.copy()
    initial_rows = len(working)

    null_ts_mask = working["ts"].isna()
    null_ts_rows = int(null_ts_mask.sum())
    working = working[~null_ts_mask]

    bid_broken = (working["bid_price"] <= 0) | (working["bid_qty"] <= 0)
    ask_broken = (working["ask_price"] <= 0) | (working["ask_qty"] <= 0)

    broken_bid_rows = int(bid_broken.sum())
    broken_ask_rows = int(ask_broken.sum())

    # Если обе стороны поломаны — выбрасываем запись полностью.
    fully_broken_mask = bid_broken & ask_broken
    fully_broken_rows = int(fully_broken_mask.sum())
    working = working[~fully_broken_mask].copy()

    # Для частично поломанных строк сохраняем информацию, но обнуляем соответствующую сторону.
    working.loc[bid_broken, ["bid_price", "bid_qty"]] = pd.NA
    working.loc[ask_broken, ["ask_price", "ask_qty"]] = pd.NA

    bid_valid = working["bid_price"].notna() & (working["bid_price"] > 0)
    ask_valid = working["ask_price"].notna() & (working["ask_price"] > 0)
    crossed_book_mask = bid_valid & ask_valid & (working["ask_price"] < working["bid_price"])
    crossed_book_rows = int(crossed_book_mask.sum())
    working = working[~crossed_book_mask]

    duplicate_mask = working.duplicated(subset=["ts", "symbol"], keep="last")
    duplicate_rows = int(duplicate_mask.sum())
    working = working[~duplicate_mask]

    working = working.sort_values(["symbol", "ts"]).reset_index(drop=True)

    stats = CleaningStats(
        initial_rows=initial_rows,
        null_ts_rows=null_ts_rows,
        broken_bid_rows=broken_bid_rows,
        broken_ask_rows=broken_ask_rows,
        fully_broken_rows=fully_broken_rows,
        crossed_book_rows=crossed_book_rows,
        duplicate_rows=duplicate_rows,
        final_rows=len(working),
        symbol_rows=working["symbol"].value_counts().sort_index().to_dict(),
    )
    return working, stats
