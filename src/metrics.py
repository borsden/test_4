"""Метрики и вспомогательные расчёты для базовой аналитики L1 котировок."""
from __future__ import annotations

from typing import List

import pandas as pd


def compute_spread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Формирует describe-таблицу для абсолютного и относительного спреда."""
    working = df.copy()
    working["spread"] = working["ask_price"] - working["bid_price"]
    working["mid"] = (working["ask_price"] + working["bid_price"]) / 2
    working = working[working["mid"] > 0]
    working["rel_spread"] = working["spread"] / working["mid"]

    rows: List[pd.Series] = []
    for column in ["spread", "rel_spread"]:
        described = working[column].describe(percentiles=[0.25, 0.5, 0.75])
        described.name = column
        rows.append(described)
    metrics = pd.DataFrame(rows)
    metrics.reset_index(inplace=True)
    metrics.rename(columns={"index": "stat"}, inplace=True)
    return metrics


def compute_time_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Оценивает распределение лагов между апдейтами котировок в миллисекундах."""
    records = []
    for symbol, group in df.groupby("symbol"):
        sorted_group = group.sort_values("ts")
        diffs = sorted_group["ts"].diff().dropna().dt.total_seconds() * 1000
        if diffs.empty:
            continue
        record = {
            "symbol": symbol,
            "updates": int(len(sorted_group)),
            "mean_gap_ms": float(diffs.mean()),
            "median_gap_ms": float(diffs.median()),
            "p90_gap_ms": float(diffs.quantile(0.9)),
            "max_gap_ms": float(diffs.max()),
            "min_gap_ms": float(diffs.min()),
        }
        records.append(record)
    return pd.DataFrame(records)
