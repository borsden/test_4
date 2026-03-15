"""Этап 3: аналитика нормализованной пары и подготовка базовых метрик."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.analytics_utils import save_dataframe, save_json
from src.settings import ANALYTICS_DIR

DEFAULT_STAGE2_PATH = ANALYTICS_DIR / "stage2" / "pair_state.parquet"


def _zscore(series: pd.Series, window: int = 2000, min_periods: int = 500) -> pd.Series:
    """Считает rolling z-score для заданного спреда."""
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    z = (series - rolling_mean) / rolling_std
    return z


def _spread_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит describe-таблицу для mid/micro/relative спредов."""
    metrics = {}
    for column in ["mid_spread", "micro_spread", "relative_mid_spread"]:
        described = df[column].describe(percentiles=[0.25, 0.5, 0.75]).rename(column)
        metrics[column] = described
    table = pd.DataFrame(metrics)
    table.reset_index(inplace=True)
    table.rename(columns={"index": "stat"}, inplace=True)
    return table


def _edge_threshold_table(
    series: pd.Series,
    thresholds: Iterable[float],
    mask: pd.Series,
) -> pd.DataFrame:
    """Рассчитывает долю исполнимых возможностей выше разных порогов."""
    records = []
    total_raw = len(series)
    total_filtered = int(mask.sum())
    for threshold in thresholds:
        raw_share = float((series > threshold).sum() / total_raw)
        filtered_share = float((series[mask] > threshold).sum() / max(total_filtered, 1))
        records.append(
            {
                "threshold": threshold,
                "raw_share": raw_share,
                "filtered_share": filtered_share,
            }
        )
    return pd.DataFrame(records)


def _staleness_table(df: pd.DataFrame, limits_ms: Iterable[int]) -> pd.DataFrame:
    """Сводит долю наблюдений, где обе котировки свежие."""
    records = []
    total = len(df)
    for limit in limits_ms:
        mask = (df["primary_quote_age_ms"] <= limit) & (df["secondary_quote_age_ms"] <= limit)
        records.append(
            {
                "threshold_ms": limit,
                "share_alive": float(mask.sum() / total),
                "count_alive": int(mask.sum()),
            }
        )
    return pd.DataFrame(records)


def run_stage3(pair_state_path: Path, analytics_dir: Path) -> None:
    """Формирует углублённую аналитику для нормализованной пары."""
    stage_dir = analytics_dir / "stage3"
    stage_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_parquet(pair_state_path)
    pair_df = pair_df.sort_values("ts").reset_index(drop=True)

    pair_df["mid_spread_z"] = _zscore(pair_df["mid_spread"])

    spread_table = _spread_describe(pair_df)
    save_dataframe(spread_table, stage_dir / "spread_describe.csv")

    z = pair_df["mid_spread_z"].dropna()
    z_stats = {
        "count": int(len(z)),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "share_abs_gt_1": float((z.abs() > 1).mean()),
        "share_abs_gt_2": float((z.abs() > 2).mean()),
        "share_abs_gt_3": float((z.abs() > 3).mean()),
    }
    save_json(z_stats, stage_dir / "zscore_stats.json")

    freshness_mask = (pair_df["primary_quote_age_ms"] <= 500) & (
        pair_df["secondary_quote_age_ms"] <= 500
    )
    edge_table = _edge_threshold_table(
        pair_df["edge_sell_primary"],
        thresholds=[5, 10, 15, 20, 25, 30, 40, 50],
        mask=freshness_mask,
    )
    save_dataframe(edge_table, stage_dir / "edge_thresholds.csv")

    staleness_limits = [100, 250, 500, 1000, 2500, 5000]
    staleness = _staleness_table(pair_df, staleness_limits)
    save_dataframe(staleness, stage_dir / "staleness_table.csv")

    feature_columns = [
        "ts",
        "mid_spread",
        "micro_spread",
        "relative_mid_spread",
        "mid_spread_z",
        "edge_sell_primary",
        "edge_buy_primary",
        "primary_quote_age_ms",
        "secondary_quote_age_ms",
    ]
    feature_df = pair_df[feature_columns]
    save_dataframe(feature_df, stage_dir / "pair_features.parquet")

    notes = [
        "Этап 3 — описательная аналитика пары",
        f"Число наблюдений: {len(pair_df)}",
        f"Z-score рассчитан по rolling окну 2000 тиков (мин 500).",
        "Фреснесс-фильтр в edge_thresholds: обе котировки младше 500 мс.",
    ]
    (stage_dir / "notes.txt").write_text("\n".join(notes), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Поддерживает переопределение путей через CLI."""
    parser = argparse.ArgumentParser(description="Этап 3: аналитика нормализованного спреда.")
    parser.add_argument(
        "--pair-state",
        type=Path,
        default=DEFAULT_STAGE2_PATH,
        help="Путь к parquet с нормализованной парой",
    )
    parser.add_argument(
        "--analytics-dir",
        type=Path,
        default=ANALYTICS_DIR,
        help="Корневая папка analytics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage3(args.pair_state, args.analytics_dir)
