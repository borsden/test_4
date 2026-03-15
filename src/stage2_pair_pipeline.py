"""Этап 2: нормализация пары и синхронизация потока котировок."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analytics_utils import save_dataframe, save_json
from src.normalization import build_normalized_pair
from src.pair_config import DEFAULT_PAIR
from src.settings import ANALYTICS_DIR

DEFAULT_STAGE1_OUTPUT = ANALYTICS_DIR / "stage1" / "quotes_clean.parquet"


def _quote_age_stats(series: pd.Series) -> dict[str, float]:
    """Сводит распределение quote age в миллисекундах."""
    return {
        "mean_ms": float(series.mean()),
        "median_ms": float(series.median()),
        "p90_ms": float(series.quantile(0.9)),
        "p99_ms": float(series.quantile(0.99)),
        "max_ms": float(series.max()),
        "share_above_500ms": float((series > 500).mean()),
        "share_above_1000ms": float((series > 1000).mean()),
    }


def _edge_stats(series: pd.Series) -> dict[str, float]:
    """Агрегирует ключевые квантильные характеристики executable edge."""
    positive = series[series > 0]
    negative = series[series < 0]
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "p90": float(series.quantile(0.9)),
        "p99": float(series.quantile(0.99)),
        "min": float(series.min()),
        "max": float(series.max()),
        "share_positive": float(len(positive) / len(series)),
        "share_negative": float(len(negative) / len(series)),
    }


def run_stage2(clean_quotes_path: Path, analytics_dir: Path) -> None:
    """Готовит нормализованный поток и сохраняет основную статистику."""
    stage_dir = analytics_dir / "stage2"
    stage_dir.mkdir(parents=True, exist_ok=True)

    quotes = pd.read_parquet(clean_quotes_path)
    pair = build_normalized_pair(quotes, pair_cfg=DEFAULT_PAIR)
    pair_df = pair.data

    save_dataframe(pair_df, stage_dir / "pair_state.parquet")
    save_dataframe(pair_df.head(1000), stage_dir / "pair_state_sample.csv")

    age_stats = {
        "primary": _quote_age_stats(pair_df["primary_quote_age_ms"]),
        "secondary": _quote_age_stats(pair_df["secondary_quote_age_ms"]),
    }
    edges_summary = {
        "edge_sell_primary": _edge_stats(pair_df["edge_sell_primary"]),
        "edge_buy_primary": _edge_stats(pair_df["edge_buy_primary"]),
    }

    summary = {
        "rows": int(len(pair_df)),
        "pair": DEFAULT_PAIR.name,
        "primary_symbol": DEFAULT_PAIR.primary_symbol,
        "secondary_symbol": DEFAULT_PAIR.secondary_symbol,
        "time_start": pair_df["ts"].min().isoformat(),
        "time_end": pair_df["ts"].max().isoformat(),
        "mid_spread_mean": float(pair_df["mid_spread"].mean()),
        "mid_spread_std": float(pair_df["mid_spread"].std()),
        "relative_mid_spread_mean": float(pair_df["relative_mid_spread"].mean()),
    }

    save_json(summary, stage_dir / "pair_state_summary.json")
    save_json(age_stats, stage_dir / "quote_age_stats.json")
    save_json(edges_summary, stage_dir / "edge_stats.json")


def parse_args() -> argparse.Namespace:
    """Готовит CLI для гибкого выбора входных данных."""
    parser = argparse.ArgumentParser(description="Этап 2 пайплайна: нормализация пары.")
    parser.add_argument(
        "--clean-quotes",
        type=Path,
        default=DEFAULT_STAGE1_OUTPUT,
        help="Путь к очищенным котировкам из этапа 1",
    )
    parser.add_argument(
        "--analytics-dir",
        type=Path,
        default=ANALYTICS_DIR,
        help="Корень папки analytics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage2(args.clean_quotes, args.analytics_dir)
