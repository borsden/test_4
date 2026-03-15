"""Пайплайн первого этапа: загрузка, очистка и sanity-проверки котировок."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analytics_utils import ensure_parent_dir, save_dataframe, save_json
from src.cleaning import clean_quotes
from src.data_loading import load_raw_quotes
from src.metrics import compute_spread_metrics, compute_time_gap_stats
from src.settings import ANALYTICS_DIR, DEFAULT_RAW_QUOTES_FILE


def run_stage1(input_path: Path, analytics_dir: Path) -> None:
    """Запускает полный цикл первичной обработки котировок."""
    stage_dir = analytics_dir / "stage1"
    stage_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_quotes(input_path)
    cleaned_df, stats = clean_quotes(raw_df)

    save_json(stats.to_dict(), stage_dir / "cleaning_stats.json")

    spread_metrics = compute_spread_metrics(cleaned_df)
    save_dataframe(spread_metrics, stage_dir / "spread_metrics.csv")

    time_gap_stats = compute_time_gap_stats(cleaned_df)
    if not time_gap_stats.empty:
        save_dataframe(time_gap_stats, stage_dir / "time_gap_stats.csv")

    save_dataframe(cleaned_df, stage_dir / "quotes_clean.parquet")
    sample = cleaned_df.head(100)
    save_dataframe(sample, stage_dir / "quotes_sample.csv")

    summary_lines = [
        "Этап 1 — базовая очистка котировок",
        f"Источник: {input_path}",
        f"Всего строк до очистки: {stats.initial_rows}",
        f"Всего строк после очистки: {stats.final_rows}",
        f"Количество тикеров: {len(stats.symbol_rows)}",
    ]
    summary_path = stage_dir / "summary.txt"
    ensure_parent_dir(summary_path)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Разбирает CLI-аргументы для гибкого запуска пайплайна."""
    parser = argparse.ArgumentParser(
        description=(
            "Первичная обработка L1- котировок: очистка, описательная статистика и"
            " сохранение результатов в analytics/stage1."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_QUOTES_FILE,
        help="Путь к исходному csv с котировками",
    )
    parser.add_argument(
        "--analytics-dir",
        type=Path,
        default=ANALYTICS_DIR,
        help="Корневая папка, куда складываются результаты аналитики",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage1(args.input, args.analytics_dir)
