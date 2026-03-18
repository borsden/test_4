"""Основной пайплайн: загрузка raw котировок, очистка и конвертация в xarray."""
from __future__ import annotations

import argparse
from pathlib import Path

from generated.analytics_utils import ensure_parent_dir, save_dataframe, save_json
from generated.arb_analysis import save_arb_analysis
from generated.cleaning import clean_quotes
from generated.data_loading import load_raw_quotes
from generated.metrics import compute_spread_metrics, compute_time_gap_stats
from generated.settings import ANALYTICS_DIR, DEFAULT_RAW_QUOTES_FILE
from generated.xarray_builder import build_xarray_from_raw, flatten_dataset


def _save_cleaning_artifacts(
    cleaned_df,
    stats,
    analytics_dir: Path,
    source_path: Path,
) -> Path:
    stage_dir = analytics_dir / "main"
    stage_dir.mkdir(parents=True, exist_ok=True)

    save_json(stats.to_dict(), stage_dir / "cleaning_stats.json")
    spread_metrics = compute_spread_metrics(cleaned_df)
    save_dataframe(spread_metrics, stage_dir / "spread_metrics.csv")
    time_gap_stats = compute_time_gap_stats(cleaned_df)
    if not time_gap_stats.empty:
        save_dataframe(time_gap_stats, stage_dir / "time_gap_stats.csv")

    save_dataframe(cleaned_df, stage_dir / "quotes_clean.csv")
    save_dataframe(cleaned_df.head(100), stage_dir / "quotes_sample.csv")

    summary_lines = [
        "Основной пайплайн — базовая очистка котировок",
        f"Источник: {source_path}",
        f"Всего строк до очистки: {stats.initial_rows}",
        f"Всего строк после очистки: {stats.final_rows}",
        f"Количество тикеров: {len(stats.symbol_rows)}",
    ]
    summary_path = stage_dir / "summary.txt"
    ensure_parent_dir(summary_path)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return stage_dir / "quotes_clean.csv"


def _save_xarray_artifacts(ds, analytics_dir: Path) -> None:
    xarray_dir = analytics_dir / "xarray"
    xarray_dir.mkdir(parents=True, exist_ok=True)

    flat_df = flatten_dataset(ds)
    save_dataframe(flat_df, xarray_dir / "quotes_state.csv")
    save_dataframe(flat_df.head(1000), xarray_dir / "quotes_state_sample.csv")

    symbols = []
    if "symbol" in ds.coords:
        symbols = [str(value) for value in ds.coords["symbol"].values.tolist()]
    summary = {
        "rows": int(len(flat_df)),
        "timestamps": int(ds.sizes.get("time", 0)),
        "symbols": symbols,
    }
    save_json(summary, xarray_dir / "summary.json")


def run_pipeline(raw_path: Path, analytics_dir: Path) -> None:
    """Запускает end-to-end обработку: raw → clean → xarray/csv."""
    raw_df = load_raw_quotes(raw_path)
    cleaned_df, stats = clean_quotes(raw_df)
    clean_csv_path = _save_cleaning_artifacts(cleaned_df, stats, analytics_dir, raw_path)

    ds = build_xarray_from_raw(cleaned_df)
    _save_xarray_artifacts(ds, analytics_dir)
    save_arb_analysis(ds, analytics_dir)

    print(f"Готово. Очищенные данные → {clean_csv_path}, xarray сохранён в analytics/xarray")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Основной пайплайн обработки L1 котировок.")
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
        help="Каталог для сохранения артефактов",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.analytics_dir)
