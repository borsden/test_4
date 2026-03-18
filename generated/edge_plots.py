"""Графики исполнимых спредов в обоих направлениях арбитража."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from generated.settings import ANALYTICS_DIR

DEFAULT_EDGES_PATH = ANALYTICS_DIR / "arb" / "edges.csv"
GAP_THRESHOLD_SECONDS = 3600


def _prepare_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит timestamps к UTC и сортирует наблюдения."""
    working = df.copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True)
    working.sort_values("ts", inplace=True)
    working.reset_index(drop=True, inplace=True)
    return working


def _mask_gaps(series: pd.Series, timestamps: pd.Series) -> pd.Series:
    """Вставляет NaN там, где gap между точками > threshold."""
    gaps = timestamps.diff().dt.total_seconds()
    mask = gaps > GAP_THRESHOLD_SECONDS
    series = series.copy()
    series.loc[mask] = pd.NA
    return series


def _plot_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    output_path: Path,
) -> None:
    timestamps = df["ts"].dt.tz_convert(None)
    plt.figure(figsize=(14, 5))
    for column in columns:
        values = _mask_gaps(df[column], df["ts"])
        plt.plot(timestamps, values, linewidth=0.7, label=column)
    plt.title(title)
    plt.xlabel("Время (UTC)")
    plt.ylabel("Edge, руб")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run_edge_plots(edges_path: Path, analytics_dir: Path) -> None:
    """Строит график двух направлений исполнимого edge."""
    df = pd.read_csv(edges_path)
    df = _prepare_edges(df)

    edge_columns = [col for col in df.columns if col.startswith("edge_sell_")]
    plot_dir = analytics_dir / "arb" / "plots"
    _plot_series(
        df,
        edge_columns,
        "Executable edge в обоих направлениях",
        plot_dir / "edges_overview.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Графики спредов в обоих направлениях арбитража.")
    parser.add_argument(
        "--edges",
        type=Path,
        default=DEFAULT_EDGES_PATH,
        help="Путь к edges.csv",
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
    run_edge_plots(args.edges, args.analytics_dir)
