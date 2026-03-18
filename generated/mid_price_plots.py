"""График mid-цен GLDG26 и GOLD-3.26."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generated.settings import ANALYTICS_DIR

DEFAULT_QUOTES_CSV = ANALYTICS_DIR / "xarray" / "quotes_state.csv"
GAP_THRESHOLD_SECONDS = 3600


def _load_mid_prices(quotes_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(quotes_csv)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    pivot = df.pivot_table(index="ts", columns=["symbol", "side"], values="price", aggfunc="first")
    pivot.columns = [f"{sym}_{side}" for sym, side in pivot.columns]
    pivot.dropna(subset=["GLDG26_bid", "GLDG26_ask", "GOLD-3.26_bid", "GOLD-3.26_ask"], inplace=True)
    pivot.reset_index(inplace=True)
    pivot["mid_GLDG26"] = (pivot["GLDG26_bid"] + pivot["GLDG26_ask"]) / 2
    pivot["mid_GOLD-3.26"] = (pivot["GOLD-3.26_bid"] + pivot["GOLD-3.26_ask"]) / 2
    return pivot[["ts", "mid_GLDG26", "mid_GOLD-3.26"]]


def _apply_gap_mask(series: pd.Series, timestamps: pd.Series) -> pd.Series:
    gaps = timestamps.diff().dt.total_seconds()
    mask = gaps > GAP_THRESHOLD_SECONDS
    series = series.copy()
    series.loc[mask] = pd.NA
    return series


def run_mid_price_plot(quotes_csv: Path, analytics_dir: Path) -> None:
    df = _load_mid_prices(quotes_csv)
    timestamps = df["ts"].dt.tz_convert(None)
    gldg = _apply_gap_mask(df["mid_GLDG26"], df["ts"])
    gold = _apply_gap_mask(df["mid_GOLD-3.26"], df["ts"])

    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, gldg, label="mid_GLDG26", linewidth=0.7)
    plt.plot(timestamps, gold, label="mid_GOLD-3.26", linewidth=0.7)
    plt.title("Mid prices GLDG26 vs GOLD-3.26")
    plt.xlabel("Время (UTC)")
    plt.ylabel("Цена, руб")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = analytics_dir / "arb" / "plots" / "mid_prices.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="График mid prices для GLDG26 и GOLD-3.26.")
    parser.add_argument(
        "--quotes",
        type=Path,
        default=DEFAULT_QUOTES_CSV,
        help="Путь к quotes_state.csv",
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
    run_mid_price_plot(args.quotes, args.analytics_dir)
