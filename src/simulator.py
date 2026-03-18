"""Grid search runner for stat-arb parameters."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import prepare_edges
from src.backtest import run_stat_arb_backtest


def _logspace_windows() -> Iterable[str | int]:
    """Builds five windows from 1 minute to 2 days on log scale."""
    minutes = np.logspace(np.log10(1), np.log10(2880), num=5)
    rounded = np.unique(np.round(minutes).astype(int))
    for value in sorted(rounded):
        if value < 60:
            yield f"{int(value)}min"
        elif value < 24 * 60:
            hours = max(1, value // 60)
            yield f"{int(hours)}h"
        else:
            days = max(1, value // (24 * 60))
            yield f"{int(days)}d"


def run_simulation(
    data_path: str | Path,
    *,
    frequency: str = "1min",
    z_in_values: Iterable[float] | None = None,
    theta_values: Iterable[float] | None = None,
    alpha_values: Iterable[float] | None = None,
    windows: Iterable[str | int] | None = None,
    intraday_options: Iterable[bool] | None = None,
    min_periods: int | None = None,
    save_path: str | Path = "data/results.csv",
) -> pd.DataFrame:
    """Runs grid search over supplied parameters and stores results."""
    edges = prepare_edges(data_path, frequency=frequency)

    z_grid = list(z_in_values or [1, 1.5, 2.0, 2.5])
    theta_grid = list(theta_values or [0.0, 25.0, 50.0, 75])
    alpha_grid = list(alpha_values or [0, 0.25, 0.5, 0.7, 0.9])
    window_grid = list(windows or _logspace_windows())
    intraday_grid = list(intraday_options or [True, False])

    records = []
    iterator = product(z_grid, theta_grid, alpha_grid, window_grid, intraday_grid)
    total = len(z_grid) * len(theta_grid) * len(alpha_grid) * len(window_grid) * len(intraday_grid)

    for z_in, theta, alpha, window, intraday in tqdm(iterator, total=total, desc="Grid search"):
        result = run_stat_arb_backtest(
            edges,
            window=window,
            z_in=z_in,
            theta_enter=theta,
            alpha=alpha,
            intraday=intraday,
            min_periods=min_periods,
        )
        record = {
            "z_in": z_in,
            "theta_enter": theta,
            "alpha": alpha,
            "window": window,
            "intraday": intraday,
            "pnl": result.pnl,
        }
        record.update(result.metrics)
        records.append(record)

    df = pd.DataFrame(records)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    DATA_PATH = Path("/mnt/Projects/test_4/data/quotes_202512260854(in).csv")
    run_simulation(DATA_PATH)
