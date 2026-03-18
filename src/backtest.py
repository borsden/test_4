"""Векторные утилиты для stat-arb бэктестов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import ENTRY_EDGE_COL, EXIT_EDGE_COL

EDGE_COLUMNS = (ENTRY_EDGE_COL, EXIT_EDGE_COL)


@dataclass(frozen=True)
class BacktestResult:
    """Результат векторного прогона: сделки, позиции и агрегаты."""

    params: dict[str, float | int | bool | str | None]
    trades: pd.DataFrame
    zscore: pd.Series
    position: pd.Series
    pnl: float
    metrics: dict[str, float]


def _compute_signals(
    edges: pd.DataFrame,
    window: int | str,
    min_periods: int | None,
    z_in: float,
    theta_enter: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Считает z-scores и epnl для определения сигналов входа."""
    if isinstance(window, int):
        effective_min_periods = min_periods or window
    else:
        effective_min_periods = min_periods or 1

    entry_short = edges[ENTRY_EDGE_COL] # bid(A) - ask(B)
    exit_short = edges[EXIT_EDGE_COL]   # bid(B) - ask(A)
    
    entry_long = edges[EXIT_EDGE_COL]   # bid(B) - ask(A)
    exit_long = edges[ENTRY_EDGE_COL]   # bid(A) - ask(B)

    # Short stats
    rolling_entry_short = entry_short.rolling(window, min_periods=effective_min_periods)
    mu_entry_short = rolling_entry_short.mean()
    std_entry_short = rolling_entry_short.std(ddof=0)
    
    rolling_exit_short = exit_short.rolling(window, min_periods=effective_min_periods)
    mu_exit_short = rolling_exit_short.mean()

    z_entry_short = (entry_short - mu_entry_short) / std_entry_short
    z_entry_short = z_entry_short.where(std_entry_short > 0)
    epnl_short = entry_short + mu_exit_short

    # Long stats
    rolling_entry_long = entry_long.rolling(window, min_periods=effective_min_periods)
    mu_entry_long = rolling_entry_long.mean()
    std_entry_long = rolling_entry_long.std(ddof=0)
    
    rolling_exit_long = exit_long.rolling(window, min_periods=effective_min_periods)
    mu_exit_long = rolling_exit_long.mean()

    z_entry_long = (entry_long - mu_entry_long) / std_entry_long
    z_entry_long = z_entry_long.where(std_entry_long > 0)
    epnl_long = entry_long + mu_exit_long

    session = pd.Series(edges.index.normalize(), index=edges.index)
    session_start = session != session.shift(1)
    session_start = session_start.fillna(True)
    
    # Generate boolean masks for entry
    sig_short = (z_entry_short > z_in) & (epnl_short > theta_enter)
    sig_short[session_start] = False
    
    sig_long = (z_entry_long > z_in) & (epnl_long > theta_enter)
    sig_long[session_start] = False

    return sig_short, sig_long, epnl_short, epnl_long


def _build_position(
    sig_short: pd.Series,
    sig_long: pd.Series,
    epnl_short: pd.Series,
    epnl_long: pd.Series,
    edges: pd.DataFrame,
    alpha: float,
    intraday: bool,
) -> pd.Series:
    """Строит позицию по расчетной марже, отслеживая progress_t в цикле."""
    sig_s = sig_short.to_numpy(dtype=bool)
    sig_l = sig_long.to_numpy(dtype=bool)
    
    entry_s_arr = edges[ENTRY_EDGE_COL].to_numpy(dtype=float)
    exit_s_arr = edges[EXIT_EDGE_COL].to_numpy(dtype=float)
    
    entry_l_arr = edges[EXIT_EDGE_COL].to_numpy(dtype=float)
    exit_l_arr = edges[ENTRY_EDGE_COL].to_numpy(dtype=float)
    
    epnl_s_arr = epnl_short.to_numpy(dtype=float)
    epnl_l_arr = epnl_long.to_numpy(dtype=float)
    
    n = len(sig_s)
    position = np.zeros(n, dtype=float)
    
    if intraday:
        session = pd.Series(sig_short.index.normalize(), index=sig_short.index)
        session_ends = (session != session.shift(-1)).fillna(True).to_numpy(dtype=bool)
    else:
        session_ends = np.zeros(n, dtype=bool)

    pos = 0.0
    entry_open = 0.0
    epnl_open = 0.0

    for i in range(n):
        if session_ends[i]:
            pos = 0.0
            position[i] = 0.0
            continue
            
        if pos == 0.0:
            if sig_s[i]:
                pos = -1.0
                entry_open = entry_s_arr[i]
                epnl_open = epnl_s_arr[i]
            elif sig_l[i]:
                pos = 1.0
                entry_open = entry_l_arr[i]
                epnl_open = epnl_l_arr[i]
        elif pos == -1.0:
            pnl = entry_open + exit_s_arr[i]
            if epnl_open > 0:
                progress = pnl / epnl_open
                if progress >= alpha:
                    pos = 0.0
        elif pos == 1.0:
            pnl = entry_open + exit_l_arr[i]
            if epnl_open > 0:
                progress = pnl / epnl_open
                if progress >= alpha:
                    pos = 0.0
                    
        position[i] = pos

    return pd.Series(position, index=sig_short.index, dtype=float)


def _extract_trades(
    resampled: pd.DataFrame,
    position: pd.Series,
) -> pd.DataFrame:
    """Строит DataFrame сделок (включая перевороты)."""
    if position.empty:
        pos = position.copy()
    else:
        # Принудительно закрываем позу в самом конце для матчинга длин
        pos = position.copy()
        if pos.iloc[-1] != 0.0:
            pos.iloc[-1] = 0.0

    pos_arr = pos.to_numpy()
    prev_pos = np.roll(pos_arr, 1)
    if len(prev_pos) > 0:
        prev_pos[0] = 0.0

    is_exit = (prev_pos != 0.0) & (pos_arr != prev_pos)
    is_entry = (pos_arr != 0.0) & (pos_arr != prev_pos)

    exit_idx = np.where(is_exit)[0]
    entry_idx = np.where(is_entry)[0]

    if len(entry_idx) == 0:
        return pd.DataFrame(
            columns=[
                "entry_ts",
                "exit_ts",
                "direction",
                "entry_edge",
                "exit_edge",
                "holding_minutes",
                "pnl",
            ]
        )

    exec_entry_idx = entry_idx
    exec_exit_idx = exit_idx

    entry_times = pos.index[entry_idx]
    exit_times = pos.index[exit_idx]

    prices_primary_entry = resampled[ENTRY_EDGE_COL].to_numpy()
    prices_secondary_exit = resampled[EXIT_EDGE_COL].to_numpy()

    directions = pos_arr[entry_idx]

    entry_edge_primary = prices_primary_entry[exec_entry_idx]
    entry_edge_secondary = prices_secondary_exit[exec_entry_idx]
    exit_edge_primary = prices_primary_entry[exec_exit_idx]
    exit_edge_secondary = prices_secondary_exit[exec_exit_idx]

    entry_edge = np.where(directions == -1.0, entry_edge_primary, entry_edge_secondary)
    exit_edge = np.where(directions == -1.0, exit_edge_secondary, exit_edge_primary)

    pnl = entry_edge + exit_edge

    holding_minutes = (
        (exit_times.to_numpy() - entry_times.to_numpy()).astype("timedelta64[s]").astype(float) / 60.0
    )

    trades = pd.DataFrame(
        {
            "entry_ts": entry_times,
            "exit_ts": exit_times,
            "direction": directions,
            "entry_edge": entry_edge,
            "exit_edge": exit_edge,
            "holding_minutes": holding_minutes,
            "pnl": pnl,
        }
    )
    return trades


def _build_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Считает агрегаты по сделкам: win-rate и распределение PnL."""
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "mean_pnl": 0.0,
            "median_pnl": 0.0,
            "max_pnl": 0.0,
        }

    pnl = trades["pnl"]
    wins = (pnl > 0).mean()
    return {
        "trades": float(len(trades)),
        "win_rate": float(wins),
        "mean_pnl": float(pnl.mean()),
        "median_pnl": float(pnl.median()),
        "max_pnl": float(pnl.max()),
    }


def run_stat_arb_backtest(
    edges: pd.DataFrame,
    window: int | str,
    z_in: float,
    theta_enter: float,
    alpha: float,
    *,
    intraday: bool = True,
    min_periods: int | None = None,
) -> BacktestResult:
    """Выполняет полный stat-arb бэктест по модели progress_t."""
    missing = [column for column in EDGE_COLUMNS if column not in edges.columns]
    if missing:
        msg = f"В edges отсутствуют обязательные колонки: {missing}"
        raise ValueError(msg)

    if edges.empty:
        raise ValueError("Таблица edge'ов пуста")

    sig_short, sig_long, epnl_short, epnl_long = _compute_signals(edges, window, min_periods, z_in, theta_enter)
    position = _build_position(sig_short, sig_long, epnl_short, epnl_long, edges, alpha, intraday)
    trades = _extract_trades(edges, position)
    pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    metrics = _build_metrics(trades)

    # Combined representation of progress-based expected PnL for charts
    epnl_combined = epnl_short.where(epnl_short > epnl_long, -epnl_long)

    return BacktestResult(
        params={
            "window": window,
            "z_in": z_in,
            "theta_enter": theta_enter,
            "alpha": alpha,
            "intraday": intraday,
            "min_periods": min_periods,
        },
        trades=trades,
        zscore=epnl_combined,
        position=position,
        pnl=pnl,
        metrics=metrics,
    )
