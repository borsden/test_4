"""Vectorized statistical arbitrage backtest utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatArbParams:
    """Parameters that fully describe a stat-arb simulation."""

    z_entry: float
    window: int
    frequency: str = "1min"
    z_exit: float = 0.0
    intraday: bool = True
    min_periods: Optional[int] = None


@dataclass(frozen=True)
class BacktestResult:
    """Container for vectorized stat-arb simulation results."""

    params: StatArbParams
    trades: pd.DataFrame
    zscore: pd.Series
    position: pd.Series
    pnl: float
    metrics: Dict[str, float]


EdgeColumns = tuple[str, str]


def _resolve_edge_columns(df: pd.DataFrame, primary: Optional[str], secondary: Optional[str]) -> EdgeColumns:
    if primary and secondary:
        col_primary_secondary = f"edge_sell_{primary}_buy_{secondary}"
        col_secondary_primary = f"edge_sell_{secondary}_buy_{primary}"
        for column in (col_primary_secondary, col_secondary_primary):
            if column not in df.columns:
                msg = f"Требуемая колонка '{column}' отсутствует в данных"
                raise ValueError(msg)
        return (col_primary_secondary, col_secondary_primary)

    edge_columns = [col for col in df.columns if col.startswith("edge_sell_")]
    if len(edge_columns) < 2:
        msg = "Не удалось автоматически обнаружить edge-колонки"
        raise ValueError(msg)
    return edge_columns[0], edge_columns[1]


def _resample_edges(df: pd.DataFrame, frequency: str, edge_cols: EdgeColumns) -> pd.DataFrame:
    working = df.copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True)
    working = working.sort_values("ts").set_index("ts")

    resampled = working[list(edge_cols)].resample(frequency).last()
    resampled.dropna(subset=edge_cols, inplace=True, how="any")
    resampled["session"] = resampled.index.normalize()

    session_start = resampled["session"].diff().ne(pd.Timedelta(0))
    session_start = session_start.fillna(True)
    resampled["is_session_start"] = session_start
    return resampled.drop(columns="session")


def _compute_zscore(resampled: pd.DataFrame, params: StatArbParams, edge_col: str) -> pd.Series:
    min_periods = params.min_periods or params.window
    rolling_mean = resampled[edge_col].rolling(params.window, min_periods=min_periods).mean()
    rolling_std = resampled[edge_col].rolling(params.window, min_periods=min_periods).std(ddof=0)

    zscore = (resampled[edge_col] - rolling_mean) / rolling_std
    zscore = zscore.where(rolling_std.notna())
    zscore = zscore.copy()
    zscore[resampled["is_session_start"]] = np.nan
    return zscore


def _build_position(zscore: pd.Series, params: StatArbParams) -> tuple[pd.Series, pd.Series]:
    signals = pd.Series(np.nan, index=zscore.index, dtype=float)
    signals.loc[zscore > params.z_entry] = -1.0
    signals.loc[zscore < -params.z_entry] = 1.0

    exit_mask = zscore.abs() <= params.z_exit
    signals.loc[exit_mask] = 0.0

    signals_entry = signals.copy()
    signals_exit = signals.copy()

    if params.intraday:
        session = pd.Series(zscore.index.normalize(), index=zscore.index)
        session_end_mask = session != session.shift(-1)
        session_end_mask = session_end_mask.fillna(True)
        signals_exit.loc[session_end_mask] = 0.0

        session_start_mask = session != session.shift(1)
        session_start_mask = session_start_mask.fillna(True)
        signals_exit.loc[session_start_mask] = 0.0

    base_position = signals_entry.ffill().fillna(0.0)
    adjusted_position = signals_exit.ffill().fillna(0.0)
    return base_position, adjusted_position


def _extract_trades(
    resampled: pd.DataFrame,
    entry_position: pd.Series,
    exit_position: pd.Series,
    edge_cols: EdgeColumns,
) -> pd.DataFrame:
    entry_prev = entry_position.shift(fill_value=0.0)
    entry_mask = (entry_position != 0.0) & (entry_prev == 0.0)

    exit_prev = exit_position.shift(fill_value=0.0)
    exit_mask = (exit_position == 0.0) & (exit_prev != 0.0)

    entry_times = entry_position.index[entry_mask]
    exit_times = exit_position.index[exit_mask]

    n_trades = min(len(entry_times), len(exit_times))
    entry_times = entry_times[:n_trades]
    exit_times = exit_times[:n_trades]

    if n_trades == 0:
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

    entry_dirs = entry_position.loc[entry_times].to_numpy()
    entry_edge_primary = resampled.loc[entry_times, edge_cols[0]].to_numpy()
    entry_edge_secondary = resampled.loc[entry_times, edge_cols[1]].to_numpy()
    exit_edge_primary = resampled.loc[exit_times, edge_cols[0]].to_numpy()
    exit_edge_secondary = resampled.loc[exit_times, edge_cols[1]].to_numpy()

    entry_edge = np.where(entry_dirs == -1.0, entry_edge_primary, entry_edge_secondary)
    exit_edge = np.where(entry_dirs == -1.0, exit_edge_secondary, exit_edge_primary)
    pnl = entry_edge + exit_edge

    holding_minutes = (
        (exit_times.to_numpy() - entry_times.to_numpy()).astype("timedelta64[s]").astype(float) / 60.0
    )

    trades = pd.DataFrame(
        {
            "entry_ts": entry_times,
            "exit_ts": exit_times,
            "direction": entry_dirs,
            "entry_edge": entry_edge,
            "exit_edge": exit_edge,
            "holding_minutes": holding_minutes,
            "pnl": pnl,
        }
    )
    return trades


def _build_metrics(trades: pd.DataFrame) -> Dict[str, float]:
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
    params: StatArbParams,
    primary: Optional[str] = None,
    secondary: Optional[str] = None,
) -> BacktestResult:
    """Vectorized stat-arb backtest for a pair of edge columns."""

    edge_cols = _resolve_edge_columns(edges, primary, secondary)
    resampled = _resample_edges(edges, params.frequency, edge_cols)
    if resampled.empty:
        raise ValueError("Недостаточно данных после ресемплинга для backtest")

    zscore = _compute_zscore(resampled, params, edge_cols[0])
    entry_position, position = _build_position(zscore, params)
    trades = _extract_trades(resampled, entry_position, position, edge_cols)
    pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    metrics = _build_metrics(trades)

    return BacktestResult(
        params=params,
        trades=trades,
        zscore=zscore,
        position=position,
        pnl=pnl,
        metrics=metrics,
    )
