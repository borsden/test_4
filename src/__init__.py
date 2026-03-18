"""Минимальный API для загрузки котировок и stat-arb бэктеста."""

from .data import build_edge_frame, clean_quotes, load_raw_quotes, prepare_edges
from .backtest import BacktestResult, run_stat_arb_backtest

__all__ = [
    "BacktestResult",
    "build_edge_frame",
    "clean_quotes",
    "load_raw_quotes",
    "prepare_edges",
    "run_stat_arb_backtest",
]
