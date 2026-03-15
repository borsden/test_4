"""Аналитика арбитражных комбинаций bid/ask между двумя инструментами."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.analytics_utils import save_dataframe, save_json


@dataclass(frozen=True)
class EdgeResult:
    """Хранит рассчитанный DataFrame edge'ов и набор метрик."""

    data: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    pair: Tuple[str, str]


def _choose_symbols(ds: xr.Dataset, primary: str | None, secondary: str | None) -> Tuple[str, str]:
    symbols = [str(value) for value in ds.coords["symbol"].values.tolist()]
    if len(symbols) < 2:
        msg = "Для арбитражного анализа требуется минимум два тикера"
        raise ValueError(msg)
    if primary is None or secondary is None:
        primary, secondary = symbols[:2]
    if primary not in symbols or secondary not in symbols:
        msg = f"Указанные тикеры {primary}/{secondary} отсутствуют в наборе {symbols}"
        raise ValueError(msg)
    if primary == secondary:
        msg = "Тикеры для арбитража должны различаться"
        raise ValueError(msg)
    return primary, secondary


def _edge_series(ds: xr.Dataset, primary: str, secondary: str) -> pd.DataFrame:
    price = ds["price"]
    qty = ds["qty"]

    bid_primary = price.sel(symbol=primary, side="bid")
    ask_primary = price.sel(symbol=primary, side="ask")
    bid_secondary = price.sel(symbol=secondary, side="bid")
    ask_secondary = price.sel(symbol=secondary, side="ask")

    qty_bid_primary = qty.sel(symbol=primary, side="bid")
    qty_ask_primary = qty.sel(symbol=primary, side="ask")
    qty_bid_secondary = qty.sel(symbol=secondary, side="bid")
    qty_ask_secondary = qty.sel(symbol=secondary, side="ask")

    edge_sell_primary_buy_secondary = bid_primary - ask_secondary
    edge_sell_secondary_buy_primary = bid_secondary - ask_primary

    qty_sell_primary_buy_secondary = xr.apply_ufunc(np.minimum, qty_bid_primary, qty_ask_secondary)
    qty_sell_secondary_buy_primary = xr.apply_ufunc(np.minimum, qty_bid_secondary, qty_ask_primary)

    dataset = xr.Dataset(
        data_vars={
            "edge_sell_primary_buy_secondary": edge_sell_primary_buy_secondary,
            "edge_sell_secondary_buy_primary": edge_sell_secondary_buy_primary,
            "qty_sell_primary_buy_secondary": qty_sell_primary_buy_secondary,
            "qty_sell_secondary_buy_primary": qty_sell_secondary_buy_primary,
        }
    )
    df = dataset.to_dataframe().reset_index()
    df.rename(columns={"time": "ts"}, inplace=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    rename_map = {
        "edge_sell_primary_buy_secondary": f"edge_sell_{primary}_buy_{secondary}",
        "edge_sell_secondary_buy_primary": f"edge_sell_{secondary}_buy_{primary}",
        "qty_sell_primary_buy_secondary": f"qty_sell_{primary}_buy_{secondary}",
        "qty_sell_secondary_buy_primary": f"qty_sell_{secondary}_buy_{primary}",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def _describe_series(series: pd.Series) -> Dict[str, float]:
    valid = series.dropna()
    if valid.empty:
        return {"count": 0}
    return {
        "count": int(len(valid)),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "p25": float(valid.quantile(0.25)),
        "p50": float(valid.quantile(0.5)),
        "p75": float(valid.quantile(0.75)),
        "p90": float(valid.quantile(0.9)),
        "p99": float(valid.quantile(0.99)),
        "max": float(valid.max()),
        "share_positive": float((valid > 0).mean()),
    }


def _build_metrics(df: pd.DataFrame, primary: str, secondary: str) -> Dict[str, Dict[str, float]]:
    metrics = {}
    edge_columns = [
        f"edge_sell_{primary}_buy_{secondary}",
        f"edge_sell_{secondary}_buy_{primary}",
    ]
    qty_columns = [
        f"qty_sell_{primary}_buy_{secondary}",
        f"qty_sell_{secondary}_buy_{primary}",
    ]
    for col in edge_columns + qty_columns:
        metrics[col] = _describe_series(df[col])
    return metrics


def compute_edges(
    ds: xr.Dataset,
    primary: str | None = None,
    secondary: str | None = None,
) -> EdgeResult:
    """Строит DataFrame edge'ов и статистику по двум направлениям арбитража."""
    chosen_primary, chosen_secondary = _choose_symbols(ds, primary, secondary)
    df = _edge_series(ds, chosen_primary, chosen_secondary)
    metrics = _build_metrics(df, chosen_primary, chosen_secondary)
    return EdgeResult(data=df, metrics=metrics, pair=(chosen_primary, chosen_secondary))


def save_arb_analysis(
    ds: xr.Dataset,
    analytics_dir: Path,
    primary: str | None = None,
    secondary: str | None = None,
) -> None:
    """Сохраняет edge-таблицу и метрики в analytics/arb."""
    result = compute_edges(ds, primary, secondary)
    arb_dir = analytics_dir / "arb"
    arb_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(result.data, arb_dir / "edges.csv")
    save_dataframe(result.data.head(1000), arb_dir / "edges_sample.csv")
    summary = {
        "pair": {
            "primary": result.pair[0],
            "secondary": result.pair[1],
        },
        "metrics": result.metrics,
    }
    save_json(summary, arb_dir / "edge_metrics.json")
