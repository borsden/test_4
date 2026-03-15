"""Конвертация очищенных котировок в xarray Dataset и обратно."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import xarray as xr

REQUIRED_COLUMNS = [
    "ts",
    "symbol",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
]


def _prepare_aligned_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, List[str]]:
    """Создаёт полный MultiIndex (ts, symbol) и заполняет пропуски ffill."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        msg = f"В данных отсутствуют обязательные столбцы: {sorted(missing)}"
        raise ValueError(msg)

    working = df[REQUIRED_COLUMNS].copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True)
    working["last_update_ts"] = working["ts"]
    working = working.sort_values(["symbol", "ts"])  # локальная сортировка для ffill

    time_index = pd.DatetimeIndex(sorted(working["ts"].unique()))
    symbols = sorted(working["symbol"].unique())

    full_index = pd.MultiIndex.from_product([time_index, symbols], names=["ts", "symbol"])
    aligned = working.set_index(["ts", "symbol"]).reindex(full_index)
    aligned = aligned.groupby(level="symbol").ffill()

    aligned = aligned.reset_index()
    aligned["quote_age_ms"] = (
        (aligned["ts"] - aligned["last_update_ts"]).dt.total_seconds() * 1000
    )
    aligned = aligned.set_index(["ts", "symbol"])
    aligned.drop(columns=["last_update_ts"], inplace=True)
    return aligned, time_index, symbols


def _stack_bid_ask(
    aligned: pd.DataFrame,
    bid_column: str,
    ask_column: str,
    symbols: List[str],
) -> np.ndarray:
    """Формирует трёхмерный массив значений по осям (time, symbol, side)."""
    bid = aligned[bid_column].unstack(level="symbol")[symbols]
    ask = aligned[ask_column].unstack(level="symbol")[symbols]
    return np.stack([bid.to_numpy(), ask.to_numpy()], axis=-1)


def build_xarray_from_raw(df: pd.DataFrame) -> xr.Dataset:
    """Преобразует очищенные котировки в xarray Dataset."""
    aligned, time_index, symbols = _prepare_aligned_frame(df)

    price_array = _stack_bid_ask(aligned, "bid_price", "ask_price", symbols)
    qty_array = _stack_bid_ask(aligned, "bid_qty", "ask_qty", symbols)
    age_array = aligned["quote_age_ms"].unstack(level="symbol")[symbols].to_numpy()

    dataset = xr.Dataset(
        data_vars={
            "price": (("time", "symbol", "side"), price_array),
            "qty": (("time", "symbol", "side"), qty_array),
        },
        coords={
            "time": time_index,
            "symbol": symbols,
            "side": ["bid", "ask"],
        },
    )
    dataset["quote_age_ms"] = (("time", "symbol"), age_array)
    return dataset


def flatten_dataset(ds: xr.Dataset) -> pd.DataFrame:
    """Разворачивает Dataset в tidy DataFrame для сохранения в csv."""
    flat = ds.to_dataframe().reset_index()
    flat.rename(columns={"time": "ts"}, inplace=True)
    ts = pd.to_datetime(flat["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    flat["ts"] = ts
    flat = flat.sort_values(["ts", "symbol", "side"]).reset_index(drop=True)
    return flat


def build_xarray_from_flatten(flat_df: pd.DataFrame) -> xr.Dataset:
    """Собирает Dataset из плоского csv представления."""
    required = {"ts", "symbol", "side", "price", "qty"}
    missing = required - set(flat_df.columns)
    if missing:
        msg = f"В flatten-данных отсутствуют столбцы: {sorted(missing)}"
        raise ValueError(msg)

    working = flat_df.copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True)
    working["side"] = pd.Categorical(working["side"], categories=["bid", "ask"], ordered=True)
    ds = working.set_index(["ts", "symbol", "side"]).to_xarray()
    ds = ds.rename(ts="time").transpose("time", "symbol", "side")
    if "quote_age_ms" in ds:
        age = ds["quote_age_ms"].isel(side=0)
        ds = ds.drop_vars("quote_age_ms")
        ds["quote_age_ms"] = age
    return ds
