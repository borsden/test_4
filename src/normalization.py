"""Нормализация котировок и сбор синхронизированного состояния пары."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.instruments import InstrumentMeta, DEFAULT_INSTRUMENTS
from src.pair_config import PairConfig, DEFAULT_PAIR


@dataclass(frozen=True)
class NormalizedPair:
    """Результат подготовки синхронизированного состояния пары."""

    data: pd.DataFrame
    primary_meta: InstrumentMeta
    secondary_meta: InstrumentMeta


def normalize_quotes(quotes: pd.DataFrame, meta: InstrumentMeta) -> pd.DataFrame:
    """Обогащает котировки экономикой контракта и пересчитывает mid/microprice."""
    filtered = quotes[quotes["symbol"] == meta.symbol].copy()
    if filtered.empty:
        msg = f"Нет котировок для символа {meta.symbol}"
        raise ValueError(msg)

    filtered["mid_price"] = (filtered["bid_price"] + filtered["ask_price"]) / 2
    qty_sum = filtered["bid_qty"] + filtered["ask_qty"]
    micro = (filtered["ask_price"] * filtered["bid_qty"] + filtered["bid_price"] * filtered["ask_qty"])
    filtered["microprice"] = micro / qty_sum
    filtered.loc[qty_sum == 0, "microprice"] = filtered.loc[qty_sum == 0, "mid_price"]

    scale = meta.multiplier * meta.fx_to_rub
    filtered["contract_bid"] = filtered["bid_price"] * scale
    filtered["contract_ask"] = filtered["ask_price"] * scale
    filtered["contract_mid"] = filtered["mid_price"] * scale
    filtered["contract_microprice"] = filtered["microprice"] * scale
    filtered["price_scale"] = scale
    filtered["tick_size"] = meta.tick_size
    filtered["tick_value"] = meta.tick_value
    filtered["currency"] = meta.currency
    filtered["fx_to_rub"] = meta.fx_to_rub
    filtered["multiplier"] = meta.multiplier
    filtered["last_update_ts"] = filtered["ts"]
    return filtered


def _merge_streams(
    primary: pd.DataFrame, secondary: pd.DataFrame, pair_cfg: PairConfig
) -> pd.DataFrame:
    """Объединяет два потока котировок в единый временной ряд с ffill."""
    primary_idx = primary.set_index("ts").add_prefix("primary_")
    secondary_idx = secondary.set_index("ts").add_prefix("secondary_")

    union_index = primary_idx.index.union(secondary_idx.index).drop_duplicates().sort_values()
    primary_ffill = primary_idx.reindex(union_index).ffill()
    secondary_ffill = secondary_idx.reindex(union_index).ffill()

    merged = pd.concat([primary_ffill, secondary_ffill], axis=1)
    merged = merged.dropna(subset=["primary_bid_price", "secondary_bid_price"])
    merged.reset_index(inplace=True)
    merged.rename(columns={"index": "ts"}, inplace=True)

    merged["primary_quote_age_ms"] = (
        (merged["ts"] - merged["primary_last_update_ts"]).dt.total_seconds() * 1000
    )
    merged["secondary_quote_age_ms"] = (
        (merged["ts"] - merged["secondary_last_update_ts"]).dt.total_seconds() * 1000
    )

    hedge_ratio = pair_cfg.hedge_ratio
    merged["mid_spread"] = (
        merged["primary_contract_mid"] - hedge_ratio * merged["secondary_contract_mid"]
    )
    merged["micro_spread"] = (
        merged["primary_contract_microprice"] - hedge_ratio * merged["secondary_contract_microprice"]
    )
    denom = (merged["primary_contract_mid"] + hedge_ratio * merged["secondary_contract_mid"]) / 2
    merged["relative_mid_spread"] = merged["mid_spread"] / denom

    merged["edge_sell_primary"] = (
        merged["primary_contract_bid"] - hedge_ratio * merged["secondary_contract_ask"]
    )
    merged["edge_buy_primary"] = (
        hedge_ratio * merged["secondary_contract_bid"] - merged["primary_contract_ask"]
    )
    return merged


def build_normalized_pair(
    quotes: pd.DataFrame,
    pair_cfg: PairConfig = DEFAULT_PAIR,
    instruments: Dict[str, InstrumentMeta] | None = None,
) -> NormalizedPair:
    """Готовит нормализованный поток для выбранной пары инструментов."""
    instruments = instruments or DEFAULT_INSTRUMENTS
    primary_meta = instruments[pair_cfg.primary_symbol]
    secondary_meta = instruments[pair_cfg.secondary_symbol]

    primary_quotes = normalize_quotes(quotes, primary_meta)
    secondary_quotes = normalize_quotes(quotes, secondary_meta)

    merged = _merge_streams(primary_quotes, secondary_quotes, pair_cfg)
    return NormalizedPair(data=merged, primary_meta=primary_meta, secondary_meta=secondary_meta)
