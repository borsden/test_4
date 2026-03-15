"""Конфигурация пар инструментов для кросс-рыночного анализа."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairConfig:
    """Описывает арбитражируемую пару инструментов."""

    name: str
    primary_symbol: str
    secondary_symbol: str
    base_currency: str = "RUB"
    hedge_ratio: float = 1.0


DEFAULT_PAIR = PairConfig(
    name="gold_cross",
    primary_symbol="GLDG26",
    secondary_symbol="GOLD-3.26",
    base_currency="RUB",
    hedge_ratio=1.0,
)
