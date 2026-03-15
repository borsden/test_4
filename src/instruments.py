"""Метаданные по инструментам: множители, тики и валюты."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentMeta:
    """Хранит экономику контракта для приведения котировок к одной базе."""

    symbol: str
    multiplier: float
    tick_size: float
    currency: str
    fx_to_rub: float = 1.0

    @property
    def tick_value(self) -> float:
        """Возвращает денежную стоимость одного тика в базовой валюте."""
        return self.tick_size * self.multiplier * self.fx_to_rub


DEFAULT_INSTRUMENTS = {
    "GLDG26": InstrumentMeta(
        symbol="GLDG26",
        multiplier=1.0,
        tick_size=0.1,
        currency="RUB",
    ),
    "GOLD-3.26": InstrumentMeta(
        symbol="GOLD-3.26",
        multiplier=1.0,
        tick_size=0.1,
        currency="RUB",
    ),
}
"""Базовая конфигурация, при необходимости заменить реальными параметрами."""
