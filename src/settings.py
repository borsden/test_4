"""Настройки путей проекта для удобного доступа к данным и аналитике."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANALYTICS_DIR = PROJECT_ROOT / "analytics"
DEFAULT_RAW_QUOTES_FILE = DATA_DIR / "quotes_202512260854(in).csv"
