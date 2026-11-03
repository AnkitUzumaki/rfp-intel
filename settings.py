"""Config loader: config/settings.yaml (non-secret) + environment / .env (secrets, model names)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@lru_cache
def settings() -> dict[str, Any]:
    path = Path(os.getenv("RFP_SETTINGS", ROOT / "config" / "settings.yaml"))
    return yaml.safe_load(path.read_text())


def cfg(section: str, key: str) -> Any:
    return settings()[section][key]
