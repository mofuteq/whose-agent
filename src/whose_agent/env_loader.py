from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


ENV_KEYS = {"OPENROUTER_API_KEY", "WHOSE_AGENT_MODEL"}


def load_env_file(path: Path = Path(".env"), *, override: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for key, value in dotenv_values(path).items():
        if key not in ENV_KEYS or value is None:
            continue
        if not override and key in os.environ:
            continue

        os.environ[key] = value
        loaded[key] = value

    return loaded
