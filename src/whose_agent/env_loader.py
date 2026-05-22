from __future__ import annotations

import os
from pathlib import Path


ENV_KEYS = {"OPENROUTER_API_KEY", "WHOSE_AGENT_MODEL"}


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, character in enumerate(value):
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def _parse_env_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_env_file(path: Path = Path(".env"), *, override: bool = False) -> dict[str, str]:
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").lstrip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in ENV_KEYS:
            continue
        if not override and key in os.environ:
            continue

        value = _parse_env_value(raw_value)
        os.environ[key] = value
        loaded[key] = value

    return loaded
