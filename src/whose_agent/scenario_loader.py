from __future__ import annotations

from pathlib import Path

import yaml

from whose_agent.models import Scenario


def load_scenario(path: Path) -> Scenario:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file {path} must contain a YAML mapping.")
    return Scenario.model_validate(data)


def load_scenarios(directory: Path) -> list[Scenario]:
    paths = sorted(directory.glob("*.yaml"))
    return [load_scenario(path) for path in paths]
