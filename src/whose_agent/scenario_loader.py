from __future__ import annotations

from pathlib import Path

import yaml

from whose_agent.schemas import Scenario


def load_scenario(path: Path) -> Scenario:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file {path} must contain a YAML mapping.")
    scenario = Scenario.model_validate(data)
    if scenario.expected_substituted != "none" and scenario.trace_template is None:
        raise ValueError(f"Fixed scenario {path} must define trace_template.")
    return scenario


def load_scenarios(directory: Path) -> list[Scenario]:
    paths = sorted(directory.glob("*.yaml"))
    return [load_scenario(path) for path in paths]
