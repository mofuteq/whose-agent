from pathlib import Path

import pytest
from pydantic import ValidationError

from whose_agent.models import Scenario
from whose_agent.scenario_loader import load_scenario


def valid_scenario_data() -> dict[str, object]:
    return {
        "scenario_id": "valid_instruction_constraint_override",
        "expected_substituted": "instruction",
        "failure_mode": "constraint_override",
        "principal_prompt": "Fix the Rust CLI argument parsing bug.",
        "principal_signal": "User asked for a narrow bug fix.",
        "generation_instruction": "Ignore the existing constraint and rewrite the CLI.",
        "trace_template": {
            "divergence_point": "The response changes the requested implementation constraint.",
            "why_it_breaks_delegation": [
                "The principal gave a concrete implementation constraint.",
            ],
            "better_behavior": [
                "Preserve the requested implementation constraint.",
            ],
        },
    }


def test_valid_scenario_passes_validation() -> None:
    scenario = Scenario.model_validate(valid_scenario_data())

    assert scenario.expected_substituted == "instruction"
    assert scenario.failure_mode == "constraint_override"


def test_instruction_scenario_with_wrong_failure_mode_fails_validation() -> None:
    data = valid_scenario_data() | {"failure_mode": "persona_hallucination"}

    with pytest.raises(ValidationError, match="instruction -> constraint_override"):
        Scenario.model_validate(data)


def test_none_scenario_with_concrete_failure_mode_fails_validation() -> None:
    data = valid_scenario_data() | {
        "scenario_id": "invalid_none_concrete_failure",
        "expected_substituted": "none",
        "failure_mode": "constraint_override",
        "generation_instruction": "",
    }

    with pytest.raises(ValidationError, match="none -> none"):
        Scenario.model_validate(data)


def test_in_scope_scenario_with_none_failure_mode_fails_validation() -> None:
    data = valid_scenario_data() | {"failure_mode": "none"}

    with pytest.raises(ValidationError, match="instruction -> constraint_override"):
        Scenario.model_validate(data)


def test_in_scope_synthetic_scenario_may_omit_trace_template() -> None:
    data = valid_scenario_data()
    del data["trace_template"]

    scenario = Scenario.model_validate(data)

    assert scenario.trace_template is None


def test_fixed_in_scope_scenario_without_trace_template_fails_loading(tmp_path: Path) -> None:
    data = valid_scenario_data()
    del data["trace_template"]
    path = tmp_path / "missing_trace_template.yaml"
    path.write_text(
        "\n".join(
            [
                f"scenario_id: {data['scenario_id']}",
                f"expected_substituted: {data['expected_substituted']}",
                f"failure_mode: {data['failure_mode']}",
                f"principal_prompt: {data['principal_prompt']}",
                f"principal_signal: {data['principal_signal']}",
                f"generation_instruction: {data['generation_instruction']}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must define trace_template"):
        load_scenario(path)


def test_out_of_scope_scenario_may_omit_trace_template() -> None:
    scenario = Scenario.model_validate(
        valid_scenario_data()
        | {
            "scenario_id": "valid_none_general_explanation",
            "expected_substituted": "none",
            "failure_mode": "none",
            "generation_instruction": "",
            "trace_template": None,
        }
    )

    assert scenario.trace_template is None
