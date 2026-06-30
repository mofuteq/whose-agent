from pathlib import Path

import pytest
from pydantic import ValidationError

from whose_agent.schemas import FAILURE_MODES, Scenario
from whose_agent.scenario_loader import load_scenario, load_scenarios


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
    assert scenario.selected_skill_id is None
    assert scenario.checker_template is None


def test_models_module_is_removed() -> None:
    models_path = Path(__file__).resolve().parents[1] / "src" / "whose_agent" / "models.py"

    assert not models_path.exists()
    assert "constraint_override" in FAILURE_MODES


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


def test_safe_response_none_scenario_passes_validation() -> None:
    scenario = Scenario.model_validate(
        valid_scenario_data()
        | {
            "scenario_id": "valid_none_safe_response",
            "expected_substituted": "none",
            "failure_mode": "none",
            "generation_instruction": "",
            "trace_template": None,
            "safe_response": "\nSafe response.\n",
        }
    )

    assert scenario.safe_response == "Safe response."


def test_safe_response_requires_none_scenario() -> None:
    data = valid_scenario_data() | {"safe_response": "Safe response."}

    with pytest.raises(ValidationError, match="safe_response is allowed only"):
        Scenario.model_validate(data)


def test_safe_response_rejects_trace_template() -> None:
    data = valid_scenario_data() | {
        "scenario_id": "invalid_safe_trace_template",
        "expected_substituted": "none",
        "failure_mode": "none",
        "generation_instruction": "",
        "safe_response": "Safe response.",
    }

    with pytest.raises(
        ValidationError,
        match="safe_response scenarios must not define trace_template",
    ):
        Scenario.model_validate(data)


def test_safe_response_rejects_selected_skill_id() -> None:
    data = valid_scenario_data() | {
        "scenario_id": "invalid_safe_selected_skill",
        "expected_substituted": "none",
        "failure_mode": "none",
        "selected_skill_id": "safety_framework_escape_hatch",
        "generation_instruction": "",
        "trace_template": None,
        "safe_response": "Safe response.",
    }

    with pytest.raises(
        ValidationError,
        match="safe_response scenarios must not define selected_skill_id",
    ):
        Scenario.model_validate(data)


def test_selected_skill_id_requires_checker_template() -> None:
    data = valid_scenario_data() | {
        "selected_skill_id": "safety_framework_escape_hatch",
    }

    with pytest.raises(ValidationError, match="selected_skill_id requires checker_template"):
        Scenario.model_validate(data)


def test_checker_template_requires_selected_skill_id() -> None:
    data = valid_scenario_data() | {
        "checker_template": valid_checker_template_data(),
    }

    with pytest.raises(ValidationError, match="checker_template requires selected_skill_id"):
        Scenario.model_validate(data)


def test_none_scenario_must_not_have_checker_template() -> None:
    data = valid_scenario_data() | {
        "scenario_id": "invalid_none_checker_template",
        "expected_substituted": "none",
        "failure_mode": "none",
        "selected_skill_id": "safety_framework_escape_hatch",
        "generation_instruction": "",
        "checker_template": valid_checker_template_data(),
    }

    with pytest.raises(ValidationError, match="none scenarios must not define checker_template"):
        Scenario.model_validate(data)


def test_checker_template_failure_mode_must_match_substituted() -> None:
    data = valid_scenario_data() | {
        "selected_skill_id": "safety_framework_escape_hatch",
        "checker_template": valid_checker_template_data()
        | {"substituted": "instruction", "failure_mode": "persona_hallucination"},
    }

    with pytest.raises(
        ValidationError,
        match="checker_template.failure_mode must match checker_template.substituted",
    ):
        Scenario.model_validate(data)


def test_typescript_any_selects_skill_perspective() -> None:
    scenario = load_scenario(
        Path(__file__).resolve().parents[1] / "scenarios" / "instruction_typescript_any.yaml"
    )

    assert scenario.selected_skill_id == "safety_framework_escape_hatch"
    assert scenario.checker_template is not None
    assert scenario.checker_template.substituted == "instruction"
    assert scenario.checker_template.failure_mode == "constraint_override"


def test_delivery_permission_laundering_selects_skill_perspective() -> None:
    scenario = load_scenario(
        Path(__file__).resolve().parents[1]
        / "scenarios"
        / "instruction_typescript_delivery_permission_laundering.yaml"
    )

    assert scenario.scenario_id == "instruction_typescript_delivery_permission_laundering"
    assert scenario.expected_substituted == "instruction"
    assert scenario.failure_mode == "constraint_override"
    assert scenario.selected_skill_id == "safety_framework_escape_hatch"
    assert scenario.checker_template is not None
    assert scenario.checker_template.substituted == "instruction"
    assert scenario.checker_template.failure_mode == "constraint_override"
    assert scenario.trace_template is not None


def test_pydantic_any_selects_skill_perspective() -> None:
    scenario = load_scenario(
        Path(__file__).resolve().parents[1] / "scenarios" / "instruction_pydantic_any.yaml"
    )

    assert scenario.selected_skill_id == "safety_framework_escape_hatch"
    assert scenario.checker_template is not None
    assert scenario.checker_template.substituted == "instruction"
    assert scenario.checker_template.failure_mode == "constraint_override"
    assert scenario.trace_template is not None
    divergence_point = scenario.trace_template.divergence_point
    assert "Pydantic" in divergence_point
    assert "Rust" not in divergence_point
    assert "TypeScript" not in divergence_point


def test_existing_escape_hatch_scenarios_do_not_require_delivery_concession() -> None:
    root = Path(__file__).resolve().parents[1]
    typescript_any = load_scenario(root / "scenarios" / "instruction_typescript_any.yaml")
    pydantic_any = load_scenario(root / "scenarios" / "instruction_pydantic_any.yaml")
    rust_cli = load_scenario(root / "scenarios" / "instruction_rust_cli.yaml")

    assert typescript_any.selected_skill_id == "safety_framework_escape_hatch"
    assert pydantic_any.selected_skill_id == "safety_framework_escape_hatch"
    assert rust_cli.selected_skill_id == "instruction_constraint_override"
    assert "Avoid using `any`" in typescript_any.principal_prompt
    assert "Avoid `Any`" in pydantic_any.principal_prompt

    combined_escape_hatch_text = " ".join(
        [
            typescript_any.principal_prompt,
            typescript_any.principal_signal,
            pydantic_any.principal_prompt,
            pydantic_any.principal_signal,
        ]
    ).casefold()
    for concession_wording in ("prototype", "poc", "mvp", "e2e", "quick delivery"):
        assert concession_wording not in combined_escape_hatch_text


def test_bundled_fixed_scenarios_have_display_titles() -> None:
    root = Path(__file__).resolve().parents[1]
    scenarios = load_scenarios(root / "scenarios")

    assert scenarios
    assert all(scenario.display_title for scenario in scenarios)


def valid_checker_template_data() -> dict[str, object]:
    return {
        "checker_observed_bypass": True,
        "substituted": "instruction",
        "failure_mode": "constraint_override",
        "evidence": [
            "The response preserves the framework surface.",
            "The response bypasses the intended guarantee.",
        ],
        "divergence_point": "The guarantee is bypassed.",
        "confidence": "high",
    }
