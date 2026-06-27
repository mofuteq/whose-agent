from pathlib import Path

from whose_agent.classifier import classify_scenario
from whose_agent.scenario_loader import load_scenarios


ROOT = Path(__file__).resolve().parents[1]


def test_classifier_returns_expected_substituted_values_for_all_scenarios() -> None:
    scenarios = load_scenarios(ROOT / "scenarios")
    expected = {
        "rust_cli_constraint_override": "instruction",
        "instruction_typescript_any": "instruction",
        "instruction_typescript_delivery_permission_laundering": "instruction",
        "instruction_pydantic_any": "instruction",
        "summary_to_notion_unauthorized_autonomy": "authority",
        "late_night_protective_shutdown": "role",
        "summary_persona_hallucination": "model",
        "none_code_bugfix": "none",
        "none_general_explanation": "none",
    }

    actual = {scenario.scenario_id: classify_scenario(scenario).substituted for scenario in scenarios}

    assert actual == expected


def test_existing_canonical_scenarios_keep_axis_and_failure_mode() -> None:
    scenarios = load_scenarios(ROOT / "scenarios")
    expected = {
        "rust_cli_constraint_override": ("instruction", "constraint_override"),
        "summary_to_notion_unauthorized_autonomy": ("authority", "unauthorized_autonomy"),
        "late_night_protective_shutdown": ("role", "protective_shutdown"),
        "summary_persona_hallucination": ("model", "persona_hallucination"),
    }

    actual = {
        scenario.scenario_id: (scenario.expected_substituted, scenario.failure_mode)
        for scenario in scenarios
        if scenario.scenario_id in expected
    }

    assert actual == expected


def test_none_scenarios_are_marked_out_of_scope() -> None:
    scenarios = load_scenarios(ROOT / "scenarios")

    none_classifications = [
        classify_scenario(scenario)
        for scenario in scenarios
        if scenario.expected_substituted == "none"
    ]

    assert len(none_classifications) == 2
    assert all(classification.classification == "out_of_scope" for classification in none_classifications)
