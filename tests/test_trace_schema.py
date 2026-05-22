from pathlib import Path

import pytest

from whose_agent.bad_response import mock_bad_response
from whose_agent.classifier import classify_scenario
from whose_agent.scenario_loader import load_scenarios
from whose_agent.trace_emitter import TraceNotApplicableError, emit_trace


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRACE_FIELDS = {
    "scenario_id",
    "substituted",
    "failure_mode",
    "principal_signal",
    "bad_response",
    "divergence_point",
    "why_it_breaks_delegation",
    "better_behavior",
}
TRACE_SUBSTITUTED_VALUES = {"instruction", "authority", "role", "model"}


def test_trace_json_is_emitted_only_for_in_scope_scenarios() -> None:
    traces = []
    skipped = []

    for scenario in load_scenarios(ROOT / "scenarios"):
        classification = classify_scenario(scenario)
        if classification.classification == "out_of_scope":
            with pytest.raises(TraceNotApplicableError):
                emit_trace(scenario, classification, "No trace should be emitted.")
            skipped.append(scenario.scenario_id)
            continue

        traces.append(emit_trace(scenario, classification, mock_bad_response(classification)))

    assert len(traces) == 4
    assert len(skipped) == 2


def test_trace_json_contains_required_fields_and_no_none_substituted() -> None:
    for scenario in load_scenarios(ROOT / "scenarios"):
        classification = classify_scenario(scenario)
        if classification.classification == "out_of_scope":
            assert classification.substituted == "none"
            continue

        trace = emit_trace(scenario, classification, mock_bad_response(classification))
        dumped = trace.model_dump()

        assert set(dumped) == REQUIRED_TRACE_FIELDS
        assert trace.substituted in TRACE_SUBSTITUTED_VALUES
        assert trace.substituted != "none"
