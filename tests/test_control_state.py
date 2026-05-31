import json
from pathlib import Path

from whose_agent.schemas import AgentId, ControlState, NextAction, Principal, StepKind, StepTrace


ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_dependency_declared() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    assert "langgraph" in pyproject


def test_control_state_defaults_are_deterministic() -> None:
    state = ControlState(
        principal="user",
        agent="assistant",
        principal_instruction="Fix the parser bug.",
        principal_signal="Narrow bug fix request.",
        step_kind="plan",
        step_index=0,
    )
    other_state = ControlState(
        principal="user",
        agent="assistant",
        principal_instruction="Fix the parser bug.",
        principal_signal="Narrow bug fix request.",
        step_kind="plan",
        step_index=1,
    )

    assert Principal is str
    assert AgentId is str
    assert state.next_action is None
    assert state.handoff_ready is False
    assert state.selected_skill_id is None
    assert state.selected_skill_perspective is None
    assert state.framework_specified is False
    assert state.guarantee_bypass_observed is False
    assert state.guarantee_bypass_evidence == []
    assert state.checker_id is None
    assert state.checker_confidence is None
    assert state.boundary_flags == []
    assert state.guarantee_bypass_evidence is not other_state.guarantee_bypass_evidence
    assert state.boundary_flags is not other_state.boundary_flags


def test_control_state_explicit_construction_and_json_serialization() -> None:
    step_kind: StepKind = "check"
    next_action: NextAction = "handoff"
    state = ControlState(
        principal="customer",
        agent="coding-agent",
        principal_instruction="Keep the TypeScript schema strict.",
        principal_signal="Do not weaken the guarantee.",
        step_kind=step_kind,
        step_index=3,
        next_action=next_action,
        handoff_ready=True,
        selected_skill_id="safety_framework_escape_hatch",
        selected_skill_perspective="Watch for a surface framework with a bypassed guarantee.",
        framework_specified=True,
        guarantee_bypass_observed=True,
        guarantee_bypass_evidence=["Uses any where the schema guarantee was requested."],
        checker_id="skill-perspective-checker",
        checker_confidence="high",
        boundary_flags=["constraint_override"],
    )

    dumped = json.loads(state.model_dump_json())

    assert dumped == {
        "principal": "customer",
        "agent": "coding-agent",
        "principal_instruction": "Keep the TypeScript schema strict.",
        "principal_signal": "Do not weaken the guarantee.",
        "step_kind": "check",
        "step_index": 3,
        "next_action": "handoff",
        "handoff_ready": True,
        "selected_skill_id": "safety_framework_escape_hatch",
        "selected_skill_perspective": (
            "Watch for a surface framework with a bypassed guarantee."
        ),
        "framework_specified": True,
        "guarantee_bypass_observed": True,
        "guarantee_bypass_evidence": [
            "Uses any where the schema guarantee was requested.",
        ],
        "checker_id": "skill-perspective-checker",
        "checker_confidence": "high",
        "boundary_flags": ["constraint_override"],
    }


def test_step_trace_defaults_are_deterministic() -> None:
    trace = StepTrace(
        step_index=0,
        step_kind="do",
        principal="user",
        agent="assistant",
    )
    other_trace = StepTrace(
        step_index=1,
        step_kind="do",
        principal="user",
        agent="assistant",
    )

    assert trace.misreader_skill_fired is False
    assert trace.selected_skill_id is None
    assert trace.checker_ran is False
    assert trace.checker_observed_bypass is False
    assert trace.trigger_evidence == []
    assert trace.substituted is None
    assert trace.boundary_flags == []
    assert trace.divergence_point is None
    assert trace.trigger_evidence is not other_trace.trigger_evidence
    assert trace.boundary_flags is not other_trace.boundary_flags


def test_step_trace_explicit_construction_and_json_serialization() -> None:
    trace = StepTrace(
        step_index=2,
        step_kind="check",
        principal="customer",
        agent="coding-agent",
        misreader_skill_fired=False,
        selected_skill_id="safety_framework_escape_hatch",
        checker_ran=True,
        checker_observed_bypass=True,
        trigger_evidence=["The response keeps TypeScript but replaces the guard with any."],
        substituted="instruction",
        boundary_flags=["constraint_override"],
        divergence_point="The guarantee is bypassed while the named framework remains.",
    )

    dumped = json.loads(trace.model_dump_json())

    assert dumped == {
        "step_index": 2,
        "step_kind": "check",
        "principal": "customer",
        "agent": "coding-agent",
        "misreader_skill_fired": False,
        "selected_skill_id": "safety_framework_escape_hatch",
        "checker_ran": True,
        "checker_observed_bypass": True,
        "trigger_evidence": [
            "The response keeps TypeScript but replaces the guard with any.",
        ],
        "substituted": "instruction",
        "boundary_flags": ["constraint_override"],
        "divergence_point": "The guarantee is bypassed while the named framework remains.",
    }
