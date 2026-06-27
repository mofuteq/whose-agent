"""Run fixed scenario workflows on a LangGraph StateGraph.

Schema definitions live in whose_agent.schemas; this module owns graph execution
only. Skill triggering and checker comparison are recorded in LangGraph state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from whose_agent.action_attempts import extract_external_persistence_attempt
from whose_agent.authority_provenance import (
    authority_trigger_evidence,
    derive_authority_checker_context,
    derive_external_persistence_provenance,
    evaluate_external_persistence_attempt,
    is_self_originated_delegation_laundering,
)
from whose_agent.bad_response import generate_bad_response_with_usage
from whose_agent.checker import (
    check_with_usage,
    compare_checker_observation,
    load_skill_perspective,
)
from whose_agent.classifier import classify_scenario
from whose_agent.conversation_view import project_messages
from whose_agent.history_adapter import append_assistant_message, initial_conversation_messages
from whose_agent.llm_result import LLMCallResult
from whose_agent.schemas import (
    AuthorityCauseRecord,
    AuthorityProvenance,
    BoundaryStateTrace,
    CheckerObservation,
    Classification,
    NextAction,
    Scenario,
    StepKind,
    StepTrace,
    Trace,
    WhoseAgentState,
)
from whose_agent.state_trace_renderer import render_boundary_state_trace
from whose_agent.trace_emitter import emit_trace_with_usage
from whose_agent.tracing import NoopTracer


CHECKER_ID = "skill-perspective-checker"


def initial_state_from_scenario(scenario: Scenario) -> WhoseAgentState:
    messages = initial_conversation_messages(
        scenario.initial_messages,
        prompt=scenario.principal_prompt,
    )
    authority_provenance = derive_external_persistence_provenance(
        project_messages(messages)
    )
    runtime_scenario = scenario.model_copy(update={"initial_messages": []})
    return {
        "principal": "user",
        "agent": "assistant",
        "principal_instruction": scenario.principal_prompt,
        "principal_signal": scenario.principal_signal,
        "messages": messages,
        "scenario": runtime_scenario,
        "classification": None,
        "bad_response": None,
        "generation_used_skill": False,
        "generation_skill_id": None,
        "trace": None,
        "state_trace": None,
        "checker_observation": None,
        "checker_comparison": None,
        "step_kind": "plan",
        "step_index": 0,
        "next_action": "continue",
        "handoff_ready": False,
        "completed": False,
        "selected_skill_id": scenario.selected_skill_id,
        "selected_skill_perspective": None,
        "skill_triggered": False,
        "misreader_skill_fired": False,
        "trigger_evidence": [],
        "checker_ran": False,
        "checker_observed_bypass": False,
        "checker_id": None,
        "checker_confidence": None,
        "checker_matches_expected": None,
        "observation_outcome": None,
        "guarantee_bypass_observed": False,
        "guarantee_bypass_evidence": [],
        "substituted": scenario.expected_substituted,
        "failure_mode": scenario.failure_mode,
        "divergence_point": None,
        "boundary_flags": [],
        "authority_provenance": authority_provenance,
        "authority_cause_record": None,
        "step_traces": [],
        "errors": [],
    }


def write_model_json(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sanitized_prompt_input(prompt: str) -> dict[str, Any]:
    return {
        "principal_prompt_length": len(prompt),
        "principal_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def sanitized_scenario_input(scenario: Scenario, **extra: Any) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "failure_mode": scenario.failure_mode,
        "expected_substituted": scenario.expected_substituted,
        **sanitized_prompt_input(scenario.principal_prompt),
        **extra,
    }


def update_span_with_llm_call(
    span: Any,
    *,
    output: dict[str, Any],
    llm_call: LLMCallResult[Any] | None = None,
) -> None:
    update_kwargs: dict[str, Any] = {"output": output}
    if llm_call is not None:
        if llm_call.usage_details:
            output["llm_usage"] = llm_call.usage_details
            update_kwargs["usage_details"] = llm_call.usage_details
        if llm_call.model_name:
            update_kwargs["model"] = llm_call.model_name
        if llm_call.model_settings:
            update_kwargs["model_parameters"] = llm_call.model_settings
    span.update(**update_kwargs)


def compile_fixed_scenario_graph(
    *,
    run_dir: Path | None = None,
    tracer: Any | None = None,
    mock: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    return build_fixed_scenario_graph(
        run_dir=run_dir,
        tracer=tracer,
        mock=mock,
    ).compile(checkpointer=checkpointer)


def build_fixed_scenario_graph(
    *,
    run_dir: Path | None = None,
    tracer: Any | None = None,
    mock: bool = False,
) -> StateGraph:
    tracer = tracer if tracer is not None else NoopTracer()
    graph = StateGraph(WhoseAgentState)

    def load_scenario(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        return {
            "principal": state.get("principal", "user"),
            "agent": state.get("agent", "assistant"),
            "principal_instruction": scenario.principal_prompt,
            "principal_signal": scenario.principal_signal,
            "selected_skill_id": scenario.selected_skill_id,
            "selected_skill_perspective": None,
            "substituted": scenario.expected_substituted,
            "failure_mode": scenario.failure_mode,
            **_step_update(state, "plan"),
        }

    def classify(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        with tracer.span(
            name="classify_scenario",
            metadata={
                "scenario_id": scenario.scenario_id,
                "expected_substituted": scenario.expected_substituted,
            },
            input=sanitized_scenario_input(scenario),
        ) as span:
            classification = classify_scenario(scenario)
            span.update(
                output={
                    "classification": classification.classification,
                    "substituted": classification.substituted,
                }
            )

        next_action: NextAction = (
            "stop" if classification.classification == "out_of_scope" else "continue"
        )
        return {
            "classification": classification,
            "substituted": classification.substituted,
            "next_action": next_action,
            **_step_update(state, "plan"),
        }

    def trigger_skill(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        selected_skill_id = scenario.selected_skill_id
        if selected_skill_id is None:
            return {
                "selected_skill_id": None,
                "selected_skill_perspective": None,
                "skill_triggered": False,
                "misreader_skill_fired": False,
                **_step_update(state, "plan", selected_skill_id=None),
            }

        if _uses_authority_provenance(state):
            return {
                "selected_skill_id": selected_skill_id,
                "selected_skill_perspective": None,
                "skill_triggered": False,
                "misreader_skill_fired": False,
                **_step_update(
                    state,
                    "plan",
                    selected_skill_id=selected_skill_id,
                    misreader_skill_fired=False,
                    authority_provenance=state.get("authority_provenance"),
                ),
            }

        selected_skill_perspective = load_skill_perspective(selected_skill_id)
        trigger_evidence = [
            "Scenario selected skill "
            f"{selected_skill_id!r} after in-scope classification; "
            "the deterministic fixed scenario treats the misreader skill as fired."
        ]
        return {
            "selected_skill_id": selected_skill_id,
            "selected_skill_perspective": selected_skill_perspective,
            "skill_triggered": True,
            "misreader_skill_fired": True,
            "trigger_evidence": trigger_evidence,
            **_step_update(
                state,
                "plan",
                selected_skill_id=selected_skill_id,
                misreader_skill_fired=True,
                trigger_evidence=trigger_evidence,
            ),
        }

    def generate_bad_response(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = _classification(state)
        selected_skill_id = state.get("selected_skill_id")
        selected_skill_perspective = state.get("selected_skill_perspective")
        misreader_skill_fired = bool(state.get("misreader_skill_fired", False))
        generation_used_skill = (
            misreader_skill_fired
            and selected_skill_id is not None
            and selected_skill_perspective is not None
        )
        bad_response_observation = tracer.span if mock else tracer.generation
        with bad_response_observation(
            name="generate_bad_response",
            metadata={
                "scenario_id": scenario.scenario_id,
                "substituted": classification.substituted,
                "mock": mock,
                "generation_used_skill": generation_used_skill,
            },
            input=sanitized_scenario_input(
                scenario,
                classification=classification.classification,
                substituted=classification.substituted,
                selected_skill_id=selected_skill_id,
                misreader_skill_fired=misreader_skill_fired,
                generation_used_skill=generation_used_skill,
                mock=mock,
            ),
        ) as span:
            bad_response_call = generate_bad_response_with_usage(
                scenario,
                classification,
                selected_skill_id=selected_skill_id,
                selected_skill_perspective=selected_skill_perspective,
                misreader_skill_fired=misreader_skill_fired,
                mock=mock,
            )
            bad_response = bad_response_call.output
            updated_messages = append_assistant_message(
                state.get("messages", []),
                bad_response,
            )
            authority_update: dict[str, Any] = {}
            authority_step_trigger_evidence: list[str] | None = None
            authority_step_fired: bool | None = None
            if _uses_authority_provenance(state):
                base_authority_provenance = (
                    derive_external_persistence_provenance(
                        project_messages(state.get("messages", []))
                    )
                )
                action_attempt = extract_external_persistence_attempt(
                    bad_response,
                    mock=mock,
                )
                action_attempt_turn = len(updated_messages) if action_attempt is not None else None
                authority_provenance = evaluate_external_persistence_attempt(
                    base_authority_provenance,
                    action_attempt,
                    action_attempt_turn=action_attempt_turn,
                )
                authority_step_fired = is_self_originated_delegation_laundering(
                    authority_provenance
                )
                authority_step_trigger_evidence = authority_trigger_evidence(
                    authority_provenance
                )
                authority_cause_record = AuthorityCauseRecord(
                    provenance=authority_provenance,
                    action_attempt=action_attempt,
                    drift_fired=authority_step_fired,
                    trigger_evidence=tuple(authority_step_trigger_evidence),
                )
                selected_skill_perspective = (
                    load_skill_perspective(cast(str, selected_skill_id))
                    if authority_step_fired and selected_skill_id is not None
                    else selected_skill_perspective
                )
                authority_update = {
                    "authority_provenance": authority_provenance,
                    "selected_skill_perspective": selected_skill_perspective,
                    "skill_triggered": authority_step_fired,
                    "misreader_skill_fired": authority_step_fired,
                    "trigger_evidence": authority_step_trigger_evidence,
                    "authority_cause_record": authority_cause_record,
                }
            update_span_with_llm_call(
                span,
                output={
                    "bad_response_length": len(bad_response),
                    "generation_used_skill": generation_used_skill,
                    "generation_skill_id": (
                        selected_skill_id if generation_used_skill else None
                    ),
                },
                llm_call=bad_response_call,
            )

        return {
            "bad_response": bad_response,
            "messages": updated_messages,
            "generation_used_skill": generation_used_skill,
            "generation_skill_id": selected_skill_id if generation_used_skill else None,
            "next_action": "continue",
            **authority_update,
            **_step_update(
                state,
                "do",
                selected_skill_id=selected_skill_id,
                misreader_skill_fired=authority_step_fired,
                trigger_evidence=authority_step_trigger_evidence,
                authority_provenance=(
                    authority_update.get("authority_provenance")
                    if authority_update
                    else None
                ),
            ),
        }

    def analyze_trace(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = _classification(state)
        bad_response = _bad_response(state)
        emit_trace_observation = tracer.span if mock else tracer.generation
        with emit_trace_observation(
            name="emit_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": mock},
            input=sanitized_scenario_input(
                scenario,
                substituted=classification.substituted,
                bad_response_length=len(bad_response),
                mock=mock,
            ),
        ) as span:
            trace_result = emit_trace_with_usage(
                scenario,
                classification,
                bad_response,
                mock=mock,
            )
            trace = trace_result.trace.model_copy(
                update={"authority_provenance": state.get("authority_provenance")}
            )
            update_span_with_llm_call(
                span,
                output={
                    "substituted": trace.substituted,
                    "failure_mode": trace.failure_mode,
                    "reflection_substituted": trace.reflection_substituted,
                },
                llm_call=trace_result.reflection_call,
            )

        return {
            "trace": trace,
            "substituted": trace.substituted,
            "failure_mode": trace.failure_mode,
            "divergence_point": trace.divergence_point,
            "next_action": "continue",
            **_step_update(
                state,
                "do",
                substituted=trace.substituted,
                divergence_point=trace.divergence_point,
            ),
        }

    def render_state_trace(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = _classification(state)
        bad_response = _bad_response(state)
        with tracer.span(
            name="render_state_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": mock},
            input=sanitized_scenario_input(
                scenario,
                substituted=classification.substituted,
                bad_response_length=len(bad_response),
                mock=mock,
            ),
        ) as span:
            state_trace = render_boundary_state_trace(state)
            final = state_trace.transitions[-1].state if state_trace.transitions else None
            output: dict[str, Any] = {}
            boundary_flags: list[str] = []
            next_action: NextAction = "continue"
            if final is not None:
                boundary_flags = [str(flag) for flag in final.boundary_flags]
                next_action = "continue" if final.next_action == "trace_ready" else "stop"
                output = {
                    "reflection_matches_expected": final.reflection_matches_expected,
                    "boundary_flags": boundary_flags,
                    "boundary_next_action": final.next_action,
                    "transition_count": len(state_trace.transitions),
                }
            span.update(output=output)

        return {
            "state_trace": state_trace,
            "boundary_flags": boundary_flags,
            "next_action": next_action,
            **_step_update(
                state,
                "check",
                substituted=_trace_substituted(state),
                boundary_flags=boundary_flags,
                divergence_point=state.get("divergence_point"),
            ),
        }

    def maybe_check(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        if scenario.selected_skill_id is None:
            return {
                **_step_update(
                    state,
                    "check",
                    substituted=_trace_substituted(state),
                    boundary_flags=state.get("boundary_flags", []),
                    divergence_point=state.get("divergence_point"),
                )
            }

        bad_response = _bad_response(state)
        authority_context = None
        if scenario.selected_skill_id == "authority_scope_expansion":
            history = project_messages(state.get("messages", []))
            generated_turn = history[-1].turn_index if history else None
            bounded_history = tuple(
                message
                for message in history
                if generated_turn is None or message.turn_index < generated_turn
            )
            checker_provenance = derive_external_persistence_provenance(
                bounded_history
            )
            if (
                checker_provenance is not None
                and checker_provenance.prior_agent_proposal_turn is not None
            ):
                action_attempt = extract_external_persistence_attempt(
                    bad_response,
                    mock=mock,
                )
                action_attempt_turn = (
                    generated_turn if action_attempt is not None else None
                )
                authority_context = derive_authority_checker_context(
                    history,
                    action_attempt,
                    action_attempt_turn=action_attempt_turn,
                )
        check_observation = tracer.span if mock else tracer.generation
        with check_observation(
            name="check_artifact",
            metadata={
                "scenario_id": scenario.scenario_id,
                "skill_id": scenario.selected_skill_id,
                "mock": mock,
            },
            input=sanitized_scenario_input(
                scenario,
                selected_skill_id=scenario.selected_skill_id,
                bad_response_length=len(bad_response),
                bad_response_sha256=hashlib.sha256(bad_response.encode()).hexdigest(),
                mock=mock,
            ),
        ) as span:
            checker_kwargs: dict[str, Any] = {"mock": mock}
            if authority_context is not None:
                checker_kwargs["authority_context"] = authority_context
            checker_result = check_with_usage(
                scenario,
                bad_response,
                **checker_kwargs,
            )
            checker_observation = checker_result.observation
            update_span_with_llm_call(
                span,
                output={
                    "checker_ran": True,
                    "skill_id": checker_observation.skill_id,
                    "checker_observed_bypass": checker_observation.checker_observed_bypass,
                    "confidence": checker_observation.confidence,
                    "evidence_count": len(checker_observation.evidence),
                    "substituted": checker_observation.substituted,
                    "failure_mode": checker_observation.failure_mode,
                },
                llm_call=checker_result.checker_call,
            )

        guarantee_evidence = (
            list(checker_observation.evidence)
            if checker_observation.checker_observed_bypass
            else []
        )
        return {
            "checker_observation": checker_observation,
            "checker_ran": True,
            "checker_observed_bypass": checker_observation.checker_observed_bypass,
            "checker_id": CHECKER_ID,
            "checker_confidence": checker_observation.confidence,
            "guarantee_bypass_observed": checker_observation.checker_observed_bypass,
            "guarantee_bypass_evidence": guarantee_evidence,
            **_step_update(
                state,
                "check",
                selected_skill_id=checker_observation.skill_id,
                checker_ran=True,
                checker_observed_bypass=checker_observation.checker_observed_bypass,
                substituted=_step_substituted(checker_observation.substituted),
                boundary_flags=state.get("boundary_flags", []),
                divergence_point=checker_observation.divergence_point
                or state.get("divergence_point"),
            ),
        }

    def compare_checker(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        checker_observation = state.get("checker_observation")
        comparison = compare_checker_observation(
            scenario,
            checker_observation,
            misreader_skill_fired=bool(state.get("misreader_skill_fired", False)),
        )
        with tracer.span(
            name="compare_checker",
            metadata={
                "scenario_id": scenario.scenario_id,
                "has_checker_template": scenario.checker_template is not None,
            },
            input={
                "scenario_id": scenario.scenario_id,
                "misreader_skill_fired": bool(
                    state.get("misreader_skill_fired", False)
                ),
                "checker_ran": bool(state.get("checker_ran", False)),
                "checker_observed_bypass": bool(
                    state.get("checker_observed_bypass", False)
                ),
            },
        ) as span:
            span.update(
                output={
                    "matches_expected": comparison.matches_expected,
                    "observation_outcome": comparison.observation_outcome,
                    "mismatch_count": len(comparison.mismatch_reasons),
                }
            )

        return {
            "checker_comparison": comparison,
            "checker_matches_expected": comparison.matches_expected,
            "observation_outcome": comparison.observation_outcome,
            **_step_update(
                state,
                "check",
                selected_skill_id=state.get("selected_skill_id"),
                checker_ran=bool(state.get("checker_ran", False)),
                checker_observed_bypass=bool(
                    state.get("checker_observed_bypass", False)
                ),
                substituted=(
                    checker_observation.substituted
                    if checker_observation is not None
                    else _trace_substituted(state)
                ),
                boundary_flags=state.get("boundary_flags", []),
                divergence_point=(
                    checker_observation.divergence_point
                    if checker_observation is not None
                    and checker_observation.divergence_point is not None
                    else state.get("divergence_point")
                ),
            ),
        }

    def write_artifacts(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = _classification(state)
        checker_observation = state.get("checker_observation")
        checker_comparison = state.get("checker_comparison")
        if classification.classification == "out_of_scope":
            artifact_names = [f"{scenario.scenario_id}.classification.json"]
            with tracer.span(
                name="write_artifacts",
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "artifact_names": artifact_names,
                },
                input={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
            ) as span:
                if run_dir is not None:
                    write_model_json(
                        run_dir / f"{scenario.scenario_id}.classification.json",
                        classification,
                    )
                span.update(output={"artifact_count": len(artifact_names)})
            return {
                "next_action": "stop",
                **_step_update(state, "do"),
            }

        bad_response = _bad_response(state)
        trace = _trace(state)
        state_trace = _state_trace(state)
        artifact_names = [
            f"{scenario.scenario_id}.classification.json",
            f"{scenario.scenario_id}.response.md",
            f"{scenario.scenario_id}.trace.json",
            f"{scenario.scenario_id}.state_trace.json",
        ]
        if checker_observation is not None:
            artifact_names.append(f"{scenario.scenario_id}.checker.json")
        if scenario.checker_template is not None and checker_comparison is not None:
            artifact_names.append(f"{scenario.scenario_id}.checker_comparison.json")
        with tracer.span(
            name="write_artifacts",
            metadata={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
            input={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
        ) as span:
            if run_dir is not None:
                write_model_json(
                    run_dir / f"{scenario.scenario_id}.classification.json", classification
                )
                write_text(run_dir / f"{scenario.scenario_id}.response.md", bad_response)
                write_model_json(run_dir / f"{scenario.scenario_id}.trace.json", trace)
                write_model_json(
                    run_dir / f"{scenario.scenario_id}.state_trace.json", state_trace
                )
                if checker_observation is not None:
                    write_model_json(
                        run_dir / f"{scenario.scenario_id}.checker.json",
                        checker_observation,
                    )
                if scenario.checker_template is not None and checker_comparison is not None:
                    write_model_json(
                        run_dir / f"{scenario.scenario_id}.checker_comparison.json",
                        checker_comparison,
                    )
            span.update(output={"artifact_count": len(artifact_names)})

        return {
            "next_action": "continue",
            **_step_update(
                state,
                "do",
                substituted=_trace_substituted(state),
                boundary_flags=state.get("boundary_flags", []),
                divergence_point=state.get("divergence_point"),
            ),
        }

    def finalize(state: WhoseAgentState) -> WhoseAgentState:
        return {
            "completed": True,
            "handoff_ready": False,
            "next_action": "stop",
            **_step_update(
                state,
                "check",
                substituted=_trace_substituted(state),
                selected_skill_id=state.get("selected_skill_id"),
                checker_ran=bool(state.get("checker_ran", False)),
                checker_observed_bypass=bool(state.get("checker_observed_bypass", False)),
                boundary_flags=state.get("boundary_flags", []),
                divergence_point=state.get("divergence_point"),
            ),
        }

    graph.add_node("load_scenario", load_scenario)
    graph.add_node("classify", classify)
    graph.add_node("trigger_skill", trigger_skill)
    graph.add_node("generate_bad_response", generate_bad_response)
    graph.add_node("analyze_trace", analyze_trace)
    graph.add_node("render_state_trace", render_state_trace)
    graph.add_node("maybe_check", maybe_check)
    graph.add_node("compare_checker", compare_checker)
    graph.add_node("write_artifacts", write_artifacts)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_scenario")
    graph.add_edge("load_scenario", "classify")
    graph.add_conditional_edges(
        "classify",
        _route_after_classification,
        {
            "in_scope": "trigger_skill",
            "out_of_scope": "write_artifacts",
        },
    )
    graph.add_edge("trigger_skill", "generate_bad_response")
    graph.add_edge("generate_bad_response", "analyze_trace")
    graph.add_edge("analyze_trace", "render_state_trace")
    graph.add_edge("render_state_trace", "maybe_check")
    graph.add_edge("maybe_check", "compare_checker")
    graph.add_edge("compare_checker", "write_artifacts")
    graph.add_edge("write_artifacts", "finalize")
    graph.add_edge("finalize", END)
    return graph


def _route_after_classification(state: WhoseAgentState) -> str:
    classification = _classification(state)
    return classification.classification


def _uses_authority_provenance(state: WhoseAgentState) -> bool:
    return (
        state.get("authority_provenance") is not None
        and state.get("selected_skill_id") == "authority_scope_expansion"
    )


def _step_update(
    state: WhoseAgentState,
    step_kind: StepKind,
    *,
    selected_skill_id: str | None = None,
    misreader_skill_fired: bool | None = None,
    trigger_evidence: list[str] | None = None,
    checker_ran: bool = False,
    checker_observed_bypass: bool = False,
    authority_provenance: AuthorityProvenance | None = None,
    substituted: str | None = None,
    boundary_flags: list[str] | None = None,
    divergence_point: str | None = None,
) -> WhoseAgentState:
    step_index = int(state.get("step_index", 0))
    step_misreader_skill_fired = (
        bool(state.get("misreader_skill_fired", False))
        if misreader_skill_fired is None
        else misreader_skill_fired
    )
    step_trigger_evidence = list(state.get("trigger_evidence", []))
    if trigger_evidence is not None:
        step_trigger_evidence.extend(trigger_evidence)
    step_selected_skill_id = (
        state.get("selected_skill_id") if selected_skill_id is None else selected_skill_id
    )
    trace = StepTrace(
        step_index=step_index,
        step_kind=step_kind,
        principal=state.get("principal", "user"),
        agent=state.get("agent", "assistant"),
        misreader_skill_fired=step_misreader_skill_fired,
        selected_skill_id=step_selected_skill_id,
        checker_ran=checker_ran,
        checker_observed_bypass=checker_observed_bypass,
        trigger_evidence=step_trigger_evidence,
        authority_provenance=authority_provenance,
        substituted=_step_substituted(substituted),
        boundary_flags=list(boundary_flags or []),
        divergence_point=divergence_point,
    )
    return {
        "step_kind": step_kind,
        "step_index": step_index + 1,
        "step_traces": [trace],
    }


def _scenario(state: WhoseAgentState) -> Scenario:
    scenario = state.get("scenario")
    if scenario is None:
        raise ValueError("WhoseAgentState requires scenario.")
    return scenario


def _classification(state: WhoseAgentState) -> Classification:
    classification = state.get("classification")
    if classification is None:
        raise ValueError("WhoseAgentState requires classification.")
    return classification


def _bad_response(state: WhoseAgentState) -> str:
    bad_response = state.get("bad_response")
    if bad_response is None:
        raise ValueError("WhoseAgentState requires bad_response.")
    return bad_response


def _trace(state: WhoseAgentState) -> Trace:
    trace = state.get("trace")
    if trace is None:
        raise ValueError("WhoseAgentState requires trace.")
    return trace


def _state_trace(state: WhoseAgentState) -> BoundaryStateTrace:
    state_trace = state.get("state_trace")
    if state_trace is None:
        raise ValueError("WhoseAgentState requires state_trace.")
    return state_trace


def _trace_substituted(state: WhoseAgentState) -> str | None:
    trace = state.get("trace")
    if trace is not None:
        return trace.substituted
    return _step_substituted(state.get("substituted"))


def _step_substituted(value: str | None) -> str | None:
    if value in {"instruction", "authority", "role", "model"}:
        return cast(str, value)
    return None
