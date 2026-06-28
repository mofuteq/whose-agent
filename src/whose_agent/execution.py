from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from whose_agent.authority_provenance import derive_external_persistence_provenance
from whose_agent.conversation_view import project_messages
from whose_agent.firing_signals import FiringSignals
from whose_agent.history_adapter import conversation_from_prompt, require_unique_message_ids
from whose_agent.loop_artifacts import write_loop_trace, write_prompt_loop_generated
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import compile_minimal_loop_graph
from whose_agent.prompt_contract_artifacts import write_prompt_contract
from whose_agent.prompt_contract_detector import detect_prompt_contract
from whose_agent.prompt_loop import (
    PROMPT_LOOP_SCENARIO_ID,
    _last_do_step_index,
    _should_emit_prompt_loop_generated,
    initial_loop_state_from_prompt_contract,
)
from whose_agent.public_projection import (
    CauseProjection,
    CheckerProjection,
    CompletedProjection,
    ExplainProjection,
    PhaseProjection,
    RunMode,
    ScenarioMetadata,
    project_cause,
    project_checker,
    project_completed,
    project_explain,
    project_scenario_metadata,
)
from whose_agent.run_directory import create_run_directory
from whose_agent.scenario_loader import load_scenario, load_scenarios
from whose_agent.schemas import (
    ConversationMessage,
    PromptContract,
    Scenario,
    WhoseAgentState,
)
from whose_agent.self_explanation import write_self_explanation
from whose_agent.state_graph import compile_fixed_scenario_graph, initial_state_from_scenario
from whose_agent.tracing import NoopTracer


RunnerEventKind = Literal["phase", "cause", "checker", "explain", "text", "completed"]


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    mode: RunMode
    run_dir: Path
    artifact_names: list[str]
    completed: CompletedProjection


@dataclass(frozen=True)
class FixedBatchResult:
    run_dir: Path
    scenario_count: int
    artifact_names: list[str]
    results: list[ExecutionResult]

    @property
    def classification_count(self) -> int:
        return _count_suffix(self.artifact_names, ".classification.json")

    @property
    def response_count(self) -> int:
        return _count_suffix(self.artifact_names, ".response.md")

    @property
    def trace_count(self) -> int:
        return len(
            [
                name
                for name in self.artifact_names
                if name.endswith(".trace.json")
                and not name.endswith(".state_trace.json")
            ]
        )

    @property
    def state_trace_count(self) -> int:
        return _count_suffix(self.artifact_names, ".state_trace.json")

    @property
    def checker_count(self) -> int:
        return _count_suffix(self.artifact_names, ".checker.json")

    @property
    def checker_comparison_count(self) -> int:
        return _count_suffix(self.artifact_names, ".checker_comparison.json")

    @property
    def explanation_count(self) -> int:
        return _count_suffix(self.artifact_names, ".explanation.json")


@dataclass(frozen=True)
class RunnerEvent:
    kind: RunnerEventKind
    phase: PhaseProjection | None = None
    cause: CauseProjection | None = None
    checker: CheckerProjection | None = None
    explain: ExplainProjection | None = None
    text: str | None = None
    result: ExecutionResult | None = None


def list_fixed_scenarios(scenarios_dir: Path) -> list[ScenarioMetadata]:
    return [
        project_scenario_metadata(scenario)
        for scenario in load_scenarios(scenarios_dir)
    ]


def load_known_scenario(scenarios_dir: Path, scenario_id: str) -> Scenario | None:
    for scenario in load_scenarios(scenarios_dir):
        if scenario.scenario_id == scenario_id:
            return scenario
    return None


async def stream_fixed_scenario(
    *,
    run_id: str,
    scenario: Scenario,
    outputs_dir: Path,
    mock: bool,
    run_dir: Path | None = None,
    tracer: object | None = None,
) -> AsyncIterator[RunnerEvent]:
    run_dir = run_dir if run_dir is not None else create_run_directory(outputs_dir)
    graph = compile_fixed_scenario_graph(
        run_dir=run_dir,
        tracer=tracer if tracer is not None else NoopTracer(),
        mock=mock,
    )
    async for event in _stream_fixed_scenario_graph(
        run_id=run_id,
        scenario=scenario,
        run_dir=run_dir,
        graph=graph,
    ):
        yield event


def run_fixed_scenarios(
    *,
    scenarios_dir: Path,
    outputs_dir: Path,
    mock: bool,
    run_dir: Path | None = None,
    tracer: object | None = None,
) -> FixedBatchResult:
    return asyncio.run(
        _run_fixed_scenarios_async(
            scenarios_dir=scenarios_dir,
            outputs_dir=outputs_dir,
            mock=mock,
            run_dir=run_dir,
            tracer=tracer,
        )
    )


async def stream_prompt_loop(
    *,
    run_id: str,
    prompt: str,
    outputs_dir: Path,
    mock: bool,
    max_iterations: int,
    messages: list[ConversationMessage] | None = None,
    run_dir: Path | None = None,
    firing_signals: FiringSignals | None = None,
    tracer: object | None = None,
) -> AsyncIterator[RunnerEvent]:
    run_dir = run_dir if run_dir is not None else create_run_directory(outputs_dir)
    canonical_messages = (
        list(messages) if messages is not None else conversation_from_prompt(prompt)
    )
    require_unique_message_ids(canonical_messages)
    authority_provenance = derive_external_persistence_provenance(
        project_messages(canonical_messages)
    )
    if authority_provenance is None:
        contract = detect_prompt_contract(prompt, mock=mock)
    else:
        contract = detect_prompt_contract(
            prompt,
            mock=mock,
            authority_provenance=authority_provenance,
        )
    contract_path = write_prompt_contract(contract, run_dir)

    graph = compile_minimal_loop_graph(
        mock=mock,
        tracer=tracer if tracer is not None else NoopTracer(),
    )
    initial_state = initial_loop_state_from_prompt_contract(
        contract,
        max_iterations=max_iterations,
        firing_signals=firing_signals,
        messages=canonical_messages,
    )
    initial_state["prompt_contract_artifact"] = contract_path.name

    async for event in _stream_prompt_loop_graph(
        run_id=run_id,
        contract=contract,
        run_dir=run_dir,
        graph=graph,
        initial_state=initial_state,
    ):
        yield event


def run_prompt_loop(
    *,
    run_id: str,
    prompt: str,
    outputs_dir: Path,
    mock: bool,
    max_iterations: int,
    messages: list[ConversationMessage] | None = None,
    run_dir: Path | None = None,
    firing_signals: FiringSignals | None = None,
    tracer: object | None = None,
) -> ExecutionResult:
    return asyncio.run(
        _collect_execution_result(
            stream_prompt_loop(
                run_id=run_id,
                prompt=prompt,
                outputs_dir=outputs_dir,
                mock=mock,
                max_iterations=max_iterations,
                messages=messages,
                run_dir=run_dir,
                firing_signals=firing_signals,
                tracer=tracer,
            )
        )
    )


async def _run_fixed_scenarios_async(
    *,
    scenarios_dir: Path,
    outputs_dir: Path,
    mock: bool,
    run_dir: Path | None,
    tracer: object | None,
) -> FixedBatchResult:
    scenarios = load_scenarios(scenarios_dir)
    run_dir = run_dir if run_dir is not None else create_run_directory(outputs_dir)
    graph = compile_fixed_scenario_graph(
        run_dir=run_dir,
        tracer=tracer if tracer is not None else NoopTracer(),
        mock=mock,
    )
    results: list[ExecutionResult] = []
    for index, scenario in enumerate(scenarios, start=1):
        result = await _collect_execution_result(
            _stream_fixed_scenario_graph(
                run_id=f"fixed_batch_{index}",
                scenario=scenario,
                run_dir=run_dir,
                graph=graph,
            )
        )
        results.append(result)
    return FixedBatchResult(
        run_dir=run_dir,
        scenario_count=len(scenarios),
        artifact_names=_artifact_names(run_dir),
        results=results,
    )


async def _stream_fixed_scenario_graph(
    *,
    run_id: str,
    scenario: Scenario,
    run_dir: Path,
    graph: object,
) -> AsyncIterator[RunnerEvent]:
    yield RunnerEvent(kind="phase", phase=PhaseProjection(phase="plan"))

    final_state: WhoseAgentState | None = None
    emitted_step_indexes: set[int] = set()
    checker_emitted = False
    async for stream_mode, payload in graph.astream(
        initial_state_from_scenario(scenario),
        stream_mode=["values"],
    ):
        if stream_mode != "values":
            continue
        state = payload
        final_state = state
        async for event in _project_state_progress(
            state,
            emitted_step_indexes=emitted_step_indexes,
            checker_emitted=checker_emitted,
        ):
            if event.kind == "checker":
                checker_emitted = True
            yield event

    if final_state is None:
        raise RuntimeError("fixed scenario graph did not produce a final state")
    artifact_names = _artifact_names(run_dir)
    result = ExecutionResult(
        run_id=run_id,
        mode="fixed",
        run_dir=run_dir,
        artifact_names=artifact_names,
        completed=project_completed(
            run_id=run_id,
            mode="fixed",
            state=final_state,
            artifact_names=artifact_names,
        ),
    )
    yield RunnerEvent(kind="completed", result=result)


async def _stream_prompt_loop_graph(
    *,
    run_id: str,
    contract: PromptContract,
    run_dir: Path,
    graph: object,
    initial_state: WhoseAgentState,
) -> AsyncIterator[RunnerEvent]:
    yield RunnerEvent(kind="phase", phase=PhaseProjection(phase="plan"))

    final_state: WhoseAgentState | None = None
    emitted_step_indexes: set[int] = set()
    checker_emitted = False
    async for stream_mode, payload in graph.astream(
        initial_state,
        stream_mode=["values"],
    ):
        if stream_mode != "values":
            continue
        state = payload
        final_state = state
        async for event in _project_state_progress(
            state,
            emitted_step_indexes=emitted_step_indexes,
            checker_emitted=checker_emitted,
        ):
            if event.kind == "checker":
                checker_emitted = True
            yield event

    if final_state is None:
        raise RuntimeError("prompt-loop graph did not produce a final state")
    _write_prompt_loop_artifacts(run_dir, contract, final_state)
    artifact_names = _artifact_names(run_dir)
    result = ExecutionResult(
        run_id=run_id,
        mode="prompt_loop",
        run_dir=run_dir,
        artifact_names=artifact_names,
        completed=project_completed(
            run_id=run_id,
            mode="prompt_loop",
            state=final_state,
            artifact_names=artifact_names,
        ),
    )
    yield RunnerEvent(kind="completed", result=result)


async def _project_state_progress(
    state: WhoseAgentState,
    *,
    emitted_step_indexes: set[int],
    checker_emitted: bool,
) -> AsyncIterator[RunnerEvent]:
    traces = state.get("step_traces", [])
    if not traces:
        return
    latest_trace = traces[-1]
    if latest_trace.step_index in emitted_step_indexes:
        return
    if latest_trace.step_kind == "plan":
        emitted_step_indexes.add(latest_trace.step_index)
        return
    if latest_trace.step_kind == "do":
        emitted_step_indexes.add(latest_trace.step_index)
        yield RunnerEvent(kind="phase", phase=PhaseProjection(phase="do"))
        bad_response = state.get("bad_response")
        if bad_response is not None:
            yield RunnerEvent(kind="text", text=bad_response)
        yield RunnerEvent(kind="cause", cause=project_cause(state))
        return
    if latest_trace.step_kind == "check":
        if state.get("observation_outcome") is None or checker_emitted:
            return
        emitted_step_indexes.add(latest_trace.step_index)
        yield RunnerEvent(kind="phase", phase=PhaseProjection(phase="check"))
        yield RunnerEvent(kind="checker", checker=project_checker(state))
        return
    if latest_trace.step_kind == "explain":
        explanation = state.get("self_explanation")
        if explanation is None:
            emitted_step_indexes.add(latest_trace.step_index)
            return
        emitted_step_indexes.add(latest_trace.step_index)
        yield RunnerEvent(kind="phase", phase=PhaseProjection(phase="explain"))
        yield RunnerEvent(kind="explain", explain=project_explain(explanation))


def _write_prompt_loop_artifacts(
    run_dir: Path,
    contract: PromptContract,
    state: WhoseAgentState,
) -> None:
    if _should_emit_prompt_loop_generated(contract):
        generated_output = state.get("bad_response")
        if generated_output is None:
            raise RuntimeError("supported prompt loop did not produce generated output")
        generated_step_index = _last_do_step_index(state)
        if generated_step_index is None:
            raise RuntimeError("supported prompt loop did not record a do step")
        generated_path = write_prompt_loop_generated(run_dir, generated_output)
        state["prompt_loop_generated_artifact"] = generated_path.name
        state["prompt_loop_generated_step_index"] = generated_step_index

    loop_trace = render_loop_trace(state)
    write_loop_trace(run_dir, loop_trace)
    self_explanation = state.get("self_explanation")
    if self_explanation is not None:
        write_self_explanation(run_dir, PROMPT_LOOP_SCENARIO_ID, self_explanation)


async def _collect_execution_result(
    events: AsyncIterator[RunnerEvent],
) -> ExecutionResult:
    async for event in events:
        if event.kind == "completed" and event.result is not None:
            return event.result
    raise RuntimeError("execution did not produce a completed result")


def _artifact_names(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(path.name for path in run_dir.iterdir() if path.is_file())


def _count_suffix(names: Sequence[str], suffix: str) -> int:
    return len([name for name in names if name.endswith(suffix)])
