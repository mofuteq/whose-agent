"""Artifact writer for loop trace artifacts.

Writes <scenario_id>.loop_trace.json from a LoopTrace.
Does not wire into the existing fixed scenario run command.
"""

from __future__ import annotations

import json
from pathlib import Path

from whose_agent.schemas import LoopTrace, Scenario


def write_loop_trace(output_dir: Path, loop_trace: LoopTrace) -> Path:
    """Write a LoopTrace to output_dir/<scenario_id>.loop_trace.json."""
    path = output_dir / f"{loop_trace.scenario_id}.loop_trace.json"
    path.write_text(
        json.dumps(loop_trace.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def run_minimal_loop_to_artifact(
    scenario: Scenario,
    output_dir: Path,
    *,
    max_iterations: int = 1,
    mock: bool = True,
) -> LoopTrace:
    """Run the minimal loop, render a LoopTrace, and write the artifact.

    Test-only / programmatic helper. Not wired into CLI commands.
    """
    from whose_agent.loop_trace_renderer import render_loop_trace
    from whose_agent.minimal_loop_graph import (
        compile_minimal_loop_graph,
        initial_loop_state_from_scenario,
    )

    graph = compile_minimal_loop_graph(mock=mock)
    state = graph.invoke(
        initial_loop_state_from_scenario(scenario, max_iterations=max_iterations)
    )
    loop_trace = render_loop_trace(state)
    write_loop_trace(output_dir, loop_trace)
    return loop_trace
