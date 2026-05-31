from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from whose_agent.bad_response import BadResponseError
from whose_agent.checker import CheckerError
from whose_agent.env_loader import load_env_file
from whose_agent.flow_emitter import emit_prompt_flow
from whose_agent.llm_classifier import PromptClassifierError, classify_prompt_with_usage
from whose_agent.llm_result import LLMCallResult
from whose_agent.loop_artifacts import run_minimal_loop_to_artifact
from whose_agent.prompt_contract_artifacts import write_prompt_contract
from whose_agent.prompt_contract_detector import (
    PromptContractDetectorError,
    detect_prompt_contract,
)
from whose_agent.prompt_loop import run_prompt_loop_to_artifact
from whose_agent.schemas import Scenario
from whose_agent.prompt_run import build_prompt_run, mock_classify_prompt
from whose_agent.reflection import ReflectionError
from whose_agent.run_directory import create_run_directory
from whose_agent.scenario_loader import load_scenario, load_scenarios
from whose_agent.state_graph import compile_fixed_scenario_graph, initial_state_from_scenario
from whose_agent.tracing import create_observability_tracer


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


def run_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))
    tracer = create_observability_tracer()

    scenarios_dir = Path(args.scenarios)
    run_dir = create_run_directory(Path(args.outputs))

    scenarios = load_scenarios(scenarios_dir)
    session_id = run_dir.name
    tracer.start_run(
        name="run",
        metadata={
            "command": "run",
            "mock": args.mock,
            "scenario_count": len(scenarios),
            "run_dir": run_dir.name,
        },
        session_id=session_id,
    )

    graph = compile_fixed_scenario_graph(run_dir=run_dir, tracer=tracer, mock=args.mock)
    final_states = []
    for scenario in scenarios:
        final_states.append(graph.invoke(initial_state_from_scenario(scenario)))

    classification_count = sum(1 for state in final_states if state.get("classification") is not None)
    response_count = sum(1 for state in final_states if state.get("bad_response") is not None)
    trace_count = sum(1 for state in final_states if state.get("trace") is not None)
    state_trace_count = sum(1 for state in final_states if state.get("state_trace") is not None)
    checker_count = sum(
        1 for state in final_states if state.get("checker_observation") is not None
    )
    checker_comparison_count = sum(
        1
        for state in final_states
        if state.get("checker_comparison") is not None
        and state["scenario"].checker_template is not None
    )

    checker_file_label = "checker file" if checker_count == 1 else "checker files"
    checker_comparison_file_label = (
        "checker comparison file"
        if checker_comparison_count == 1
        else "checker comparison files"
    )
    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        f"{classification_count} classification files, "
        f"{response_count} response files, "
        f"{trace_count} trace files, "
        f"{state_trace_count} state trace files, "
        f"{checker_count} {checker_file_label}, and "
        f"{checker_comparison_count} {checker_comparison_file_label}."
    )
    tracer.flush()
    return 0


def run_prompt_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))
    tracer = create_observability_tracer()

    run_dir = create_run_directory(Path(args.outputs))
    prompt = args.prompt

    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    session_id = run_dir.name
    tracer.start_run(
        name="run-prompt",
        metadata={
            "command": "run-prompt",
            "mock": args.mock,
            "prompt_hash": prompt_sha256[:16],
            "principal_prompt_length": len(prompt),
            "run_dir": run_dir.name,
        },
        session_id=session_id,
    )

    classify_prompt_observation = tracer.span if args.mock else tracer.generation
    with classify_prompt_observation(
        name="classify_prompt",
        metadata={
            "principal_prompt_length": len(prompt),
            "principal_prompt_sha256": prompt_sha256,
        },
        input={
            "principal_prompt_length": len(prompt),
            "principal_prompt_sha256": prompt_sha256,
            "mock": args.mock,
        },
    ) as span:
        prompt_classification_call: LLMCallResult[Any] | None = None
        if args.mock:
            prompt_classification = mock_classify_prompt(prompt)
        else:
            prompt_classification_call = classify_prompt_with_usage(prompt)
            prompt_classification = prompt_classification_call.output
        update_span_with_llm_call(
            span,
            output={
                "classification": prompt_classification.classification,
                "substituted": prompt_classification.substituted,
            },
            llm_call=prompt_classification_call,
        )

    prompt_run = build_prompt_run(prompt, prompt_classification)

    with tracer.span(
        name="emit_prompt_flow",
        metadata={"scenario_id": prompt_run.scenario_id},
        input={
            "scenario_id": prompt_run.scenario_id,
            "classification": prompt_classification.classification,
            "substituted": prompt_classification.substituted,
        },
    ) as span:
        flow = emit_prompt_flow(prompt_run)
        span.update(
            output={
                "classification": prompt_classification.classification,
                "flow_length": len(flow),
            }
        )

    artifact_names = [
        f"{prompt_run.scenario_id}.classification.json",
        f"{prompt_run.scenario_id}.flow.mmd",
    ]
    with tracer.span(
        name="write_artifacts",
        metadata={"scenario_id": prompt_run.scenario_id, "artifact_names": artifact_names},
        input={"scenario_id": prompt_run.scenario_id, "artifact_names": artifact_names},
    ) as span:
        write_model_json(
            run_dir / f"{prompt_run.scenario_id}.classification.json",
            prompt_run.classification,
        )
        write_text(run_dir / f"{prompt_run.scenario_id}.flow.mmd", flow)
        span.update(output={"artifact_count": len(artifact_names)})

    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        "1 classification files, "
        "0 response files, "
        "0 trace files, "
        "1 flow files, and "
        "0 state trace files."
    )
    tracer.flush()
    return 0


def run_loop_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))

    scenario_path = Path(args.scenario)
    run_dir = create_run_directory(Path(args.outputs))

    scenario = load_scenario(scenario_path)
    run_minimal_loop_to_artifact(
        scenario,
        run_dir,
        max_iterations=args.max_iterations,
        mock=args.mock,
    )

    print(f"Wrote outputs to {run_dir}")
    print("Wrote 1 loop trace file.")
    return 0


def detect_contract_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))

    run_dir = create_run_directory(Path(args.outputs))
    contract = detect_prompt_contract(args.prompt, mock=args.mock)
    write_prompt_contract(contract, run_dir)

    print(f"Wrote outputs to {run_dir}")
    print("Wrote 1 prompt contract file.")
    return 0


def run_prompt_loop_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))

    run_dir = create_run_directory(Path(args.outputs))
    run_prompt_loop_to_artifact(
        args.prompt,
        run_dir,
        mock=args.mock,
        max_iterations=args.max_iterations,
    )

    print(f"Wrote outputs to {run_dir}")
    print("Wrote 1 prompt contract file and 1 loop trace file.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whose-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run file-based scenarios.")
    run_parser.add_argument("--scenarios", required=True, help="Directory containing scenario YAML files.")
    run_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    run_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file for OpenRouter settings.")
    run_parser.add_argument("--mock", action="store_true", help="Use deterministic local bad responses.")
    run_parser.set_defaults(func=run_command)

    prompt_parser = subparsers.add_parser("run-prompt", help="Run one arbitrary principal prompt.")
    prompt_parser.add_argument("--prompt", required=True, help="Principal prompt text to classify and run.")
    prompt_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    prompt_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file for OpenRouter settings.")
    prompt_parser.add_argument("--mock", action="store_true", help="Use deterministic local classification and bad responses.")
    prompt_parser.set_defaults(func=run_prompt_command)

    loop_parser = subparsers.add_parser("run-loop", help="Run the minimal loop for one scenario.")
    loop_parser.add_argument("--scenario", required=True, help="Path to a scenario YAML file.")
    loop_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    loop_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file.")
    loop_parser.add_argument("--mock", action="store_true", help="Use deterministic local mock responses.")
    loop_parser.add_argument("--max-iterations", type=int, default=1, help="Maximum loop iterations (default: 1).")
    loop_parser.set_defaults(func=run_loop_command)

    contract_parser = subparsers.add_parser(
        "detect-contract",
        help="Detect a prompt contract for one arbitrary principal prompt.",
    )
    contract_parser.add_argument("--prompt", required=True, help="Principal prompt text to inspect.")
    contract_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    contract_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file.")
    contract_parser.add_argument("--mock", action="store_true", help="Use deterministic local contract detection.")
    contract_parser.set_defaults(func=detect_contract_command)

    prompt_loop_parser = subparsers.add_parser(
        "run-prompt-loop",
        help="Run experimental loop observability for one arbitrary prompt.",
    )
    prompt_loop_parser.add_argument("--prompt", required=True, help="Principal prompt text to inspect and loop.")
    prompt_loop_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    prompt_loop_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file.")
    prompt_loop_parser.add_argument("--mock", action="store_true", help="Use deterministic local contract detection and loop responses.")
    prompt_loop_parser.add_argument("--max-iterations", type=int, default=1, help="Maximum loop iterations (default: 1).")
    prompt_loop_parser.set_defaults(func=run_prompt_loop_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        BadResponseError,
        CheckerError,
        PromptClassifierError,
        PromptContractDetectorError,
        ReflectionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
