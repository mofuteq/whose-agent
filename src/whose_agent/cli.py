from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from whose_agent.bad_response import BadResponseError, generate_bad_response_with_usage
from whose_agent.boundary_state.trace import BoundaryStateTrace, emit_state_trace_with_usage
from whose_agent.checker import CheckerError, check_with_usage
from whose_agent.classifier import classify_scenario
from whose_agent.env_loader import load_env_file
from whose_agent.flow_emitter import emit_prompt_flow
from whose_agent.llm_classifier import PromptClassifierError, classify_prompt_with_usage
from whose_agent.llm_result import LLMCallResult
from whose_agent.models import CheckerObservation, Classification, Scenario, Trace
from whose_agent.prompt_run import build_prompt_run, mock_classify_prompt
from whose_agent.reflection import ReflectionError
from whose_agent.run_directory import create_run_directory
from whose_agent.scenario_loader import load_scenarios
from whose_agent.trace_emitter import emit_trace_with_usage
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

    classification_count = 0
    response_count = 0
    trace_count = 0
    state_trace_count = 0
    checker_count = 0

    for scenario in scenarios:
        with tracer.span(
            name="classify_scenario",
            metadata={
                "scenario_id": scenario.scenario_id,
                "expected_substituted": scenario.expected_substituted,
            },
            input=sanitized_scenario_input(scenario),
        ) as span:
            classification: Classification = classify_scenario(scenario)
            span.update(
                output={
                    "classification": classification.classification,
                    "substituted": classification.substituted,
                }
            )

        classification_count += 1

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
                write_model_json(
                    run_dir / f"{scenario.scenario_id}.classification.json", classification
                )
                span.update(output={"artifact_count": len(artifact_names)})
            continue

        bad_response_observation = tracer.span if args.mock else tracer.generation
        with bad_response_observation(
            name="generate_bad_response",
            metadata={
                "scenario_id": scenario.scenario_id,
                "substituted": classification.substituted,
                "mock": args.mock,
            },
            input=sanitized_scenario_input(
                scenario,
                classification=classification.classification,
                substituted=classification.substituted,
                mock=args.mock,
            ),
        ) as span:
            bad_response_call = generate_bad_response_with_usage(
                scenario,
                classification,
                mock=args.mock,
            )
            bad_response = bad_response_call.output
            update_span_with_llm_call(
                span,
                output={"bad_response_length": len(bad_response)},
                llm_call=bad_response_call,
            )

        emit_trace_observation = tracer.span if args.mock else tracer.generation
        with emit_trace_observation(
            name="emit_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": args.mock},
            input=sanitized_scenario_input(
                scenario,
                substituted=classification.substituted,
                bad_response_length=len(bad_response),
                mock=args.mock,
            ),
        ) as span:
            trace_result = emit_trace_with_usage(
                scenario,
                classification,
                bad_response,
                mock=args.mock,
            )
            trace: Trace = trace_result.trace
            update_span_with_llm_call(
                span,
                output={
                    "substituted": trace.substituted,
                    "failure_mode": trace.failure_mode,
                    "reflection_substituted": trace.reflection_substituted,
                },
                llm_call=trace_result.reflection_call,
            )

        emit_state_trace_observation = tracer.span if args.mock else tracer.generation
        with emit_state_trace_observation(
            name="emit_state_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": args.mock},
            input=sanitized_scenario_input(
                scenario,
                substituted=classification.substituted,
                bad_response_length=len(bad_response),
                mock=args.mock,
            ),
        ) as span:
            state_trace_result = emit_state_trace_with_usage(
                scenario, classification, bad_response, mock=args.mock
            )
            state_trace: BoundaryStateTrace = state_trace_result.state_trace
            if state_trace.transitions:
                final = state_trace.transitions[-1].state
                update_span_with_llm_call(
                    span,
                    output={
                        "reflection_matches_expected": final.reflection_matches_expected,
                        "boundary_flags": list(final.boundary_flags),
                        "next_action": final.next_action,
                    },
                    llm_call=state_trace_result.reflection_call,
                )

        checker_observation: CheckerObservation | None = None
        if scenario.selected_skill_id is not None:
            check_observation = tracer.span if args.mock else tracer.generation
            with check_observation(
                name="check_artifact",
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "skill_id": scenario.selected_skill_id,
                    "mock": args.mock,
                },
                input=sanitized_scenario_input(
                    scenario,
                    selected_skill_id=scenario.selected_skill_id,
                    bad_response_length=len(bad_response),
                    bad_response_sha256=hashlib.sha256(bad_response.encode()).hexdigest(),
                    mock=args.mock,
                ),
            ) as span:
                checker_result = check_with_usage(
                    scenario,
                    bad_response,
                    mock=args.mock,
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

        artifact_names = [
            f"{scenario.scenario_id}.classification.json",
            f"{scenario.scenario_id}.response.md",
            f"{scenario.scenario_id}.trace.json",
            f"{scenario.scenario_id}.state_trace.json",
        ]
        if checker_observation is not None:
            artifact_names.append(f"{scenario.scenario_id}.checker.json")
        with tracer.span(
            name="write_artifacts",
            metadata={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
            input={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
        ) as span:
            write_model_json(
                run_dir / f"{scenario.scenario_id}.classification.json", classification
            )
            write_text(run_dir / f"{scenario.scenario_id}.response.md", bad_response)
            write_model_json(run_dir / f"{scenario.scenario_id}.trace.json", trace)
            write_model_json(run_dir / f"{scenario.scenario_id}.state_trace.json", state_trace)
            if checker_observation is not None:
                write_model_json(
                    run_dir / f"{scenario.scenario_id}.checker.json",
                    checker_observation,
                )
            span.update(output={"artifact_count": len(artifact_names)})

        response_count += 1
        trace_count += 1
        state_trace_count += 1
        if checker_observation is not None:
            checker_count += 1

    checker_file_label = "checker file" if checker_count == 1 else "checker files"
    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        f"{classification_count} classification files, "
        f"{response_count} response files, "
        f"{trace_count} trace files, "
        f"{state_trace_count} state trace files, and "
        f"{checker_count} {checker_file_label}."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BadResponseError, CheckerError, PromptClassifierError, ReflectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
