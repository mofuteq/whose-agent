from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from whose_agent.bad_response import BadResponseError, generate_bad_response
from whose_agent.boundary_state.trace import BoundaryStateTrace, emit_state_trace
from whose_agent.classifier import classify_scenario
from whose_agent.env_loader import load_env_file
from whose_agent.flow_emitter import emit_prompt_flow
from whose_agent.llm_classifier import PromptClassifierError, classify_prompt
from whose_agent.models import Classification, Trace
from whose_agent.observability import create_observability_tracer
from whose_agent.prompt_run import build_prompt_run, mock_classify_prompt, to_scenario_classification
from whose_agent.reflection import ReflectionError
from whose_agent.run_directory import create_run_directory
from whose_agent.scenario_loader import load_scenarios
from whose_agent.trace_emitter import emit_trace


def write_model_json(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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

    for scenario in scenarios:
        with tracer.span(
            name="classify_scenario",
            metadata={
                "scenario_id": scenario.scenario_id,
                "expected_substituted": scenario.expected_substituted,
            },
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
            with tracer.span(
                name="write_artifacts",
                metadata={
                    "scenario_id": scenario.scenario_id,
                    "artifact_names": [f"{scenario.scenario_id}.classification.json"],
                },
            ):
                write_model_json(
                    run_dir / f"{scenario.scenario_id}.classification.json", classification
                )
            continue

        with tracer.span(
            name="generate_bad_response",
            metadata={
                "scenario_id": scenario.scenario_id,
                "substituted": classification.substituted,
                "mock": args.mock,
            },
        ) as span:
            bad_response = generate_bad_response(scenario, classification, mock=args.mock)
            span.update(output={"bad_response_length": len(bad_response)})

        with tracer.span(
            name="emit_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": args.mock},
        ) as span:
            trace: Trace = emit_trace(scenario, classification, bad_response, mock=args.mock)
            span.update(
                output={
                    "substituted": trace.substituted,
                    "failure_mode": trace.failure_mode,
                    "reflection_substituted": trace.reflection_substituted,
                }
            )

        with tracer.span(
            name="emit_state_trace",
            metadata={"scenario_id": scenario.scenario_id, "mock": args.mock},
        ) as span:
            state_trace: BoundaryStateTrace = emit_state_trace(
                scenario, classification, bad_response, mock=args.mock
            )
            if state_trace.transitions:
                final = state_trace.transitions[-1].state
                span.update(
                    output={
                        "reflection_matches_expected": final.reflection_matches_expected,
                        "boundary_flags": list(final.boundary_flags),
                        "next_action": final.next_action,
                    }
                )

        artifact_names = [
            f"{scenario.scenario_id}.classification.json",
            f"{scenario.scenario_id}.response.md",
            f"{scenario.scenario_id}.trace.json",
            f"{scenario.scenario_id}.state_trace.json",
        ]
        with tracer.span(
            name="write_artifacts",
            metadata={"scenario_id": scenario.scenario_id, "artifact_names": artifact_names},
        ):
            write_model_json(
                run_dir / f"{scenario.scenario_id}.classification.json", classification
            )
            write_text(run_dir / f"{scenario.scenario_id}.response.md", bad_response)
            write_model_json(run_dir / f"{scenario.scenario_id}.trace.json", trace)
            write_model_json(run_dir / f"{scenario.scenario_id}.state_trace.json", state_trace)

        response_count += 1
        trace_count += 1
        state_trace_count += 1

    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        f"{classification_count} classification files, "
        f"{response_count} response files, "
        f"{trace_count} trace files, and "
        f"{state_trace_count} state trace files."
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

    with tracer.span(
        name="classify_prompt",
        metadata={
            "principal_prompt_length": len(prompt),
            "principal_prompt_sha256": prompt_sha256,
        },
    ) as span:
        prompt_classification = (
            mock_classify_prompt(prompt) if args.mock else classify_prompt(prompt)
        )
        span.update(
            output={
                "classification": prompt_classification.classification,
                "substituted": prompt_classification.substituted,
            }
        )

    prompt_run = build_prompt_run(prompt, prompt_classification)

    with tracer.span(
        name="emit_prompt_flow",
        metadata={"scenario_id": prompt_run.scenario_id},
    ) as span:
        flow = emit_prompt_flow(prompt_run)
        span.update(output={"classification": prompt_classification.classification})

    response_count = 0
    trace_count = 0
    flow_count = 1
    state_trace_count = 0

    if prompt_run.scenario is not None:
        classification = to_scenario_classification(
            prompt_run.scenario,
            prompt_run.classification,
        )

        with tracer.span(
            name="generate_bad_response",
            metadata={
                "scenario_id": prompt_run.scenario_id,
                "substituted": classification.substituted,
                "mock": args.mock,
            },
        ) as span:
            bad_response = generate_bad_response(
                prompt_run.scenario,
                classification,
                mock=args.mock,
            )
            span.update(output={"bad_response_length": len(bad_response)})

        with tracer.span(
            name="emit_trace",
            metadata={"scenario_id": prompt_run.scenario_id, "mock": args.mock},
        ) as span:
            trace: Trace = emit_trace(
                prompt_run.scenario, classification, bad_response, mock=args.mock
            )
            span.update(
                output={
                    "substituted": trace.substituted,
                    "failure_mode": trace.failure_mode,
                    "reflection_substituted": trace.reflection_substituted,
                }
            )

        with tracer.span(
            name="emit_state_trace",
            metadata={"scenario_id": prompt_run.scenario_id, "mock": args.mock},
        ) as span:
            state_trace: BoundaryStateTrace = emit_state_trace(
                prompt_run.scenario, classification, bad_response, mock=args.mock
            )
            if state_trace.transitions:
                final = state_trace.transitions[-1].state
                span.update(
                    output={
                        "reflection_matches_expected": final.reflection_matches_expected,
                        "boundary_flags": list(final.boundary_flags),
                        "next_action": final.next_action,
                    }
                )

        artifact_names = [
            f"{prompt_run.scenario_id}.classification.json",
            f"{prompt_run.scenario_id}.response.md",
            f"{prompt_run.scenario_id}.trace.json",
            f"{prompt_run.scenario_id}.state_trace.json",
            f"{prompt_run.scenario_id}.flow.mmd",
        ]
        with tracer.span(
            name="write_artifacts",
            metadata={"scenario_id": prompt_run.scenario_id, "artifact_names": artifact_names},
        ):
            write_model_json(
                run_dir / f"{prompt_run.scenario_id}.classification.json",
                prompt_run.classification,
            )
            write_text(run_dir / f"{prompt_run.scenario_id}.flow.mmd", flow)
            write_text(run_dir / f"{prompt_run.scenario_id}.response.md", bad_response)
            write_model_json(run_dir / f"{prompt_run.scenario_id}.trace.json", trace)
            write_model_json(
                run_dir / f"{prompt_run.scenario_id}.state_trace.json", state_trace
            )

        response_count = 1
        trace_count = 1
        state_trace_count = 1
    else:
        artifact_names = [
            f"{prompt_run.scenario_id}.classification.json",
            f"{prompt_run.scenario_id}.flow.mmd",
        ]
        with tracer.span(
            name="write_artifacts",
            metadata={"scenario_id": prompt_run.scenario_id, "artifact_names": artifact_names},
        ):
            write_model_json(
                run_dir / f"{prompt_run.scenario_id}.classification.json",
                prompt_run.classification,
            )
            write_text(run_dir / f"{prompt_run.scenario_id}.flow.mmd", flow)

    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        "1 classification files, "
        f"{response_count} response files, "
        f"{trace_count} trace files, "
        f"{flow_count} flow files, and "
        f"{state_trace_count} state trace files."
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
    except (BadResponseError, PromptClassifierError, ReflectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
