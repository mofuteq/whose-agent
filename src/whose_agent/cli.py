from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from whose_agent.bad_response import BadResponseError, generate_bad_response
from whose_agent.classifier import classify_scenario
from whose_agent.env_loader import load_env_file
from whose_agent.llm_classifier import PromptClassifierError, classify_prompt
from whose_agent.models import Classification, Trace
from whose_agent.prompt_run import build_prompt_run, mock_classify_prompt, to_scenario_classification
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

    scenarios_dir = Path(args.scenarios)
    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(scenarios_dir)
    classification_count = 0
    response_count = 0
    trace_count = 0

    for scenario in scenarios:
        classification: Classification = classify_scenario(scenario)
        write_model_json(outputs_dir / f"{scenario.scenario_id}.classification.json", classification)
        classification_count += 1

        if classification.classification == "out_of_scope":
            continue

        bad_response = generate_bad_response(scenario, classification, mock=args.mock)
        write_text(outputs_dir / f"{scenario.scenario_id}.response.md", bad_response)
        response_count += 1

        trace: Trace = emit_trace(scenario, classification, bad_response)
        write_model_json(outputs_dir / f"{scenario.scenario_id}.trace.json", trace)
        trace_count += 1

    print(
        "Wrote "
        f"{classification_count} classification files, "
        f"{response_count} response files, and "
        f"{trace_count} trace files."
    )
    return 0


def run_prompt_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))

    outputs_dir = Path(args.outputs)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt
    prompt_classification = (
        mock_classify_prompt(prompt) if args.mock else classify_prompt(prompt)
    )
    prompt_run = build_prompt_run(prompt, prompt_classification)
    write_model_json(
        outputs_dir / f"{prompt_run.scenario_id}.classification.json",
        prompt_run.classification,
    )

    response_count = 0
    trace_count = 0
    if prompt_run.scenario is not None:
        classification = to_scenario_classification(
            prompt_run.scenario,
            prompt_run.classification,
        )
        bad_response = generate_bad_response(
            prompt_run.scenario,
            classification,
            mock=args.mock,
        )
        write_text(outputs_dir / f"{prompt_run.scenario_id}.response.md", bad_response)
        response_count = 1

        trace: Trace = emit_trace(prompt_run.scenario, classification, bad_response)
        write_model_json(outputs_dir / f"{prompt_run.scenario_id}.trace.json", trace)
        trace_count = 1

    print(
        "Wrote "
        "1 classification files, "
        f"{response_count} response files, and "
        f"{trace_count} trace files."
    )
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
    except (BadResponseError, PromptClassifierError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
