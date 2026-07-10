from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from whose_agent.bad_response import BadResponseError
from whose_agent.checker import CheckerError
from whose_agent.env_loader import load_env_file
from whose_agent.firing_signals import FiringSignals, QuotaSignal
from whose_agent.authority_provenance import (
    derive_external_persistence_provenance,
)
from whose_agent.conversation_view import project_messages
from whose_agent.execution import run_fixed_scenarios, run_prompt_loop
from whose_agent.history_adapter import (
    MessageHistoryError,
    conversation_from_prompt,
    current_principal_prompt,
    load_message_history_file,
)
from whose_agent.loop_artifacts import run_minimal_loop_to_artifact
from whose_agent.loop_artifacts import PROMPT_LOOP_GENERATED_FILENAME
from whose_agent.prompt_contract_artifacts import write_prompt_contract
from whose_agent.prompt_contract_detector import (
    PromptContractDetectorError,
    detect_prompt_contract,
)
from whose_agent.prompt_loop_seed import (
    DEFAULT_PROMPT_LOOP_PRESETS_DIR,
    PromptLoopSeed,
    resolve_prompt_loop_seed,
)
from whose_agent.reflection import ReflectionError
from whose_agent.run_directory import create_run_directory
from whose_agent.scenario_loader import load_scenario, load_scenarios
from whose_agent.schemas import ConversationMessage
from whose_agent.tracing import create_observability_tracer


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

    result = run_fixed_scenarios(
        scenarios_dir=scenarios_dir,
        outputs_dir=Path(args.outputs),
        run_dir=run_dir,
        mock=args.mock,
        tracer=tracer,
    )
    classification_count = result.classification_count
    response_count = result.response_count
    trace_count = result.trace_count
    state_trace_count = result.state_trace_count
    checker_count = result.checker_count
    checker_comparison_count = result.checker_comparison_count
    explanation_count = result.explanation_count

    checker_file_label = "checker file" if checker_count == 1 else "checker files"
    checker_comparison_file_label = (
        "checker comparison file"
        if checker_comparison_count == 1
        else "checker comparison files"
    )
    explanation_file_label = (
        "explanation file" if explanation_count == 1 else "explanation files"
    )
    print(f"Wrote outputs to {run_dir}")
    print(
        "Wrote "
        f"{classification_count} classification files, "
        f"{response_count} response files, "
        f"{trace_count} trace files, "
        f"{state_trace_count} state trace files, "
        f"{checker_count} {checker_file_label}, and "
        f"{checker_comparison_count} {checker_comparison_file_label}; "
        f"{explanation_count} {explanation_file_label}."
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
    prompt, messages = _prompt_input_from_args(args)
    authority_provenance = derive_external_persistence_provenance(
        project_messages(messages)
    )
    contract = detect_prompt_contract(
        prompt,
        mock=args.mock,
        authority_provenance=authority_provenance,
    )
    write_prompt_contract(contract, run_dir)

    print(f"Wrote outputs to {run_dir}")
    print("Wrote 1 prompt contract file.")
    return 0


def run_prompt_loop_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))

    run_dir = create_run_directory(Path(args.outputs))
    tracer = create_observability_tracer()
    tracer.start_run(
        name="run-prompt-loop",
        metadata={
            "command": "run-prompt-loop",
            "mock": args.mock,
            "run_dir": run_dir.name,
        },
        session_id=run_dir.name,
    )
    try:
        seed = _prompt_loop_seed_from_args(args)
        firing_signals = _firing_signals_from_args(args)
    except (MessageHistoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        tracer.flush()
        return 1
    result = run_prompt_loop(
        run_id=run_dir.name,
        outputs_dir=Path(args.outputs),
        run_dir=run_dir,
        mock=args.mock,
        max_iterations=args.max_iterations,
        firing_signals=firing_signals,
        seed=seed,
        tracer=tracer,
    )
    generated_emitted = PROMPT_LOOP_GENERATED_FILENAME in result.artifact_names
    explanation_emitted = "prompt_loop.explanation.json" in result.artifact_names

    print(f"Wrote outputs to {run_dir}")
    if not generated_emitted:
        if explanation_emitted:
            print(
                "Wrote 1 prompt contract file, 1 loop trace file, "
                "and 1 explanation file."
            )
        else:
            print("Wrote 1 prompt contract file and 1 loop trace file.")
    else:
        if explanation_emitted:
            print(
                "Wrote 1 prompt contract file, 1 loop trace file, "
                "1 generated file, and 1 explanation file."
            )
        else:
            print(
                "Wrote 1 prompt contract file, 1 loop trace file, and 1 generated file."
            )
    tracer.flush()
    return 0


def serve_command(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file))
    from whose_agent.agui_host import create_app

    import uvicorn

    app = create_app(
        scenarios_dir=Path(args.scenarios),
        presets_dir=Path(args.presets),
        outputs_dir=Path(args.outputs),
    )
    uvicorn.run(app, host=args.host, port=args.port)
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
    contract_input_group = contract_parser.add_mutually_exclusive_group(required=True)
    contract_input_group.add_argument("--prompt", help="Principal prompt text to inspect.")
    contract_input_group.add_argument(
        "--messages-file",
        help="JSON array of OpenAI-compatible role/content messages.",
    )
    contract_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    contract_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file.")
    contract_parser.add_argument("--mock", action="store_true", help="Use deterministic local contract detection.")
    contract_parser.set_defaults(func=detect_contract_command)

    prompt_loop_parser = subparsers.add_parser(
        "run-prompt-loop",
        help="Run experimental loop observability for one arbitrary prompt.",
    )
    prompt_loop_input_group = prompt_loop_parser.add_mutually_exclusive_group()
    prompt_loop_input_group.add_argument("--prompt", help="Principal prompt text to inspect and loop.")
    prompt_loop_input_group.add_argument(
        "--messages-file",
        help="JSON array of OpenAI-compatible role/content messages.",
    )
    prompt_loop_parser.add_argument(
        "--preset",
        help="Server-owned prompt-loop history preset id. Requires --prompt.",
    )
    prompt_loop_parser.add_argument(
        "--presets",
        default=str(DEFAULT_PROMPT_LOOP_PRESETS_DIR),
        help="Directory containing prompt-loop preset YAML files.",
    )
    prompt_loop_parser.add_argument("--outputs", required=True, help="Directory for generated output files.")
    prompt_loop_parser.add_argument("--env-file", default=".env", help="Path to a dotenv file.")
    prompt_loop_parser.add_argument("--mock", action="store_true", help="Use deterministic local contract detection and loop responses.")
    prompt_loop_parser.add_argument("--max-iterations", type=int, default=1, help="Maximum loop iterations (default: 1).")
    prompt_loop_parser.add_argument(
        "--firing-time",
        type=_parse_firing_time,
        default=None,
        help="Inject the prompt-loop firing time as an ISO-8601 datetime.",
    )
    prompt_loop_parser.add_argument(
        "--quota-used",
        type=float,
        default=None,
        help="Inject used quota for prompt-loop firing pressure.",
    )
    prompt_loop_parser.add_argument(
        "--quota-limit",
        type=float,
        default=None,
        help="Inject quota limit for prompt-loop firing pressure.",
    )
    prompt_loop_parser.set_defaults(func=run_prompt_loop_command)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the local FastAPI AG-UI execution host.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve_parser.add_argument(
        "--scenarios",
        default="scenarios",
        help="Directory containing scenario YAML files.",
    )
    serve_parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory for generated output files.",
    )
    serve_parser.add_argument(
        "--presets",
        default=str(DEFAULT_PROMPT_LOOP_PRESETS_DIR),
        help="Directory containing prompt-loop preset YAML files.",
    )
    serve_parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to a dotenv file.",
    )
    serve_parser.set_defaults(func=serve_command)

    return parser


def _parse_firing_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--firing-time must be an ISO-8601 datetime"
        ) from exc


def _firing_signals_from_args(args: argparse.Namespace) -> FiringSignals:
    quota_used = args.quota_used
    quota_limit = args.quota_limit
    if (quota_used is None) != (quota_limit is None):
        raise ValueError("--quota-used and --quota-limit must be provided together")
    if quota_used is not None and quota_used < 0:
        raise ValueError("--quota-used must be greater than or equal to 0")
    if quota_limit is not None and quota_limit <= 0:
        raise ValueError("--quota-limit must be greater than 0")
    quota = (
        QuotaSignal(used=quota_used, limit=quota_limit)
        if quota_used is not None and quota_limit is not None
        else None
    )
    return FiringSignals(
        time=args.firing_time or datetime.now().astimezone(),
        quota=quota,
    )


def _prompt_input_from_args(
    args: argparse.Namespace,
) -> tuple[str, list[ConversationMessage]]:
    messages_file = getattr(args, "messages_file", None)
    if messages_file is None:
        return args.prompt, conversation_from_prompt(args.prompt)

    messages = load_message_history_file(Path(messages_file))
    prompt = current_principal_prompt(messages)
    return prompt, messages


def _prompt_loop_seed_from_args(args: argparse.Namespace) -> PromptLoopSeed:
    messages_file = getattr(args, "messages_file", None)
    messages = (
        load_message_history_file(Path(messages_file))
        if messages_file is not None
        else None
    )
    return resolve_prompt_loop_seed(
        prompt=getattr(args, "prompt", None),
        messages=messages,
        preset_id=getattr(args, "preset", None),
        presets_dir=Path(getattr(args, "presets", DEFAULT_PROMPT_LOOP_PRESETS_DIR)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        BadResponseError,
        CheckerError,
        MessageHistoryError,
        PromptContractDetectorError,
        ReflectionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
