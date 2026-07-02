from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ag_ui.core import (
    CustomEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from whose_agent.execution import (
    RunnerEvent,
    list_fixed_scenarios,
    list_server_prompt_loop_presets,
    load_known_scenario,
    stream_fixed_scenario,
    stream_prompt_loop,
)
from whose_agent.history_adapter import (
    MessageHistoryError,
    normalize_role_tagged_messages,
)
from whose_agent.prompt_loop_seed import (
    DEFAULT_PROMPT_LOOP_PRESETS_DIR,
    PromptLoopSeed,
    resolve_prompt_loop_seed,
)
from whose_agent.public_projection import (
    CompletedProjection,
    RunMode,
    RunProjection,
    SafeErrorCode,
)
from whose_agent.schemas import ConversationMessage, Scenario


MAX_API_ITERATIONS = 3
PUBLIC_THREAD_ID_PATTERN = re.compile(r"^(?=.*[0-9])[A-Za-z0-9_-]{1,128}$")


class WhoseAgentOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RunMode
    scenario_id: str | None = None
    preset_id: str | None = None
    prompt: str | None = None
    mock: bool = True
    max_iterations: int = Field(default=1, ge=1, le=MAX_API_ITERATIONS)

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "WhoseAgentOptions":
        if self.mode == "fixed" and self.scenario_id is None:
            raise ValueError("fixed mode requires scenario_id")
        if self.mode == "fixed" and self.preset_id is not None:
            raise ValueError("fixed mode does not accept preset_id")
        if self.mode == "fixed" and self.prompt is not None:
            raise ValueError("fixed mode does not accept prompt")
        if self.mode == "prompt_loop" and self.scenario_id is not None:
            raise ValueError("prompt_loop mode does not accept scenario_id")
        return self


@dataclass(frozen=True)
class ExecutionRequest:
    thread_id: str
    mode: RunMode
    mock: bool
    max_iterations: int
    scenario: Scenario | None = None
    prompt: str | None = None
    messages: list[ConversationMessage] | None = None
    seed: PromptLoopSeed | None = None


class AguiRequestError(ValueError):
    def __init__(self, code: SafeErrorCode):
        super().__init__(code)
        self.code = code


class RunRegistry:
    def __init__(self) -> None:
        self._records: dict[str, RunProjection] = {}

    def start(self, *, run_id: str, thread_id: str, mode: RunMode) -> None:
        self._records[run_id] = RunProjection(
            run_id=run_id,
            thread_id=thread_id,
            status="in_progress",
            mode=mode,
        )

    def complete(self, *, run_id: str, result: CompletedProjection) -> None:
        current = self._records[run_id]
        self._records[run_id] = current.model_copy(
            update={
                "status": "completed",
                "result": result,
                "artifact_names": result.artifact_names,
                "safe_error_code": None,
            }
        )

    def fail(self, *, run_id: str, code: SafeErrorCode, cancelled: bool = False) -> None:
        current = self._records.get(run_id)
        if current is None:
            return
        self._records[run_id] = current.model_copy(
            update={
                "status": "cancelled" if cancelled else "failed",
                "safe_error_code": code,
            }
        )

    def get(self, run_id: str) -> RunProjection | None:
        return self._records.get(run_id)


def create_app(
    *,
    scenarios_dir: Path | str = Path("scenarios"),
    presets_dir: Path | str = DEFAULT_PROMPT_LOOP_PRESETS_DIR,
    outputs_dir: Path | str = Path("outputs"),
    frontend_dist_dir: Path | str | None = Path("frontend/dist"),
    registry: RunRegistry | None = None,
) -> FastAPI:
    scenarios_path = Path(scenarios_dir)
    presets_path = Path(presets_dir)
    outputs_path = Path(outputs_dir)
    frontend_dist_path = (
        Path(frontend_dist_dir) if frontend_dist_dir is not None else None
    )
    run_registry = registry if registry is not None else RunRegistry()
    app = FastAPI(title="whose-agent local AG-UI host")
    app.state.run_registry = run_registry

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/scenarios")
    async def scenarios() -> dict[str, list[dict[str, object]]]:
        return {
            "scenarios": [
                item.model_dump(mode="json")
                for item in list_fixed_scenarios(scenarios_path)
            ]
        }

    @app.get("/api/prompt-loop-presets")
    async def prompt_loop_presets() -> dict[str, list[dict[str, object]]]:
        return {
            "prompt_loop_presets": [
                item.model_dump(mode="json")
                for item in list_server_prompt_loop_presets(presets_path)
            ]
        }

    @app.post("/agui")
    async def agui(request: Request) -> StreamingResponse:
        encoder = EventEncoder(accept=request.headers.get("accept"))
        run_id = _new_run_id()
        try:
            body = await request.json()
            run_input = RunAgentInput.model_validate(body)
            execution_request = _execution_request_from_input(
                run_input,
                scenarios_path,
                presets_path,
            )
        except AguiRequestError as exc:
            thread_id = _public_thread_id_from_body_or_generated(locals().get("body"))
            return _safe_error_response(
                encoder=encoder,
                thread_id=thread_id,
                run_id=run_id,
                code=exc.code,
            )
        except (ValidationError, ValueError, TypeError):
            thread_id = _public_thread_id_from_body_or_generated(locals().get("body"))
            return _safe_error_response(
                encoder=encoder,
                thread_id=thread_id,
                run_id=run_id,
                code="invalid_request",
            )

        async def event_generator() -> Any:
            run_registry.start(
                run_id=run_id,
                thread_id=execution_request.thread_id,
                mode=execution_request.mode,
            )
            try:
                yield encoder.encode(
                    RunStartedEvent(
                        thread_id=execution_request.thread_id,
                        run_id=run_id,
                    )
                )
                yield encoder.encode(
                    CustomEvent(
                        name="whose_agent.run.started",
                        value={
                            "run_id": run_id,
                            "thread_id": execution_request.thread_id,
                            "mode": execution_request.mode,
                            "status": "in_progress",
                        },
                    )
                )
                async for runner_event in _runner_events(
                    run_id=run_id,
                    execution_request=execution_request,
                    outputs_dir=outputs_path,
                ):
                    encoded = _encode_runner_event(
                        encoder=encoder,
                        event=runner_event,
                        thread_id=execution_request.thread_id,
                        run_id=run_id,
                        registry=run_registry,
                    )
                    if encoded is not None:
                        for event_chunk in encoded:
                            yield event_chunk
            except asyncio.CancelledError:
                run_registry.fail(
                    run_id=run_id,
                    code="stream_cancelled",
                    cancelled=True,
                )
                raise
            except Exception:
                run_registry.fail(run_id=run_id, code="run_failed")
                yield encoder.encode(
                    RunErrorEvent(
                        message="Run failed.",
                        code="run_failed",
                    )
                )

        return StreamingResponse(
            event_generator(),
            media_type=encoder.get_content_type(),
        )

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, object]:
        record = run_registry.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "not_found"})
        return record.model_dump(mode="json")

    if frontend_dist_path is not None and frontend_dist_path.exists():
        app.mount(
            "/",
            StaticFiles(directory=frontend_dist_path, html=True),
            name="frontend",
        )
    else:

        @app.get("/", response_class=PlainTextResponse)
        async def frontend_missing() -> str:
            return (
                "Built frontend not found. For development, run `cd frontend && "
                "npm run dev`. For local built serving, run `cd frontend && "
                "npm run build` first."
            )

    return app


def _safe_error_response(
    *,
    encoder: EventEncoder,
    thread_id: str,
    run_id: str,
    code: SafeErrorCode,
) -> StreamingResponse:
    async def event_generator() -> Any:
        yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id))
        yield encoder.encode(RunErrorEvent(message=_safe_error_message(code), code=code))

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
    )


def _execution_request_from_input(
    run_input: RunAgentInput,
    scenarios_dir: Path,
    presets_dir: Path,
) -> ExecutionRequest:
    _reject_client_controlled_execution_surfaces(run_input)
    options = _options_from_state(run_input.state)
    thread_id = _public_thread_id_or_generated(run_input.thread_id)
    if options.mode == "fixed":
        scenario = load_known_scenario(scenarios_dir, options.scenario_id or "")
        if scenario is None:
            raise AguiRequestError("unknown_scenario")
        return ExecutionRequest(
            thread_id=thread_id,
            mode="fixed",
            mock=options.mock,
            max_iterations=options.max_iterations,
            scenario=scenario,
        )

    has_prompt = options.prompt is not None
    has_messages = bool(run_input.messages)
    if options.preset_id is not None:
        if not has_prompt:
            raise AguiRequestError("invalid_request")
        if run_input.messages:
            raise AguiRequestError("invalid_request")
        try:
            seed = resolve_prompt_loop_seed(
                prompt=options.prompt,
                preset_id=options.preset_id,
                presets_dir=presets_dir,
            )
        except ValueError as exc:
            raise AguiRequestError("invalid_request") from exc
    elif has_prompt:
        if has_messages:
            raise AguiRequestError("invalid_request")
        try:
            seed = resolve_prompt_loop_seed(prompt=options.prompt)
        except ValueError as exc:
            raise AguiRequestError("invalid_request") from exc
    else:
        messages = _canonical_messages_from_agui(run_input)
        try:
            seed = resolve_prompt_loop_seed(messages=messages)
        except ValueError as exc:
            raise AguiRequestError("invalid_request") from exc
    return ExecutionRequest(
        thread_id=thread_id,
        mode="prompt_loop",
        mock=options.mock,
        max_iterations=options.max_iterations,
        prompt=seed.current_principal_prompt,
        messages=seed.messages,
        seed=seed,
    )


def _reject_client_controlled_execution_surfaces(run_input: RunAgentInput) -> None:
    if run_input.tools:
        raise AguiRequestError("invalid_request")
    if run_input.context:
        raise AguiRequestError("invalid_request")
    if run_input.forwarded_props:
        raise AguiRequestError("invalid_request")


def _options_from_state(state: object) -> WhoseAgentOptions:
    if not isinstance(state, dict):
        raise AguiRequestError("invalid_request")
    raw_options = state.get("whose_agent")
    if not isinstance(raw_options, dict):
        raise AguiRequestError("invalid_request")
    try:
        return WhoseAgentOptions.model_validate(raw_options)
    except ValidationError as exc:
        raise AguiRequestError("invalid_request") from exc


def _canonical_messages_from_agui(
    run_input: RunAgentInput,
) -> list[ConversationMessage]:
    records: list[dict[str, str]] = []
    for message in run_input.messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role not in {"user", "assistant", "tool", "system"}:
            raise AguiRequestError("invalid_request")
        if not isinstance(content, str):
            raise AguiRequestError("invalid_request")
        records.append({"role": role, "content": content})
    try:
        return normalize_role_tagged_messages(records)
    except MessageHistoryError as exc:
        raise AguiRequestError("invalid_request") from exc


async def _runner_events(
    *,
    run_id: str,
    execution_request: ExecutionRequest,
    outputs_dir: Path,
) -> AsyncIterator[RunnerEvent]:
    if execution_request.mode == "fixed":
        if execution_request.scenario is None:
            raise RuntimeError("fixed execution requires scenario")
        async for event in stream_fixed_scenario(
            run_id=run_id,
            scenario=execution_request.scenario,
            outputs_dir=outputs_dir,
            mock=execution_request.mock,
        ):
            yield event
        return

    if execution_request.seed is None:
        raise RuntimeError("prompt-loop execution requires a prompt-loop seed")
    async for event in stream_prompt_loop(
        run_id=run_id,
        outputs_dir=outputs_dir,
        mock=execution_request.mock,
        max_iterations=execution_request.max_iterations,
        seed=execution_request.seed,
    ):
        yield event


def _encode_runner_event(
    *,
    encoder: EventEncoder,
    event: RunnerEvent,
    thread_id: str,
    run_id: str,
    registry: RunRegistry,
) -> list[str] | None:
    if event.kind == "phase" and event.phase is not None:
        return [
            encoder.encode(
                CustomEvent(
                    name="whose_agent.phase",
                    value=event.phase.model_dump(mode="json"),
                )
            )
        ]
    if event.kind == "cause" and event.cause is not None:
        return [
            encoder.encode(
                CustomEvent(
                    name="whose_agent.cause",
                    value=event.cause.model_dump(mode="json"),
                )
            )
        ]
    if event.kind == "checker" and event.checker is not None:
        return [
            encoder.encode(
                CustomEvent(
                    name="whose_agent.checker",
                    value=event.checker.model_dump(mode="json"),
                )
            )
        ]
    if event.kind == "explain" and event.explain is not None:
        return [
            encoder.encode(
                CustomEvent(
                    name="whose_agent.explain",
                    value=event.explain.model_dump(mode="json"),
                )
            )
        ]
    if event.kind == "text" and event.text is not None:
        message_id = f"assistant_{run_id}_{uuid4().hex}"
        return [
            encoder.encode(TextMessageStartEvent(message_id=message_id)),
            encoder.encode(
                TextMessageContentEvent(
                    message_id=message_id,
                    delta=event.text,
                )
            ),
            encoder.encode(TextMessageEndEvent(message_id=message_id)),
        ]
    if event.kind == "completed" and event.result is not None:
        completed = event.result.completed
        registry.complete(run_id=run_id, result=completed)
        completed_payload = completed.model_dump(mode="json")
        return [
            encoder.encode(
                CustomEvent(
                    name="whose_agent.run.completed",
                    value=completed_payload,
                )
            ),
            encoder.encode(
                RunFinishedEvent(
                    thread_id=thread_id,
                    run_id=run_id,
                    result=completed_payload,
                )
            ),
        ]
    return None


def _public_thread_id_or_generated(value: object) -> str:
    if isinstance(value, str):
        thread_id = value.strip()
        if PUBLIC_THREAD_ID_PATTERN.fullmatch(thread_id):
            return thread_id
    return _new_thread_id()


def _public_thread_id_from_body_or_generated(body: object) -> str:
    if isinstance(body, dict):
        return _public_thread_id_or_generated(body.get("threadId"))
    return _new_thread_id()


def _new_thread_id() -> str:
    return f"thread_1_{uuid4().hex}"


def _new_run_id() -> str:
    return f"run_{uuid4().hex}"


def _safe_error_message(code: SafeErrorCode) -> str:
    messages: dict[SafeErrorCode, str] = {
        "invalid_request": "Invalid request.",
        "unknown_scenario": "Unknown scenario.",
        "run_failed": "Run failed.",
        "stream_cancelled": "Stream cancelled.",
    }
    return messages[code]
