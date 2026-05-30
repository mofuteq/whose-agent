from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import single_run_dir
from whose_agent.tracing import (
    LangfuseTracer,
    NoopSpan,
    NoopTracer,
    create_observability_tracer,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------


class SpySpan:
    def __init__(self, name: str, metadata: dict[str, Any] | None, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.input = kwargs.get("input")
        self.output = kwargs.get("output")
        self.model = kwargs.get("model")
        self.model_parameters = kwargs.get("model_parameters")
        self.usage_details = kwargs.get("usage_details")

    def __enter__(self) -> "SpySpan":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        if "output" in kwargs:
            self.output = kwargs["output"]
        if "model" in kwargs:
            self.model = kwargs["model"]
        if "model_parameters" in kwargs:
            self.model_parameters = kwargs["model_parameters"]
        if "usage_details" in kwargs:
            self.usage_details = kwargs["usage_details"]


class SpyTracer:
    enabled: bool = False

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.spans: list[SpySpan] = []
        self.flush_count: int = 0

    def start_run(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        self.runs.append({"name": name, "metadata": metadata, "session_id": session_id})

    def span(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SpySpan:
        s = SpySpan(name, metadata, kwargs)
        self.spans.append(s)
        return s

    def generation(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SpySpan:
        return self.span(name=name, metadata=metadata, **kwargs)

    def flush(self) -> None:
        self.flush_count += 1


class FakeLangfuseV4Span:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.children: list[FakeLangfuseV4Span] = []
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def start_observation(self, **kwargs: Any) -> "FakeLangfuseV4Span":
        child = FakeLangfuseV4Span(kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class FakeLangfuseV4Client:
    def __init__(self) -> None:
        self.trace_id_seeds: list[str] = []
        self.observations: list[FakeLangfuseV4Span] = []
        self.flushed = False

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self.trace_id_seeds.append(seed or "")
        return "1" * 32

    def start_observation(self, **kwargs: Any) -> FakeLangfuseV4Span:
        span = FakeLangfuseV4Span(kwargs)
        self.observations.append(span)
        return span

    def flush(self) -> None:
        self.flushed = True


def _make_run_args(outputs: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scenarios=str(ROOT / "scenarios"),
        outputs=str(outputs),
        env_file="/nonexistent/.env",
        mock=True,
    )


def _make_prompt_args(outputs: Path, prompt: str = "Implement a CLI in Rust.") -> argparse.Namespace:
    return argparse.Namespace(
        prompt=prompt,
        outputs=str(outputs),
        env_file="/nonexistent/.env",
        mock=True,
    )


def _run_fixed_cli_subprocess(outputs: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run",
        "--scenarios",
        "scenarios",
        "--outputs",
        str(outputs),
        "--env-file",
        "/nonexistent/.env",
        "--mock",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Test 1: missing credentials → NoopTracer
# ---------------------------------------------------------------------------


def test_create_tracer_without_langfuse_credentials_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = create_observability_tracer()
    assert isinstance(tracer, NoopTracer)
    assert tracer.enabled is False


def test_create_tracer_with_only_public_key_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = create_observability_tracer()
    assert isinstance(tracer, NoopTracer)


def test_create_tracer_with_only_secret_key_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    tracer = create_observability_tracer()
    assert isinstance(tracer, NoopTracer)


# ---------------------------------------------------------------------------
# Test 2: NoopTracer methods do not raise
# ---------------------------------------------------------------------------


def test_noop_tracer_methods_do_not_raise() -> None:
    tracer = NoopTracer()
    tracer.start_run(name="test-run", metadata={"command": "run", "mock": True})
    with tracer.span(name="step", metadata={"scenario_id": "s1"}) as span:
        span.update(output={"classification": "in_scope"})
    tracer.flush()


def test_noop_span_context_manager() -> None:
    span = NoopSpan()
    with span as s:
        s.update(output={"x": 1}, extra="ignored")
    assert span is s


# ---------------------------------------------------------------------------
# Test 3: CLI mock run succeeds without Langfuse env vars
# ---------------------------------------------------------------------------


def test_cli_mock_run_succeeds_without_langfuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    result = _run_fixed_cli_subprocess(tmp_path)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Test 4: CLI mock run emits same artifacts without Langfuse env vars
# ---------------------------------------------------------------------------


def test_cli_mock_run_emits_same_artifacts_without_langfuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _run_fixed_cli_subprocess(tmp_path)
    run_dir = single_run_dir(tmp_path)
    assert len(list(run_dir.glob("*.classification.json"))) == 7
    assert len(list(run_dir.glob("*.response.md"))) == 5
    assert len([f for f in run_dir.glob("*.trace.json") if not f.name.endswith(".state_trace.json")]) == 5
    assert len(list(run_dir.glob("*.state_trace.json"))) == 5
    assert list(run_dir.glob("*.flow.mmd")) == []


# ---------------------------------------------------------------------------
# Test 5: fake credentials + mocked SDK → LangfuseTracer enabled
# ---------------------------------------------------------------------------


def test_create_tracer_with_langfuse_credentials_enables_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-fake")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    mock_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = create_observability_tracer()

    assert isinstance(tracer, LangfuseTracer)
    assert tracer.enabled is True


def test_langfuse_tracer_passes_base_url_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://my.langfuse.example")

    mock_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_client) as mock_cls:
        create_observability_tracer()

    _, kwargs = mock_cls.call_args
    assert kwargs.get("base_url") == "https://my.langfuse.example"


# ---------------------------------------------------------------------------
# Test 6: tracer receives small metadata (scenario_id, substituted, failure_mode)
# ---------------------------------------------------------------------------


def test_tracer_receives_scenario_id_in_classify_span(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    classify_spans = [s for s in tracer.spans if s.name == "classify_scenario"]
    assert classify_spans
    for s in classify_spans:
        assert "scenario_id" in s.metadata
        assert s.metadata["scenario_id"]


def test_tracer_receives_substituted_in_run_level_metadata(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    assert tracer.runs
    run_meta = tracer.runs[0]["metadata"]
    assert run_meta["command"] == "run"
    assert run_meta["mock"] is True
    assert isinstance(run_meta["scenario_count"], int)


def test_tracer_receives_failure_mode_in_emit_trace_span(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    emit_trace_spans = [s for s in tracer.spans if s.name == "emit_trace"]
    assert emit_trace_spans
    for s in emit_trace_spans:
        assert s.output is not None
        assert "failure_mode" in s.output
        assert "substituted" in s.output


def test_tracer_receives_boundary_flags_in_state_trace_span(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    state_spans = [s for s in tracer.spans if s.name == "emit_state_trace"]
    assert state_spans
    for s in state_spans:
        assert s.output is not None
        assert "boundary_flags" in s.output
        assert "next_action" in s.output


def test_run_command_tracer_spans_have_sanitized_input_and_output(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    assert tracer.spans
    for span in tracer.spans:
        assert span.input is not None, span.name
        assert span.output is not None, span.name
        _assert_no_large_strings(span.input)
        _assert_no_large_strings(span.output)

    classify_span = next(s for s in tracer.spans if s.name == "classify_scenario")
    assert "principal_prompt_length" in classify_span.input
    assert "principal_prompt_sha256" in classify_span.input
    assert len(classify_span.input["principal_prompt_sha256"]) == 64


def test_run_prompt_tracer_spans_have_sanitized_input_and_output(tmp_path: Path) -> None:
    from whose_agent.cli import run_prompt_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_prompt_command(_make_prompt_args(tmp_path))

    assert tracer.spans
    for span in tracer.spans:
        assert span.input is not None, span.name
        assert span.output is not None, span.name
        _assert_no_large_strings(span.input)
        _assert_no_large_strings(span.output)

    classify_span = next(s for s in tracer.spans if s.name == "classify_prompt")
    assert classify_span.input["principal_prompt_length"] == len("Implement a CLI in Rust.")
    assert len(classify_span.input["principal_prompt_sha256"]) == 64


# ---------------------------------------------------------------------------
# Test 7: tracer does not receive full bad_response text
# ---------------------------------------------------------------------------


def test_tracer_does_not_receive_full_bad_response(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    bad_response_spans = [s for s in tracer.spans if s.name == "generate_bad_response"]
    assert bad_response_spans
    for s in bad_response_spans:
        assert s.output is not None
        assert "bad_response_length" in s.output
        assert isinstance(s.output["bad_response_length"], int)
        assert "bad_response" not in s.output or not isinstance(s.output.get("bad_response"), str)

    # No span should carry the raw bad_response text anywhere
    for s in tracer.spans:
        _assert_no_large_strings(s.input)
        _assert_no_large_strings(s.output)


def _assert_no_large_strings(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_large_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_no_large_strings(item)
        return
    if isinstance(value, str):
        # Bad responses in mock mode are hundreds of chars; trace values should be short.
        assert len(value) < 200, f"Suspiciously long string in traced value: {value[:80]!r}..."


# ---------------------------------------------------------------------------
# Test 8: tracer.flush() called at end of each command
# ---------------------------------------------------------------------------


def test_tracer_flush_called_at_end_of_run_command(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        result = run_command(_make_run_args(tmp_path))

    assert result == 0
    assert tracer.flush_count == 1


def test_tracer_flush_called_at_end_of_run_prompt_command(tmp_path: Path) -> None:
    from whose_agent.cli import run_prompt_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        result = run_prompt_command(_make_prompt_args(tmp_path))

    assert result == 0
    assert tracer.flush_count == 1


# ---------------------------------------------------------------------------
# Test 9: .env.example documents optional Langfuse variables as comments
# ---------------------------------------------------------------------------


def test_env_example_documents_optional_langfuse_variables() -> None:
    content = (ROOT / ".env.example").read_text()
    assert "LANGFUSE_PUBLIC_KEY" in content
    assert "LANGFUSE_SECRET_KEY" in content
    assert "LANGFUSE_BASE_URL" in content
    assert "LANGFUSE_HOST" not in content
    # Must be commented out (optional, not required)
    for line in content.splitlines():
        if "LANGFUSE_PUBLIC_KEY" in line and not line.startswith("#"):
            pytest.fail(f"LANGFUSE_PUBLIC_KEY should be commented in .env.example: {line!r}")
        if "LANGFUSE_SECRET_KEY" in line and not line.startswith("#"):
            pytest.fail(f"LANGFUSE_SECRET_KEY should be commented in .env.example: {line!r}")


# ---------------------------------------------------------------------------
# session_id tests
# ---------------------------------------------------------------------------


def test_noop_tracer_start_run_accepts_session_id() -> None:
    tracer = NoopTracer()
    tracer.start_run(name="run", metadata={"command": "run"}, session_id="20260530T120000Z")


def test_langfuse_tracer_start_run_passes_session_id_to_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = create_observability_tracer()

    assert isinstance(tracer, LangfuseTracer)
    tracer.start_run(name="run", metadata={"command": "run"}, session_id="20260530T120000Z")
    mock_client.trace.assert_called_once()
    _, trace_kwargs = mock_client.trace.call_args
    assert trace_kwargs["session_id"] == "20260530T120000Z"


def test_langfuse_tracer_start_run_omits_session_id_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=mock_client):
        tracer = create_observability_tracer()

    tracer.start_run(name="run", metadata={"command": "run"}, session_id=None)
    _, trace_kwargs = mock_client.trace.call_args
    assert "session_id" not in trace_kwargs


def test_langfuse_tracer_supports_v4_observation_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_client = FakeLangfuseV4Client()
    with patch("langfuse.Langfuse", return_value=fake_client):
        tracer = create_observability_tracer()

    assert isinstance(tracer, LangfuseTracer)
    tracer.start_run(name="run", metadata={"command": "run"}, session_id="20260530T120000Z")
    with tracer.span(name="classify_scenario", metadata={"scenario_id": "s1"}) as span:
        span.update(output={"classification": "in_scope"})
    tracer.flush()

    assert fake_client.trace_id_seeds == ["20260530T120000Z"]
    root = fake_client.observations[0]
    assert root.kwargs["name"] == "run"
    assert root.kwargs["metadata"] == {"command": "run"}
    assert root.kwargs["trace_context"] == {"trace_id": "1" * 32}
    assert root.ended is True

    child = root.children[0]
    assert child.kwargs["name"] == "classify_scenario"
    assert child.kwargs["metadata"] == {"scenario_id": "s1"}
    assert child.updates == [{"output": {"classification": "in_scope"}}]
    assert child.ended is True
    assert fake_client.flushed is True


def test_langfuse_tracer_supports_v4_generation_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_client = FakeLangfuseV4Client()
    with patch("langfuse.Langfuse", return_value=fake_client):
        tracer = create_observability_tracer()

    assert isinstance(tracer, LangfuseTracer)
    tracer.start_run(name="run", metadata={"command": "run"}, session_id="20260530T120000Z")
    with tracer.generation(
        name="generate_bad_response",
        metadata={"scenario_id": "s1"},
        model="openrouter:test/model",
        model_parameters={"temperature": 0.2},
        usage_details={"input": 11, "output": 7, "total": 18},
    ) as span:
        span.update(
            output={"bad_response_length": 42},
            usage_details={"input": 11, "output": 7, "total": 18},
        )
    tracer.flush()

    child = fake_client.observations[0].children[0]
    assert child.kwargs["as_type"] == "generation"
    assert child.kwargs["model"] == "openrouter:test/model"
    assert child.kwargs["model_parameters"] == {"temperature": 0.2}
    assert child.kwargs["usage_details"] == {"input": 11, "output": 7, "total": 18}
    assert child.updates == [
        {
            "output": {"bad_response_length": 42},
            "usage_details": {"input": 11, "output": 7, "total": 18},
        }
    ]


def test_run_command_uses_run_dir_name_as_session_id(tmp_path: Path) -> None:
    import re
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    assert tracer.runs
    run_record = tracer.runs[0]
    session_id = run_record["session_id"]
    assert session_id is not None
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:-\d{3})?", session_id), f"Unexpected session_id: {session_id!r}"
    # Must not be an absolute path
    assert not session_id.startswith("/")


def test_run_command_metadata_includes_run_dir_name_not_path(tmp_path: Path) -> None:
    from whose_agent.cli import run_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_command(_make_run_args(tmp_path))

    run_meta = tracer.runs[0]["metadata"]
    assert "run_dir" in run_meta
    run_dir_value = run_meta["run_dir"]
    assert not run_dir_value.startswith("/"), "run_dir in metadata must be a name, not an absolute path"
    assert run_dir_value == tracer.runs[0]["session_id"]


def test_run_prompt_command_uses_run_dir_name_as_session_id(tmp_path: Path) -> None:
    import re
    from whose_agent.cli import run_prompt_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_prompt_command(_make_prompt_args(tmp_path))

    assert tracer.runs
    run_record = tracer.runs[0]
    session_id = run_record["session_id"]
    assert session_id is not None
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:-\d{3})?", session_id), f"Unexpected session_id: {session_id!r}"
    assert not session_id.startswith("/")


def test_run_prompt_command_metadata_includes_run_dir_name_not_path(tmp_path: Path) -> None:
    from whose_agent.cli import run_prompt_command

    tracer = SpyTracer()
    with patch("whose_agent.cli.create_observability_tracer", return_value=tracer):
        run_prompt_command(_make_prompt_args(tmp_path))

    run_meta = tracer.runs[0]["metadata"]
    assert "run_dir" in run_meta
    assert not run_meta["run_dir"].startswith("/")


# ---------------------------------------------------------------------------
# Test 10: no OpenRouter credentials required for mock mode without Langfuse
# ---------------------------------------------------------------------------


def test_mock_run_without_openrouter_and_without_langfuse(tmp_path: Path) -> None:
    result = _run_fixed_cli_subprocess(
        tmp_path,
        extra_env={
            "OPENROUTER_API_KEY": "",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "",
        },
    )
    assert result.returncode == 0, result.stderr
