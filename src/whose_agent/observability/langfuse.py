from __future__ import annotations

import os
import warnings
from typing import Any


class NoopSpan:
    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


class NoopTracer:
    enabled: bool = False

    def start_run(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        return None

    def span(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> NoopSpan:
        return NoopSpan()

    def flush(self) -> None:
        return None


class LangfuseSpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def __enter__(self) -> "LangfuseSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._span.end()

    def update(self, **kwargs: Any) -> None:
        self._span.update(**kwargs)


class LangfuseTracer:
    enabled: bool = True

    def __init__(self, public_key: str, secret_key: str, host: str | None = None) -> None:
        from langfuse import Langfuse

        init_kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        if host:
            init_kwargs["host"] = host
        self._client = Langfuse(**init_kwargs)
        self._trace: Any = None

    def start_run(
        self,
        *,
        name: str,
        metadata: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        trace_kwargs: dict[str, Any] = {"name": name, "metadata": metadata}
        if session_id:
            trace_kwargs["session_id"] = session_id
        self._trace = self._client.trace(**trace_kwargs)

    def span(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> LangfuseSpan:
        span_kwargs: dict[str, Any] = {"name": name, "metadata": metadata or {}}
        if input is not None:
            span_kwargs["input"] = input
        if output is not None:
            span_kwargs["output"] = output

        if self._trace is not None:
            raw_span = self._trace.span(**span_kwargs)
        else:
            raw_trace = self._client.trace(name=name)
            raw_span = raw_trace.span(**span_kwargs)
        return LangfuseSpan(raw_span)

    def flush(self) -> None:
        self._client.flush()


def create_observability_tracer() -> NoopTracer | LangfuseTracer:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if not public_key or not secret_key:
        return NoopTracer()

    try:
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host or None)
    except Exception as exc:
        warnings.warn(
            f"Langfuse initialization failed, using no-op tracer: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return NoopTracer()
