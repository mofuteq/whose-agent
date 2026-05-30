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

    def __init__(self, public_key: str, secret_key: str, base_url: str | None = None) -> None:
        from langfuse import Langfuse

        init_kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
        if base_url:
            init_kwargs["base_url"] = base_url
        try:
            self._client = Langfuse(**init_kwargs)
        except TypeError:
            if not base_url:
                raise
            init_kwargs.pop("base_url", None)
            init_kwargs["host"] = base_url
            self._client = Langfuse(**init_kwargs)
        self._trace: Any = None
        self._root_span: Any = None
        self._trace_context: dict[str, str] | None = None
        self._propagation_context: Any = None

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
        if hasattr(self._client, "trace"):
            self._trace = self._client.trace(**trace_kwargs)
            return

        self._trace = None
        self._trace_context = None
        if session_id and hasattr(self._client, "create_trace_id"):
            self._trace_context = {"trace_id": self._client.create_trace_id(seed=session_id)}

        try:
            from langfuse import propagate_attributes
        except ImportError:
            propagate_attributes = None

        if propagate_attributes is not None:
            propagation_kwargs: dict[str, Any] = {"trace_name": name}
            if session_id:
                propagation_kwargs["session_id"] = session_id
            self._propagation_context = propagate_attributes(**propagation_kwargs)
            self._propagation_context.__enter__()

        observation_kwargs: dict[str, Any] = {"name": name, "metadata": metadata}
        if self._trace_context:
            observation_kwargs["trace_context"] = self._trace_context
        self._root_span = self._client.start_observation(**observation_kwargs)

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
        elif self._root_span is not None:
            raw_span = self._root_span.start_observation(**span_kwargs)
        else:
            if self._trace_context:
                span_kwargs["trace_context"] = self._trace_context
            raw_span = self._client.start_observation(**span_kwargs)
        return LangfuseSpan(raw_span)

    def flush(self) -> None:
        if self._root_span is not None:
            self._root_span.end()
            self._root_span = None
        if self._propagation_context is not None:
            self._propagation_context.__exit__(None, None, None)
            self._propagation_context = None
        self._client.flush()


def create_observability_tracer() -> NoopTracer | LangfuseTracer:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = os.environ.get("LANGFUSE_BASE_URL")

    if not public_key or not secret_key:
        return NoopTracer()

    try:
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, base_url=base_url or None)
    except Exception as exc:
        warnings.warn(
            f"Langfuse initialization failed, using no-op tracer: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return NoopTracer()
