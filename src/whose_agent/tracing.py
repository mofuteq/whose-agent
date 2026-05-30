"""Optional Langfuse tracing helpers.

This module is the only place that talks to the Langfuse SDK directly. It
keeps CLI code on a small tracer interface and turns into no-ops when Langfuse
credentials are not configured.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_CLIENT: Any = None


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

    def generation(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
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
        try:
            self._span.update(**kwargs)
        except TypeError:
            legacy_kwargs = dict(kwargs)
            if "usage_details" in legacy_kwargs and "usage" not in legacy_kwargs:
                legacy_kwargs["usage"] = legacy_kwargs.pop("usage_details")
            self._span.update(**legacy_kwargs)


class LangfuseTracer:
    enabled: bool = True

    def __init__(self, public_key: str, secret_key: str, base_url: str | None = None) -> None:
        self._client = _create_client(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )
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

        self._propagation_context = _propagate_trace_attributes(
            trace_name=name,
            session_id=session_id,
        )
        if self._propagation_context is not None:
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
        return self._observation(
            name=name,
            metadata=metadata,
            input=input,
            output=output,
            as_type="span",
        )

    def generation(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
    ) -> LangfuseSpan:
        return self._observation(
            name=name,
            metadata=metadata,
            input=input,
            output=output,
            as_type="generation",
            model=model,
            model_parameters=model_parameters,
            usage_details=usage_details,
        )

    def _observation(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None,
        input: Any | None = None,
        output: Any | None = None,
        as_type: str = "span",
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        usage_details: dict[str, int] | None = None,
    ) -> LangfuseSpan:
        span_kwargs: dict[str, Any] = {"name": name, "metadata": metadata or {}}
        if input is not None:
            span_kwargs["input"] = input
        if output is not None:
            span_kwargs["output"] = output
        if as_type != "span":
            span_kwargs["as_type"] = as_type
        if model is not None:
            span_kwargs["model"] = model
        if model_parameters:
            span_kwargs["model_parameters"] = model_parameters
        if usage_details:
            span_kwargs["usage_details"] = usage_details

        if self._trace is not None:
            raw_span = self._start_legacy_observation(span_kwargs, as_type=as_type)
        elif self._root_span is not None:
            raw_span = self._root_span.start_observation(**span_kwargs)
        else:
            if self._trace_context:
                span_kwargs["trace_context"] = self._trace_context
            raw_span = self._client.start_observation(**span_kwargs)
        return LangfuseSpan(raw_span)

    def _start_legacy_observation(self, span_kwargs: dict[str, Any], *, as_type: str) -> Any:
        legacy_kwargs = dict(span_kwargs)
        legacy_kwargs.pop("as_type", None)
        if as_type == "generation" and hasattr(self._trace, "generation"):
            return self._trace.generation(**legacy_kwargs)

        legacy_kwargs.pop("model", None)
        legacy_kwargs.pop("model_parameters", None)
        legacy_kwargs.pop("usage_details", None)
        return self._trace.span(**legacy_kwargs)

    def flush(self) -> None:
        if self._root_span is not None:
            self._root_span.end()
            self._root_span = None
        if self._propagation_context is not None:
            self._propagation_context.__exit__(None, None, None)
            self._propagation_context = None
        self._client.flush()


def is_enabled() -> bool:
    """Return True when Langfuse public+secret keys are configured."""
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public and secret)


def get_client() -> Any:
    """Return a cached Langfuse client or None when disabled/unavailable."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not is_enabled():
        return None

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    try:
        _CLIENT = _create_client(
            public_key=public_key,
            secret_key=secret_key,
            base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
        )
        return _CLIENT
    except Exception as exc:
        logger.warning("Langfuse initialization failed: %s", exc)
        return None


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


def observe(
    *,
    name: str | None = None,
    as_type: str | None = None,
    capture_input: bool | None = False,
    capture_output: bool | None = False,
) -> Callable[[F], F]:
    """Return the Langfuse `@observe` decorator, or a no-op when disabled."""
    if not is_enabled():
        return _passthrough
    try:
        from langfuse import observe as langfuse_observe
    except Exception as exc:
        logger.warning("Langfuse observe import failed: %s", exc)
        return _passthrough

    kwargs: dict[str, Any] = {
        "capture_input": capture_input,
        "capture_output": capture_output,
    }
    if name is not None:
        kwargs["name"] = name
    if as_type is not None:
        kwargs["as_type"] = as_type
    return langfuse_observe(**kwargs)


def update_current_generation(**kwargs: Any) -> None:
    """Update the active generation observation with model and usage details."""
    client = get_client()
    if client is None:
        return
    try:
        if hasattr(client, "update_current_generation"):
            client.update_current_generation(**kwargs)
        else:
            client.update_current_span(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse generation update failed: %s", exc)


def update_current_trace(**kwargs: Any) -> None:
    """Best-effort update for active trace metadata on SDK versions that support it."""
    client = get_client()
    if client is None:
        return
    try:
        update = getattr(client, "update_current_trace", None)
        if update is not None:
            update(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse trace update failed: %s", exc)


def flush() -> None:
    """Flush pending Langfuse events; safe when disabled."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)


def _create_client(*, public_key: str, secret_key: str, base_url: str | None = None) -> Any:
    from langfuse import Langfuse

    init_kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
    if base_url:
        init_kwargs["base_url"] = base_url
    try:
        return Langfuse(**init_kwargs)
    except TypeError:
        if not base_url:
            raise
        init_kwargs.pop("base_url", None)
        init_kwargs["host"] = base_url
        return Langfuse(**init_kwargs)


def _propagate_trace_attributes(*, trace_name: str, session_id: str | None) -> Any:
    try:
        from langfuse import propagate_attributes
    except Exception:
        return None

    propagation_kwargs: dict[str, Any] = {"trace_name": trace_name}
    if session_id:
        propagation_kwargs["session_id"] = session_id
    return propagate_attributes(**propagation_kwargs)


def _passthrough(func: F) -> F:
    return func


__all__ = [
    "LangfuseSpan",
    "LangfuseTracer",
    "NoopSpan",
    "NoopTracer",
    "create_observability_tracer",
    "flush",
    "get_client",
    "is_enabled",
    "observe",
    "update_current_generation",
    "update_current_trace",
]
