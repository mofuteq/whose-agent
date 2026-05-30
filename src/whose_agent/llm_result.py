from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LLMCallResult(Generic[T]):
    output: T
    model_name: str | None = None
    model_settings: dict[str, float | int] = field(default_factory=dict)
    usage_details: dict[str, int] = field(default_factory=dict)


def extract_output(result: Any) -> Any:
    return getattr(result, "output", getattr(result, "data", result))


def extract_usage_details(result: Any) -> dict[str, int]:
    usage = getattr(result, "usage", None)
    if callable(usage):
        usage = usage()
    if usage is None:
        return {}

    details: dict[str, int] = {}
    _copy_usage_value(details, usage, "input_tokens", "input")
    _copy_usage_value(details, usage, "output_tokens", "output")
    _copy_usage_value(details, usage, "total_tokens", "total")
    _copy_usage_value(details, usage, "cache_write_tokens", "cache_creation_input")
    _copy_usage_value(details, usage, "cache_read_tokens", "cache_read_input")
    _copy_usage_value(details, usage, "input_audio_tokens", "audio_input")
    _copy_usage_value(details, usage, "output_audio_tokens", "audio_output")
    _copy_usage_value(details, usage, "cache_audio_read_tokens", "cache_read_audio_input")

    extra_details = getattr(usage, "details", None)
    if isinstance(extra_details, dict):
        for key, value in extra_details.items():
            if isinstance(value, bool) or not isinstance(value, int) or value == 0:
                continue
            safe_key = str(key).strip().replace(" ", "_")
            if safe_key and safe_key not in details:
                details[safe_key] = value

    return details


def _copy_usage_value(
    target: dict[str, int],
    usage: Any,
    source_key: str,
    target_key: str,
) -> None:
    value = getattr(usage, source_key, None)
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        return
    target[target_key] = value
