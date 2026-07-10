from __future__ import annotations

import warnings
from types import SimpleNamespace

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from whose_agent.llm_result import extract_usage_details


def test_extract_usage_details_reads_pydantic_ai_usage_property() -> None:
    calls = {"property": 0, "call": 0}

    class CallableUsage(SimpleNamespace):
        def __call__(self) -> "CallableUsage":
            calls["call"] += 1
            warnings.warn(
                "AgentRunResult.usage is no longer a method.",
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            return self

    class FakeResult:
        @property
        def usage(self) -> CallableUsage:
            calls["property"] += 1
            return CallableUsage(
                input_tokens=13,
                output_tokens=7,
                total_tokens=20,
                cache_write_tokens=3,
                details={"provider_billable_tokens": 21, "ignored_zero": 0},
            )

    with warnings.catch_warnings():
        warnings.simplefilter("error", PydanticAIDeprecationWarning)
        usage = extract_usage_details(FakeResult())

    assert usage == {
        "input": 13,
        "output": 7,
        "total": 20,
        "cache_creation_input": 3,
        "provider_billable_tokens": 21,
    }
    assert calls == {"property": 1, "call": 0}


def test_extract_usage_details_handles_missing_usage() -> None:
    assert extract_usage_details(object()) == {}
