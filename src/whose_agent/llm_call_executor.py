from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar


T = TypeVar("T")


class LLMCallExecutor:
    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return func(*args, **kwargs)


class ThreadedLLMCallExecutor(LLMCallExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="whose-agent-llm",
        )

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return self._executor.submit(func, *args, **kwargs).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


__all__ = [
    "LLMCallExecutor",
    "ThreadedLLMCallExecutor",
]
