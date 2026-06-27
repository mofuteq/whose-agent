from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Speaker = Literal["principal", "agent", "tool", "system"]


class MessageHistoryError(ValueError):
    pass


class MessageView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=1)
    speaker: Speaker
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def require_content(cls, value: str) -> str:
        content = str(value).strip()
        if not content:
            raise ValueError("content must not be empty")
        return content


ROLE_TO_SPEAKER: dict[str, Speaker] = {
    "user": "principal",
    "assistant": "agent",
    "tool": "tool",
    "system": "system",
}


def load_message_history_file(path: Path) -> list[MessageView]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MessageHistoryError(f"could not read --messages-file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MessageHistoryError(
            f"--messages-file must be valid JSON: {exc.msg}"
        ) from exc
    return normalize_role_tagged_history(raw)


def normalize_role_tagged_history(records: object) -> list[MessageView]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise MessageHistoryError("--messages-file must contain a JSON array.")
    if not records:
        raise MessageHistoryError("--messages-file must contain at least one message.")

    views: list[MessageView] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise MessageHistoryError(
                f"message {index} must be an object with role and content."
            )
        role_value = record.get("role")
        content_value = record.get("content")
        if not isinstance(role_value, str):
            raise MessageHistoryError(f"message {index} role must be a string.")
        role = role_value.strip().casefold()
        speaker = ROLE_TO_SPEAKER.get(role)
        if speaker is None:
            raise MessageHistoryError(
                f"message {index} has unsupported role {role_value!r}."
            )
        if not isinstance(content_value, str):
            raise MessageHistoryError(f"message {index} content must be a string.")
        try:
            views.append(
                MessageView(
                    turn_index=index,
                    speaker=speaker,
                    content=content_value,
                )
            )
        except ValueError as exc:
            raise MessageHistoryError(f"message {index} is invalid: {exc}") from exc
    return views


def current_principal_prompt(views: Sequence[MessageView]) -> str:
    if not views:
        raise MessageHistoryError("--messages-file must contain at least one message.")
    final = views[-1]
    if final.speaker != "principal":
        raise MessageHistoryError(
            "--messages-file final message must have role 'user'."
        )
    return final.content


__all__ = [
    "MessageHistoryError",
    "MessageView",
    "Speaker",
    "current_principal_prompt",
    "load_message_history_file",
    "normalize_role_tagged_history",
]
