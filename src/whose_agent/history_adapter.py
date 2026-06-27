from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from whose_agent.schemas import ConversationMessage, ConversationRole


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
SUPPORTED_ROLES: set[str] = {"user", "assistant", "tool", "system"}


def load_message_history_file(path: Path) -> list[ConversationMessage]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MessageHistoryError(f"could not read --messages-file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MessageHistoryError(
            f"--messages-file must be valid JSON: {exc.msg}"
        ) from exc
    return normalize_role_tagged_messages(raw)


def normalize_role_tagged_messages(records: object) -> list[ConversationMessage]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise MessageHistoryError("--messages-file must contain a JSON array.")
    if not records:
        raise MessageHistoryError("--messages-file must contain at least one message.")

    messages: list[ConversationMessage] = []
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
        if role not in SUPPORTED_ROLES:
            raise MessageHistoryError(
                f"message {index} has unsupported role {role_value!r}."
            )
        if not isinstance(content_value, str):
            raise MessageHistoryError(f"message {index} content must be a string.")
        try:
            messages.append(
                ConversationMessage(
                    role=role,
                    content=content_value,
                )
            )
        except ValueError as exc:
            raise MessageHistoryError(f"message {index} is invalid: {exc}") from exc
    return messages


def normalize_role_tagged_history(records: object) -> list[MessageView]:
    return project_message_views(normalize_role_tagged_messages(records))


def project_message_views(messages: Sequence[ConversationMessage]) -> list[MessageView]:
    views: list[MessageView] = []
    for index, message in enumerate(messages, start=1):
        views.append(
            MessageView(
                turn_index=index,
                speaker=ROLE_TO_SPEAKER[message.role],
                content=message.content,
            )
        )
    return views


def conversation_from_prompt(prompt: str) -> list[ConversationMessage]:
    return [ConversationMessage(role="user", content=prompt)]


def initial_conversation_messages(
    initial_records: Sequence[Mapping[str, object]],
    *,
    prompt: str,
) -> list[ConversationMessage]:
    if initial_records:
        return normalize_role_tagged_messages(initial_records)
    return conversation_from_prompt(prompt)


def append_assistant_message(
    messages: Sequence[ConversationMessage],
    content: str,
) -> list[ConversationMessage]:
    return [*messages, ConversationMessage(role="assistant", content=content)]


def current_principal_prompt(messages: Sequence[ConversationMessage]) -> str:
    if not messages:
        raise MessageHistoryError("--messages-file must contain at least one message.")
    final = messages[-1]
    if final.role != "user":
        raise MessageHistoryError(
            "--messages-file final message must have role 'user'."
        )
    return final.content


__all__ = [
    "MessageHistoryError",
    "MessageView",
    "Speaker",
    "append_assistant_message",
    "conversation_from_prompt",
    "current_principal_prompt",
    "initial_conversation_messages",
    "load_message_history_file",
    "normalize_role_tagged_history",
    "normalize_role_tagged_messages",
    "project_message_views",
]
