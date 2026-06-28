from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from whose_agent.schemas import ConversationMessage


class MessageHistoryError(ValueError):
    pass


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
        message_id_value = record.get("message_id")
        if message_id_value is not None and not isinstance(message_id_value, str):
            raise MessageHistoryError(f"message {index} message_id must be a string.")
        message_payload = {
            "role": role,
            "content": content_value,
        }
        if message_id_value is not None:
            message_payload["message_id"] = message_id_value
        try:
            messages.append(ConversationMessage(**message_payload))
        except ValueError as exc:
            raise MessageHistoryError(f"message {index} is invalid: {exc}") from exc
    require_unique_message_ids(messages)
    return messages


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
    existing_ids = {message.message_id for message in messages}
    while True:
        message = ConversationMessage(
            message_id=f"msg_{uuid4().hex}",
            role="assistant",
            content=content,
        )
        if message.message_id not in existing_ids:
            return [*messages, message]


def current_principal_prompt(messages: Sequence[ConversationMessage]) -> str:
    if not messages:
        raise MessageHistoryError("--messages-file must contain at least one message.")
    final = messages[-1]
    if final.role != "user":
        raise MessageHistoryError(
            "--messages-file final message must have role 'user'."
        )
    return final.content


def require_unique_message_ids(messages: Sequence[ConversationMessage]) -> None:
    seen: set[str] = set()
    for index, message in enumerate(messages, start=1):
        if message.message_id in seen:
            raise MessageHistoryError(
                f"duplicate message_id {message.message_id!r} at message {index}."
            )
        seen.add(message.message_id)


__all__ = [
    "MessageHistoryError",
    "append_assistant_message",
    "conversation_from_prompt",
    "current_principal_prompt",
    "initial_conversation_messages",
    "load_message_history_file",
    "normalize_role_tagged_messages",
    "require_unique_message_ids",
]
