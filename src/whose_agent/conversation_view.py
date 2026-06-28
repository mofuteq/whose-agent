from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from whose_agent.schemas import ConversationMessage


Speaker = Literal["principal", "agent", "tool", "system"]


@dataclass(frozen=True, slots=True)
class MessageView:
    turn_index: int
    message_id: str
    speaker: Speaker
    content: str


ConversationView: TypeAlias = tuple[MessageView, ...]


ROLE_TO_SPEAKER: dict[str, Speaker] = {
    "user": "principal",
    "assistant": "agent",
    "tool": "tool",
    "system": "system",
}


def project_messages(
    messages: Sequence[ConversationMessage],
) -> ConversationView:
    return tuple(
        MessageView(
            turn_index=index,
            message_id=message.message_id,
            speaker=ROLE_TO_SPEAKER[message.role],
            content=message.content,
        )
        for index, message in enumerate(messages, start=1)
    )


__all__ = [
    "ConversationView",
    "MessageView",
    "Speaker",
    "project_messages",
]
