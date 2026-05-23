from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    TURN_BEGIN = "turn_begin"
    TURN_END = "turn_end"
    RUNTIME = "runtime"
    META = "meta"
    USER = "user"
    REASONING = "reasoning"
    TOOL = "tool"
    TOOL_OUTPUT = "tool_output"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class RuntimeSnapshot:
    cwd: str
    git_branch: str | None = None
    git_dirty: bool | None = None

    def to_prompt_block(self) -> str:
        branch = self.git_branch or "unknown"
        dirty = "unknown" if self.git_dirty is None else str(self.git_dirty).lower()
        return "\n".join(
            [
                "[RUNTIME CONTEXT]",
                f"cwd: {self.cwd}",
                f"git_branch: {branch}",
                f"git_dirty: {dirty}",
                "[END RUNTIME CONTEXT]",
            ]
        )


@dataclass(frozen=True)
class SessionEvent:
    type: EventType
    turn: int
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: uuid4().hex)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "turn": self.turn,
            "payload": self.payload,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SessionEvent":
        return cls(
            id=str(data["id"]),
            timestamp=str(data["timestamp"]),
            type=EventType(str(data["type"])),
            turn=int(data["turn"]),
            payload=dict(data.get("payload") or {}),
        )
