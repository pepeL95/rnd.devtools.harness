"""Session lifecycle and persistence contracts."""

from core.session.events import EventType, RuntimeSnapshot, SessionEvent
from core.session.manager import SessionManager

__all__ = ["EventType", "RuntimeSnapshot", "SessionEvent", "SessionManager"]
