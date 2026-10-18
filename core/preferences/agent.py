from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from core.preferences.llms import get_default_preferences_model
from core.preferences.store import PreferencesStore
from core.preferences.tools import EMPTY_PREFERENCES, create_preferences_tools
from core.session.events import SessionEvent

PREFERENCES_AGENT_PROMPT = f"""Maintain PREFERENCES.md from explicit preferences stated by the user.

A preference is a reusable instruction about how the user wants development work performed. Capture explicit:
- communication and output style instructions, including commit-message style;
- engineering principles, implementation preferences, and quality expectations;
- workflows and procedural requirements for testing, reviewing, committing, or releasing;
- tool, environment, and runtime conventions;
- corrections that establish how future work should be handled.

Words such as `always`, `must`, `prefer`, `remember`, and `whenever` clearly signal persistence. One explicit statement is sufficient; repetition is not required. For example, `commit messages must always be rich` must become a concise preferences instruction such as `Write rich, descriptive commit messages.` By contrast, `commit these changes` is only a task for the current turn and must not be recorded.

USER events are the only valid evidence. Assistant messages, reasoning, tool calls, tool outputs, code changes, and repository state are context only. Do not infer preferences from assistant behavior, implementation choices, questions about the code, or temporary task details. Do not record secrets or repository facts.

Use inspect_session and read_preferences first. PREFERENCES.md is accumulated knowledge, not a document to regenerate from the current session. Preserve every existing instruction by default; it does not need to be restated or supported again in the current session. Merge new preferences into the existing document. Revise or remove an existing instruction only when a newer USER event explicitly changes, contradicts, or retracts it. Silence, omission, or lack of repeated evidence is never grounds for changing existing content.

Write the complete document as GitHub-flavored Markdown. Organize it with descriptive ATX headings and subheadings (`#` and `##`). Heading names and section organization are your choice; there is no fixed schema. Keep it concise and actionable, with no explanations or filler.

Every substantive update must provide update_preferences with exact user quotes and turn numbers. When new evidence exists, write the complete merged document and retain all unaffected existing sections. When the session adds no new preference and changes none, leave an existing PREFERENCES.md untouched. Only when PREFERENCES.md does not exist and no preference has been learned, create it with exactly:

```markdown
{EMPTY_PREFERENCES}
```

The user's current instructions and repository instructions always outrank PREFERENCES.md."""


def create_preferences_agent(
    events: tuple[SessionEvent, ...],
    store: PreferencesStore,
    *,
    model: BaseChatModel | None = None,
) -> Any:
    """Build the detached preferences-maintenance agent with only scoped tools."""

    return create_agent(
        model=model or get_default_preferences_model(),
        tools=create_preferences_tools(events, store),
        system_prompt=PREFERENCES_AGENT_PROMPT,
        name="preferences_agent",
    )
