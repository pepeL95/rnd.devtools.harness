from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from core.dev_profile.llms import get_default_dev_profile_model
from core.dev_profile.store import DevProfileStore
from core.dev_profile.tools import EMPTY_DEV_PROFILE, create_dev_profile_tools
from core.session.events import SessionEvent

DEV_PROFILE_AGENT_PROMPT = f"""Maintain DEVPROFILE.md from explicit preferences stated by the user.

A preference is a reusable instruction about how the user wants development work performed. Capture explicit:
- communication and output style instructions, including commit-message style;
- engineering principles, implementation preferences, and quality expectations;
- workflows and procedural requirements for testing, reviewing, committing, or releasing;
- tool, environment, and runtime conventions;
- corrections that establish how future work should be handled.

Words such as `always`, `must`, `prefer`, `remember`, and `whenever` clearly signal persistence. One explicit statement is sufficient; repetition is not required. For example, `commit messages must always be rich` must become a concise profile instruction such as `Write rich, descriptive commit messages.` By contrast, `commit these changes` is only a task for the current turn and must not be recorded.

USER events are the only valid evidence. Assistant messages, reasoning, tool calls, tool outputs, code changes, and repository state are context only. Do not infer preferences from assistant behavior, implementation choices, questions about the code, or temporary task details. Do not record secrets or repository facts.

Use inspect_session and read_devprofile first. Inspect additional turns when needed. Preserve supported existing preferences, incorporate new ones, resolve contradictions in favor of newer explicit user statements, and remove unsupported claims.

Write the complete document as GitHub-flavored Markdown. Organize it with descriptive ATX headings and subheadings (`#` and `##`). Heading names and section organization are your choice; there is no fixed schema. Keep it concise and actionable, with no explanations or filler.

Every substantive update must provide update_devprofile with exact user quotes and turn numbers. If at least one explicit reusable preference exists, update the profile. Only when none exists, write exactly:

```markdown
{EMPTY_DEV_PROFILE}
```

The user's current instructions and repository instructions always outrank DEVPROFILE.md."""


def create_dev_profile_agent(
    events: tuple[SessionEvent, ...],
    store: DevProfileStore,
    *,
    model: BaseChatModel | None = None,
) -> Any:
    """Build the detached profile-maintenance agent with only scoped tools."""

    return create_agent(
        model=model or get_default_dev_profile_model(),
        tools=create_dev_profile_tools(events, store),
        system_prompt=DEV_PROFILE_AGENT_PROMPT,
        name="dev_profile_agent",
    )
