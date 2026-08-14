from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from core.dev_profile.llms import get_default_dev_profile_model
from core.dev_profile.store import DevProfileStore
from core.dev_profile.tools import create_dev_profile_tools
from core.session.events import SessionEvent

DEV_PROFILE_AGENT_PROMPT = """You autonomously maintain a developer-oriented profile for a coding-agent harness.

Use the provided tools to inspect the immutable full-fidelity session dump progressively and to read the existing DEVPROFILE.md. Decide what durable developer instructions, preferences, and corrections are worth preserving across future agent runs.

The profile is free-form Markdown. Evolve its organization and wording as needed; no fixed headings or schema are required. Preserve useful existing guidance, reconcile contradictions in favor of newer explicit evidence, remove stale or redundant guidance, and keep the document concise and actionable.

USER events are the only valid evidence for profile claims. Assistant messages, reasoning, tool calls, tool outputs, code changes, and repository state are context only and must never become profile guidance. Questions about how the current implementation works are not preferences. Do not infer durable preferences from a single incidental implementation choice. Do not record secrets, temporary task details, or repository facts. Distinguish one-off requests from persistent working preferences. The developer's current instructions and repository-level instructions always outrank the profile.

Start with inspect_session and read_devprofile. Read or search deeper only where user-authored evidence warrants it. Every substantive update must pass exact user quotes and turn numbers to update_devprofile. Remove existing claims that lack user-authored support. If no insightful developer preference is supported, write exactly `No dev preferences learned yet.` instead of inventing guidance. Finish with a concise status summary."""


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
