from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from core.utilities.messages import system_message_with_appended_text

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_PROMPT = """\
[SKILLS]
You have access to optional skills that provide task-specific instructions.
If a task clearly matches a skill, call `read_skill` with the skill name before proceeding.
After reading a skill, resolve relative paths against the reported skill directory and use absolute paths in tool calls.

**Available skills:**
{skills}

[END SKILLS]
"""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path


class ReadSkillSchema(BaseModel):
    """Input schema for the `read_skill` tool."""

    name: str = Field(description="Exact skill name to load, for example `code-review`.")


class SkillsMiddleware(AgentMiddleware):
    """Expose installed skills through a compact system-prompt catalog."""

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        roots: Sequence[str | Path] | None = None,
        prompt: str = DEFAULT_SKILLS_PROMPT,
    ) -> None:
        self.cwd = Path(cwd).expanduser().resolve() if cwd else None
        self.roots = tuple(Path(root).expanduser() for root in roots) if roots else None
        self.prompt = prompt.strip()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        cwd = self._runtime_cwd(getattr(request, "runtime", None)) or self.cwd or Path.cwd()
        skills = discover_skills(Path(cwd), roots=self.roots)
        if not skills:
            return handler(request)
        updated = request.override(
            system_message=system_message_with_appended_text(
                request.system_message,
                self.prompt.format(skills=_render_skills(skills)),
            )
        )
        return handler(updated)

    @staticmethod
    def _runtime_cwd(runtime: Any) -> str | Path | None:
        context = getattr(runtime, "context", None)
        return getattr(context, "cwd", None) or (context.get("cwd") if isinstance(context, dict) else None)


def discover_skills(cwd: Path, *, roots: Sequence[Path] | None = None) -> list[SkillMetadata]:
    discovered: dict[str, SkillMetadata] = {}
    for root in _skill_roots(cwd, roots=roots):
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            skill_path = path / "SKILL.md"
            if not path.is_dir() or not skill_path.is_file():
                continue
            skill = _load_skill(skill_path)
            if skill is None:
                continue
            if skill.name in discovered:
                logger.warning("Skill '%s' from %s overrides %s", skill.name, skill.path, discovered[skill.name].path)
            discovered[skill.name] = skill
    return sorted(discovered.values(), key=lambda skill: skill.name)


def _skill_roots(cwd: Path, *, roots: Sequence[Path] | None = None) -> tuple[Path, ...]:
    if roots is not None:
        return tuple(root.expanduser().resolve() for root in roots)

    base = cwd.expanduser().resolve()
    return (
        Path("~/.agents/skills").expanduser().resolve(),
        Path("~/.quasipilot/skills").expanduser().resolve(),
        (base / ".agents/skills").resolve(),
        (base / ".quasipilot/skills").resolve(),
    )


def _load_skill(path: Path) -> SkillMetadata | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read skill %s: %s", path, exc)
        return None

    frontmatter = _frontmatter_block(text)
    if frontmatter is None:
        logger.warning("Skipping skill without valid frontmatter: %s", path)
        return None

    fields = _parse_frontmatter(frontmatter)
    name = fields.get("name", "").strip()
    description = fields.get("description", "").strip()
    if not name:
        logger.warning("Skipping skill without name: %s", path)
        return None
    if not description:
        logger.warning("Skipping skill without description: %s", path)
        return None
    if path.parent.name != name:
        logger.warning("Skill name '%s' does not match directory '%s' at %s", name, path.parent.name, path)
    if len(name) > 64:
        logger.warning("Skill name exceeds 64 characters at %s", path)
    return SkillMetadata(name=name, description=description, path=path.resolve())


def _frontmatter_block(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line[:1].isspace() or ":" not in line:
            index += 1
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {"|", ">"}:
            block: list[str] = []
            index += 1
            while index < len(lines):
                nested = lines[index]
                if nested[:1].isspace():
                    block.append(nested.strip())
                    index += 1
                    continue
                break
            fields[key] = "\n".join(block).strip()
            continue
        fields[key] = _strip_quotes(value)
        index += 1
    return fields


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value.strip()


def _render_skills(skills: Iterable[SkillMetadata]) -> str:
    return "\n".join(
        f"- `{skill.name}`: {skill.description} (`SKILL.md`: {skill.path})"
        for skill in skills
    )


def create_read_skill_tool(
    *,
    cwd: str | Path | None = None,
    roots: Sequence[str | Path] | None = None,
) -> BaseTool:
    base_cwd = Path(cwd).expanduser().resolve() if cwd else None
    skill_roots = tuple(Path(root).expanduser() for root in roots) if roots else None

    def read_skill(name: str, runtime: Any = None) -> str:
        """Load a discovered skill by exact name."""

        resolved_cwd = SkillsMiddleware._runtime_cwd(runtime) or base_cwd or Path.cwd()
        skills = discover_skills(Path(resolved_cwd), roots=skill_roots)
        skill = next((item for item in skills if item.name == name), None)
        if skill is None:
            available = ", ".join(item.name for item in skills) or "(none)"
            return f"Error: unknown skill '{name}'. Available skills: {available}"

        body = _skill_body(skill.path.read_text(encoding="utf-8"))
        resources = _list_skill_resources(skill.path.parent)
        resource_lines = "\n".join(f"- {resource}" for resource in resources) if resources else "- (none)"
        return "\n".join(
            [
                f"<skill name=\"{skill.name}\">",
                f"Skill file: {skill.path}",
                f"Skill directory: {skill.path.parent}",
                "Relative paths in this skill are relative to the skill directory.",
                "",
                body,
                "",
                "<skill_resources>",
                resource_lines,
                "</skill_resources>",
                "</skill>",
            ]
        ).strip()

    return StructuredTool.from_function(
        func=read_skill,
        name="read_skill",
        description="Load the full instructions for a discovered skill by exact name. Use this when a skill from the skills catalog matches the task.",
        args_schema=ReadSkillSchema,
        infer_schema=False,
    )


def _skill_body(text: str) -> str:
    frontmatter = _frontmatter_block(text)
    if frontmatter is None:
        return text.strip()
    lines = text.splitlines()
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :]).strip()
    return text.strip()


def _list_skill_resources(skill_dir: Path, *, limit: int = 64) -> list[str]:
    resources: list[str] = []
    for root_name in ("scripts", "references", "assets"):
        root = skill_dir / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resources.append(str(path.relative_to(skill_dir)))
            if len(resources) >= limit:
                return resources
    return resources
