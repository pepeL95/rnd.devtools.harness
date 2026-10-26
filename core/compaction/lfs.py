from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from core.compaction.evidence import PreparedEvidence
from core.compaction.models import CompactionResult
from core.session.events import SessionEvent
from core.session.manager import SessionManager


class LfsEvidenceProvider:
    """Archive reviewed session arcs and their extracted task clusters as Markdown."""

    def __init__(self, manager: SessionManager) -> None:
        self.root = manager.dump_path.parent.parent / "memories" / manager.session_id

    def validate(self, manifest: dict[str, list[str]]) -> None:
        pass

    def prepare(
        self, events: list[SessionEvent], turns: list[int], episodes: list[str],
        manifest: dict[str, list[str]],
    ) -> PreparedEvidence:
        name = f"{uuid4()}.md"
        arc = self.root / "arcs" / name
        trajectory = self.root / "trajectories" / name

        def persist(result: CompactionResult) -> None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
            frontmatter = f'---\ndate: "{date}"\n---\n\n'
            created = []
            try:
                for path, body in ((arc, result.memory_document), (trajectory, result.segmentation)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("x", encoding="utf-8") as handle:
                        created.append(path)
                        handle.write(frontmatter + body + "\n")
            except Exception:
                # Never leave a half-written pair or delete a pre-existing archive.
                for path in created:
                    path.unlink(missing_ok=True)
                raise

        return PreparedEvidence(
            manifest=manifest,
            episode_references=[str(trajectory)] * len(episodes),
            persist=persist,
            references=[str(trajectory)],
        )
