from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from core.trajectory_bank.embeddings import LocalEmbeddings
from core.trajectory_bank.models import Hit, TrajectoryRecord


class TrajectoryBank:
    """Lazy embedded Chroma store. No server or embedding download at startup."""

    def __init__(self, cwd: Path, *, embedder: Any = None) -> None:
        self.root = cwd.expanduser().resolve() / ".quasipilot" / ".trajectorybank"
        self.embedder = embedder or LocalEmbeddings()
        self._collection = None
        self._lock = RLock()

    def _open(self, *, create: bool = False):
        if self._collection is not None:
            if not (self.root / "chroma.sqlite3").exists():
                raise RuntimeError("Trajectory bank was removed; restart the harness before recreating it")
            return self._collection
        if not create and not (self.root / "chroma.sqlite3").exists():
            return None
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(self.root), settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            "turn-trajectories-v1", embedding_function=None,
            metadata={"schema_version": 1, "embedding_identity": self.embedder.identity},
            configuration={"hnsw": {"space": "cosine"}},
        )
        if (collection.metadata.get("embedding_identity") != self.embedder.identity
                or collection.metadata.get("schema_version") != 1):
            raise ValueError("Trajectory bank embedding configuration differs; explicit reindexing is required")
        self._collection = collection
        return collection

    def archive(self, records: list[TrajectoryRecord]) -> None:
        if not records:
            return
        with self._lock:
            collection = self._open(create=True)
            existing = {record.id: record for record in self.get(ids=[record.id for record in records])}
            additions = []
            for record in records:
                previous = existing.get(record.id)
                if previous is not None:
                    if previous.document != record.document or any(
                        previous.metadata.get(key) != record.metadata.get(key)
                        for key in ("arc_id", "content_hash", "source_fingerprint")
                    ):
                        raise ValueError(f"Conflicting archived trajectory: {record.id}")
                else:
                    additions.append(record)
            for start in range(0, len(additions), 64):
                batch = additions[start:start + 64]
                collection.upsert(
                    ids=[record.id for record in batch], documents=[record.document for record in batch],
                    metadatas=[record.metadata for record in batch],
                    embeddings=self.embedder([record.document for record in batch]),
                )
            persisted = {record.id for record in self.get(ids=[record.id for record in records])}
            if persisted != {record.id for record in records}:
                raise RuntimeError("Trajectory archival verification failed")

    def get(self, *, ids: list[str] | None = None, where: dict | None = None) -> list[TrajectoryRecord]:
        with self._lock:
            collection = self._open()
            if collection is None or ids == []:
                return []
            kwargs = {"include": ["documents", "metadatas"]}
            if ids is not None:
                kwargs["ids"] = ids
            if where is not None:
                kwargs["where"] = where
            result = collection.get(**kwargs)
            return [TrajectoryRecord(id, doc, meta) for id, doc, meta in
                    zip(result["ids"], result["documents"], result["metadatas"], strict=True)]

    def search(self, query: str, *, where: dict | None, limit: int) -> list[Hit]:
        with self._lock:
            collection = self._open()
            if collection is None or collection.count() == 0:
                return []
            if where and not collection.get(where=where, limit=1, include=[])["ids"]:
                return []
            kwargs = {"where": where} if where else {}
            result = collection.query(
                query_embeddings=self.embedder([query]), n_results=min(limit, collection.count()),
                include=["documents", "metadatas", "distances"], **kwargs,
            )
            hits = [Hit(TrajectoryRecord(id, doc, meta), 1.0 - distance) for id, doc, meta, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True,
            )]
            return sorted(hits, key=lambda hit: (-(hit.similarity or 0), hit.record.id))
