"""Offline skill helpers over Chroma's SDK; no harness runtime integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from core.session.io import default_session_root, read_events
from core.trajectory_bank.evidence import content_hash, fingerprint
from core.trajectory_bank.models import TaskArc

COLLECTION = "turn-trajectories-v1"


def open_bank(path: Path):
    import chromadb
    from chromadb.config import Settings

    path = Path(path).expanduser().resolve()
    if not (path / "chroma.sqlite3").is_file():
        raise ValueError(f"No existing Chroma bank at {path}")
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION, embedding_function=None)
    metadata = collection.metadata or {}
    if metadata.get("schema_version") != 1 or not metadata.get("embedding_identity"):
        raise ValueError("Unsupported bank schema or missing embedding identity")
    return collection


def _records(collection) -> dict:
    result = collection.get(include=["documents", "metadatas"])
    return {id: {"document": doc, "metadata": meta} for id, doc, meta in
            zip(result["ids"], result["documents"], result["metadatas"], strict=True)}


def _scan(bank: Path, sessions: Path):
    bank, sessions = Path(bank).expanduser().resolve(), Path(sessions).expanduser().resolve()
    collection = open_bank(bank)
    records = _records(collection)
    issues, files, referenced, arcs = [], {}, {}, {}
    sources = {}
    session_ids = {item["metadata"]["session_id"] for item in records.values()}
    for session in sorted(session_ids):
        if not isinstance(session, str) or not re.fullmatch(r"[\w-]+", session):
            raise ValueError("Invalid archived session identity")
        paths = {kind: sessions / kind / f"{session}.jsonl" for kind in ("dump", "curated")}
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"Missing session file; verify --sessions before maintenance: {path}")
            files[path] = content_hash(path.read_text(encoding="utf-8"))
        pending = sessions / "pending_trajectories" / f"{session}.json"
        if pending.exists():
            issues.append(f"{session}: pending trajectory compaction; resolve it before maintenance")
            files[pending] = content_hash(pending.read_text(encoding="utf-8"))
        sources[session] = {event.id: event for event in read_events(paths["dump"])}
        arcs[session] = {}
        for event in read_events(paths["curated"]):
            payload = event.payload
            tasks = {arc.arc_id: arc for arc in
                     (TaskArc.model_validate(item) for item in payload.get("task_arcs", []))}
            arcs[session].update({id: arc.model_dump() for id, arc in tasks.items()})
            for arc, ids in payload.get("trajectory_manifest", {}).items():
                if not isinstance(ids, list) or not all(isinstance(id, str) for id in ids):
                    raise ValueError(f"Malformed manifest in {paths['curated']}")
                if "task_arcs" in payload and arc not in tasks:
                    issues.append(f"{session}/{arc}: manifest has no task arc")
                for id in ids:
                    referenced.setdefault(id, set()).add((session, arc))
                    item = records.get(id)
                    if item is None:
                        issues.append(f"{id}: published record missing from bank")
                    elif arc in tasks and item["metadata"]["turn"] not in tasks[arc].turns:
                        issues.append(f"{id}: turn absent from task arc")
    rows = []
    for id, item in sorted(records.items()):
        meta, document = item["metadata"], item["document"]
        session, arc = meta["session_id"], meta["arc_id"]
        if meta.get("schema_version") != 1:
            issues.append(f"{id}: unsupported record schema")
        if not isinstance(document, str) or content_hash(document) != meta.get("content_hash"):
            issues.append(f"{id}: document content hash mismatch")
        ids = json.loads(meta["source_event_ids"])
        if not isinstance(ids, list) or not ids or not all(isinstance(key, str) for key in ids):
            raise ValueError(f"{id}: malformed source_event_ids")
        events = [sources[session][key] for key in ids if key in sources[session]]
        if len(events) != len(ids) or fingerprint(events) != meta.get("source_fingerprint"):
            issues.append(f"{id}: raw source evidence missing or changed")
        source_turns = json.loads(meta["source_turns"]) if "source_turns" in meta else [meta["turn"]]
        if {event.turn for event in events} != set(source_turns):
            issues.append(f"{id}: raw evidence belongs to different source turns")
        if meta.get("source_path") and Path(meta["source_path"]).expanduser().resolve() != sessions / "dump" / f"{session}.jsonl":
            issues.append(f"{id}: source_path differs from the supplied sessions root")
        if referenced.get(id, set()) - {(session, arc)}:
            issues.append(f"{id}: publication disagrees with record session/arc")
        rows.append({"id": id, "session": session, "arc": arc, "turn": meta["turn"],
                     "published": id in referenced, "characters": len(document or "")})
    state = {"bank": str(bank), "sessions": str(sessions), "metadata": collection.metadata,
             "records": records, "files": {str(path): digest for path, digest in files.items()}}
    digest = content_hash(json.dumps(state, sort_keys=True, ensure_ascii=False))
    report = {"bank": str(bank), "sessions": str(sessions), "embedding_identity": collection.metadata["embedding_identity"],
              "digest": digest, "records": rows, "arcs": arcs, "issues": issues}
    return collection, records, files, report


def inspect_bank(bank: Path, sessions: Path | None = None) -> dict:
    """Audit without loading an embedding model or creating absent banks/sessions."""
    return _scan(bank, sessions or default_session_root())[3]


def _require_offline(writers_stopped: bool) -> None:
    if not writers_stopped:
        raise ValueError("Close harness/Chroma writers and explicitly assert writers_stopped=True")


def _require_clean(report: dict) -> None:
    if report["issues"]:
        raise ValueError("Bank audit failed: " + "; ".join(report["issues"]))


def _new_output(output: Path, bank: Path, sessions: Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() or any(output.is_relative_to(Path(root).expanduser().resolve())
                              for root in (bank, sessions)):
        raise ValueError("Output must be a new directory outside the bank and sessions root")
    return output


def _file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def backup(bank: Path, sessions: Path, output: Path, *, writers_stopped: bool = False) -> Path:
    """Snapshot the bank and its associated sessions, retaining the original source paths."""
    _require_offline(writers_stopped)
    output = _new_output(output, bank, sessions)
    _, _, files, report = _scan(bank, sessions)
    bank_hashes = _file_hashes(Path(report["bank"]))
    output.mkdir(parents=True, mode=0o700)
    shutil.copytree(report["bank"], output / "bank")
    for path in files:
        target = output / "sessions" / path.relative_to(report["sessions"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if inspect_bank(bank, sessions)["digest"] != report["digest"]:
        raise ValueError(f"State changed during backup; do not use incomplete snapshot {output}")
    if _file_hashes(output / "bank") != bank_hashes or _file_hashes(Path(report["bank"])) != bank_hashes:
        raise ValueError("Bank files changed during backup or snapshot verification failed")
    for path, digest in files.items():
        copied = output / "sessions" / path.relative_to(report["sessions"])
        if content_hash(copied.read_text(encoding="utf-8")) != digest:
            raise ValueError("Session backup verification failed")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def prune_unreferenced(bank: Path, sessions: Path, *, ids: list[str], expected_digest: str,
                       backup_dir: Path, writers_stopped: bool = False) -> dict:
    """Delete only explicitly reviewed, unpublished records from a clean, unchanged bank."""
    _require_offline(writers_stopped)
    collection, records, _, report = _scan(bank, sessions)
    _require_clean(report)
    if report["digest"] != expected_digest:
        raise ValueError("Bank/session state differs from the reviewed audit")
    eligible = {row["id"] for row in report["records"] if not row["published"]}
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= eligible:
        raise ValueError("Select distinct existing unreferenced IDs; published records cannot be pruned")
    backup(bank, sessions, backup_dir, writers_stopped=True)
    if inspect_bank(bank, sessions)["digest"] != expected_digest:
        raise ValueError("State changed after backup; refusing deletion")
    collection.delete(ids=ids)
    remaining = _records(collection)
    if remaining != {id: record for id, record in records.items() if id not in ids}:
        raise RuntimeError(f"Prune verification failed; restore from {backup_dir}")
    result = inspect_bank(bank, sessions)
    _require_clean(result)
    return result


def _vectors(vectors, count: int, dimension: int | None) -> int:
    if len(vectors) != count:
        raise ValueError("Embedding count does not match documents")
    for vector in vectors:
        if dimension is None:
            dimension = len(vector)
        if not dimension or len(vector) != dimension or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError("Embeddings must have consistent nonzero dimensions and finite values")
    return dimension


def reindex(bank: Path, sessions: Path, output: Path, *, writers_stopped: bool = False,
            embedder=None, embedding_identity: str | None = None) -> Path:
    """Build and verify a candidate index; never activate it or mutate source documents."""
    import chromadb
    from chromadb.config import Settings
    from core.trajectory_bank import defaults

    _require_offline(writers_stopped)
    output = _new_output(output, bank, sessions)
    collection, records, _, report = _scan(bank, sessions)
    _require_clean(report)
    if (embedder is None) != (embedding_identity is None):
        raise ValueError("Supply both embedder and embedding_identity, or neither")
    if embedder is None:
        embedder, embedding_identity = defaults.get_default_embedding_model(), defaults.DEFAULT_EMBEDDING_IDENTITY
    if not isinstance(embedding_identity, str) or not embedding_identity.strip():
        raise ValueError("Embedding identity must be nonempty")
    output.mkdir(parents=True, mode=0o700)
    client = chromadb.PersistentClient(path=str(output), settings=Settings(anonymized_telemetry=False))
    candidate = client.create_collection(COLLECTION, embedding_function=None,
                                        metadata={**collection.metadata, "embedding_identity": embedding_identity},
                                        configuration={"hnsw": {"space": "cosine"}})
    ids = sorted(records)
    dimension = None
    for start in range(0, len(ids), 64):
        batch = ids[start:start + 64]
        docs = [records[id]["document"] for id in batch]
        vectors = embedder.embed_documents(docs)
        dimension = _vectors(vectors, len(batch), dimension)
        candidate.add(ids=batch, documents=docs, embeddings=vectors,
                      metadatas=[records[id]["metadata"] for id in batch])
    if _records(candidate) != records:
        raise RuntimeError("Re-index verification failed: records changed")
    if ids:
        query = embedder.embed_query(records[ids[0]]["document"])
        _vectors([query], 1, dimension)
        if not candidate.query(query_embeddings=[query], n_results=1)["ids"][0]:
            raise RuntimeError("Re-index query probe returned no records")
    if inspect_bank(bank, sessions)["digest"] != report["digest"]:
        raise ValueError("Source changed during re-index; candidate is not safe to activate")
    _require_clean(inspect_bank(output, sessions))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit an existing trajectory bank; never mutate records.")
    parser.add_argument("bank", type=Path)
    parser.add_argument("--sessions", type=Path, default=default_session_root())
    args = parser.parse_args()
    try:
        report = inspect_bank(args.bank, args.sessions)
        print(json.dumps(report, indent=2))
        raise SystemExit(1 if report["issues"] else 0)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
