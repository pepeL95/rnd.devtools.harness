"""Offline retrieval smoke/calibration set; first use downloads the local embedding model."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.trajectory_bank import defaults
from core.trajectory_bank.policy import BankPolicy
from evals.common import ensure_dir, timestamp_slug, write_json


DOCUMENTS = {
    "edit": "edit_file failed because the source contains tabs, not spaces. Read the exact text before replacing it. "
            "Keep exact whitespace matching; do not implement fuzzy edits. Added escaped-whitespace diagnostics in filesystem.py.",
    "cli": "The CLI duplicates model thoughts from the previous turn. Clear the stream's per-run reasoning buffer when "
           "a new run starts. This concerns UI state, not the reasoning tool or persisted trajectory memory.",
    "cancel": "Cancellation now cancels the async model request. Escape pauses the active run without closing its semantic "
              "turn. Resume the same turn with the next user message; record an assistant introspection and a muted UI event.",
    "instructions": "SYSTEM.md and cwd AGENTS.md are reread each turn. Store only the latest system instructions in both "
                    "raw and curated sessions. Preserve them across compaction; do not retain revision history or load nested files.",
    "hooks": "Parallel tool hooks deduplicate steering by hook ID while retaining every originating tool_call_id. "
             "Current before.tool steering is delivered after the tool result, not as a pre-execution intercept.",
    "archive": "At session compaction archive original turn syntheses in the local Chroma trajectory bank. Group records "
               "by compaction batch, preserve exact references, and publish the session rewrite only after archive verification.",
}

# Includes close-topic negatives; an empty label set means abstain, not nearest-neighbor guessing.
QUERIES = [
    ("edit", "edit_file failed because the source contains tabs, not spaces. I should read the exact text before replacing it."),
    ("edit", "I need to fix whitespace replacement errors without introducing fuzzy matching."),
    ("cli", "The CLI duplicates model thoughts from the previous turn. I should clear the per-run reasoning buffer."),
    ("cli", "Old reasoning is appearing twice in the UI; I should inspect the stream state reset."),
    ("cancel", "Escape pauses the active run without closing its semantic turn. I should cancel the async model request."),
    ("cancel", "How should I resume a user-interrupted run without creating another turn?"),
    ("instructions", "SYSTEM.md and cwd AGENTS.md are reread each turn. Store only the latest system instructions in both raw and curated sessions."),
    ("instructions", "I should preserve current agent instructions through compaction without a revision log."),
    ("hooks", "Parallel tool hooks deduplicate steering by hook ID while retaining every originating tool_call_id."),
    ("archive", "At session compaction archive original turn syntheses in the local Chroma trajectory bank by compaction batch."),
    (None, "I need to adjust CSS tab spacing in the browser navigation bar."),
    (None, "I should cancel a PostgreSQL query that has been running too long."),
    (None, "I need to tune Chroma HNSW index ef_search for a million product catalog embeddings."),
    (None, "I should compact a SQLite database using VACUUM to reclaim disk space."),
    (None, "I should add React useEffect hooks to load account settings."),
    (None, "I need to rotate AWS credentials and configure an S3 bucket lifecycle policy."),
]


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/trajectory_bank_retrieval"))
    parser.add_argument("--live-compaction", action="store_true", help="Also exercise the configured LLM and real archive on synthetic turns")
    args = parser.parse_args()
    run_dir = ensure_dir(args.out_dir / timestamp_slug())
    embed = defaults.get_default_embedding_model()
    policy = BankPolicy()
    names = list(DOCUMENTS)
    documents = np.array(embed.embed_documents(list(DOCUMENTS.values())), dtype=float)
    queries = np.array([embed.embed_query(query) for _, query in QUERIES], dtype=float)
    for vectors in (documents, queries):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Embedding model returned invalid vectors")
        vectors /= norms
    rows = []
    false_positives = accepted = 0
    for (expected, query), scores in zip(QUERIES, queries @ documents.T, strict=True):
        ranked = sorted(zip(names, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        automatic = [name for name, score in ranked if score >= policy.automatic_min_similarity][:policy.automatic_results]
        false_positives += sum(name != expected for name in automatic)
        accepted += int(expected is not None and expected in automatic)
        rows.append({"query": query, "expected": expected, "best": ranked[0][0],
                     "similarity": round(float(ranked[0][1]), 4), "automatic": automatic,
                     "manual": [name for name, score in ranked if score >= policy.search_min_similarity][:policy.search_results]})
    report = {"embedding": defaults.DEFAULT_EMBEDDING_IDENTITY, "policy": policy.model_dump(), "cases": len(rows),
              "positive_cases": sum(expected is not None for expected, _ in QUERIES),
              "automatic_positive_cases": accepted, "automatic_false_positives": false_positives, "results": rows}
    if args.live_compaction:
        report["live_compaction"] = live_compaction(run_dir)
    write_json(run_dir / "metrics.json", report)
    print(json.dumps(report, indent=2))
    if false_positives or not accepted:
        raise SystemExit("Retrieval smoke gate failed; inspect scores before changing thresholds")


def live_compaction(run_dir: Path) -> dict:
    from core.compaction.compactor import Compactor
    from core.compaction.policy import CompactionPolicy
    from core.trajectory_bank.offloading import ChromaCompactionOffloadingStrategy
    from core.session.events import EventType, SessionEvent
    from core.session.manager import SessionManager
    from core.trajectory_bank.models import record_ref
    from core.trajectory_bank.retrieval import BankRetrieval
    from core.trajectory_bank.store import TrajectoryBank
    from evals.common import write_jsonl

    manager = SessionManager(session_id="synthetic-eval", root=run_dir / "sessions")
    bank = TrajectoryBank(run_dir)
    for turn, (topic, document) in enumerate(DOCUMENTS.items(), 1):
        manager.append([
            SessionEvent(type=EventType.USER, turn=turn, payload={"role": "user", "content": f"Work on {topic}."}),
            SessionEvent(type=EventType.REASONING, turn=turn, payload={"role": "assistant", "kind": "trajectory_memory", "content": document}),
            SessionEvent(type=EventType.ASSISTANT, turn=turn, payload={"role": "assistant", "content": "Changes described above; tests not run."}),
            SessionEvent(type=EventType.TURN_END, turn=turn, payload={}),
        ])
    manager.append([SessionEvent(type=EventType.USER, turn=7, payload={"role": "user", "content": "Continue investigating recall."})])
    source = manager.read_curated()
    result = Compactor(policy=CompactionPolicy(keep_last_turns=1), session_manager=manager,
                       offloading=ChromaCompactionOffloadingStrategy(bank, manager)).compact(source)
    manager.apply_compaction_result(result.events, snapshot_latest_turn=7, source_event_ids=[event.id for event in source])
    reopened = TrajectoryBank(run_dir)
    records = reopened.get()
    assert records and all(item.metadata["source_kind"] == "task_episode" for item in records)
    retrieval = BankRetrieval(reopened, manager)
    for item in records:
        assert retrieval.retrieve(ref=record_ref(item.id))[1] == [item.id]
    assert manager.read_dump() == source
    write_jsonl(run_dir / "source.jsonl", source)
    write_jsonl(run_dir / "rewritten.jsonl", result.events)
    (run_dir / "memory.md").write_text(result.memory_document, encoding="utf-8")
    return {"records": len(records), "groups": len(result.events[0].payload["trajectory_manifest"]),
            "model_names": result.model_names, "token_usage": result.token_usage, "revisions": result.revisions}


if __name__ == "__main__":
    main()
