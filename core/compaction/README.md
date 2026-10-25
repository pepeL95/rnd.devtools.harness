# Session Arcs

Session compaction runs one pipeline: task-episode extraction, cumulative session-arc
synthesis, and bounded semantic review/revision. Extraction and synthesis use the
configured compactor model. Review uses the critic model and sees the original
evidence, previous arc, and extracted episodes. Prompt text lives in `prompts.py`.

The pipeline runs independently of evidence storage. `Compactor` has no dependency
on an archive backend or retrieval middleware. Without an evidence provider, it
still extracts the same rich episodes and generates the same session arc using
the previous memory, evicted conversation, and retained-tail orientation.

## Optional Evidence Provider

`core.compaction.evidence.CompactionEvidenceProvider` is the storage boundary:

1. `validate(manifest)` checks prior published references before any model calls.
2. `prepare(events, turns, episodes, manifest)` prepares references after extraction.
   Its `PreparedEvidence` supplies the updated manifest and one reference per episode.
3. `PreparedEvidence.persist()` runs after arc review and before the coordinator
   publishes the rewritten session. A storage failure leaves the session intact.

The provider owns storage and never rewrites the arc or adds model calls. Session
compaction owns source selection, prompts, review, and reference bookkeeping.
The historical `trajectory_manifest` field and reference URI formats remain stable
for existing sessions, including when no provider is installed. Merely retaining
an index does not open storage or retrieve its records.

The coordinator rejects stale source snapshots and preserves pinned instructions
and newer turns. CLI status compares estimated replaced context with injected
memory. Structured telemetry keeps that delta separate from provider-reported
model input/output usage.
