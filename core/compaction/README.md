# Session Arcs

Session compaction runs one pipeline: task-episode extraction, cumulative session-arc
synthesis, and bounded semantic review/revision. Extraction and synthesis use the
configured compactor model. Review uses the critic model and sees the original
evidence, previous arc, and extracted episodes. Prompt text lives in `prompts.py`.

The pipeline runs independently of evidence storage. `Compactor` has no dependency
on an archive backend or retrieval middleware. Without an offloading strategy, it
still extracts the same rich episodes and generates the same session arc using
the previous memory, evicted conversation, and retained-tail orientation.

## Offloading Strategies

Subclass `core.compaction.offloading.BaseCompactionOffloadingStrategy` to implement
a storage destination. Its lifecycle is:

1. `validate(manifest)` optionally checks prior published references before any model calls.
2. `prepare(events, turns, episodes, manifest)` prepares references after extraction.
   Its `PreparedEvidence` supplies the updated manifest and one reference per episode.
3. `PreparedEvidence.persist(result)` runs after arc review and before the coordinator
   publishes the rewritten session. A storage failure leaves the session intact.

The strategy owns storage and never rewrites the arc or adds model calls. Session
compaction owns source selection, prompts, review, and reference bookkeeping.
The historical `trajectory_manifest` field and reference URI formats remain stable
for existing sessions, including when no strategy is installed. Merely retaining
an index does not open storage or retrieve its records.

`Compactor` accepts one strategy or an ordered sequence:

```python
from core.compaction.compactor import Compactor
from core.compaction.lfs import LfsCompactionOffloadingStrategy

compactor = Compactor(session_manager=manager, offloading=LfsCompactionOffloadingStrategy(manager))
# custom_strategy is another BaseCompactionOffloadingStrategy implementation.
compactor = Compactor(session_manager=manager, offloading=[
    LfsCompactionOffloadingStrategy(manager),
    custom_strategy,
])
```

Sequences are composed by `CompactionOffloadingStack`. All validation runs before
extraction. Each destination prepares against the same source and prior manifest;
the stack unions record IDs and deduplicates references in declared order, without
discarding older IDs. Multiple references for an episode are newline-separated in
the existing prompt field. Every persistence callback receives the same final
`CompactionResult`, after review. Strategies must not mutate sources or results;
per-run storage state belongs in the returned callback, not shared instance fields.

Persistence is ordered and fail-fast: a failure prevents session publication and
skips later destinations. Earlier successful writes can remain unreferenced; there
is no cross-backend rollback. Storage strategies must handle their own write cleanup.
The shared `build_compaction_offloading` factory resolves CLI and driver config.
The default is `offloading = ["lfs"]`; `[]` disables new offloading.
Scalar `"lfs"` and `"none"` remain supported for existing configurations.
Unknown or duplicate configured destinations are rejected. Chroma is
not implemented on main; its feature-branch adapter should inherit this base and
return a `persist(result)` callback. Retrieval middleware remains independent.
For Python callers, `offloading=` replaces the former `evidence_provider=` argument;
`LfsCompactionOffloadingStrategy` replaces `LfsEvidenceProvider`.

## Local Files

The CLI and driver default to `LfsCompactionOffloadingStrategy`. It stores `result.memory_document`
and `result.segmentation` verbatim beneath the session root's `memories/<session-id>/`,
in `arcs/<uuid>.md` and `trajectories/<uuid>.md`. A shared UTC creation date is added
as YAML frontmatter. Exclusive creation prevents overwrites; a write failure removes
the newly created partial pair. A process crash can still leave partial files, and a
subsequent stale-snapshot rejection can leave an unreferenced pair. Neither replaces
the live session. There is no automatic deletion or indexing.

`PreparedEvidence.references` supplies durable file references, carried forward in
memory-event bookkeeping and appended to the arc without relying on model copying.
Every episode in a batch references its shared trajectory file. No raw tool output
is archived separately, no additional model calls are made, and the generation and
review prompts are unchanged. Set `[compaction] offloading = []` to opt out.

The coordinator rejects stale source snapshots and preserves pinned instructions
and newer turns. CLI status compares estimated replaced context with injected
memory. Structured telemetry keeps that delta separate from provider-reported
model input/output usage.
