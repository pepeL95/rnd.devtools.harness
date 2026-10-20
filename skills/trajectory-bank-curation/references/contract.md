# Bank and Session Contract

The skill requires an installed Quasipilot harness and Chroma. It imports the
harness's event parser and fingerprint implementation rather than duplicating them.
These rules describe schema version 1; inspect implementation changes before
migrating a different version. The guard rejects unsupported collection schemas.

## Storage and Identity

- Bank: nearest workspace `.quasipilot/.trajectorybank`, collection
  `turn-trajectories-v1`, cosine distance, explicit application-supplied vectors.
- Collection metadata: `schema_version = 1`, `embedding_identity`. Do not relabel
  old vectors with a new identity. Use `core.trajectory_bank.defaults` for both
  the LangChain embedding factory and its identity.
- Documents: normally `User -> Trajectory -> Assistant`, then keywords. Older
  banks may contain different bodies. Re-indexing preserves documents verbatim;
  upgrading their construction is a separate migration.
- Record metadata: `session_id`, `arc_id`, `turn`, timestamps, `source_kind`,
  `source_event_ids` (JSON string), `source_path`, `source_fingerprint`,
  `content_hash`, `keywords` (JSON string). Preserve additional metadata.
- Record IDs include session, arc, turn, and source/content hash fragments.
  Treat IDs and historical documents as immutable during ordinary maintenance.

## Arcs and Publication

Chroma contains turn documents grouped by `arc_id`. New groups identify compaction
batches (`batch-...`), not model-inferred tasks. Historical task groups remain valid.
Session `curated/<session>.jsonl` memory-restore events contain prose `session_arc`
and `trajectory_manifest` mapping group IDs to record IDs. Older sessions may also
contain structured `task_arcs`; the guard checks those only when present. Rendered
memory contains evidence references. Raw sources are in `dump/<session>.jsonl`.

Retrieval requires publication plus matching raw source IDs/fingerprint. The
fingerprint excludes pinned SYSTEM.md/AGENTS.md instructions. It does not prove
that an arbitrarily rewritten document is faithful to those sources.

Compaction requires all previously published record IDs to exist. Therefore:

- An unreferenced record is not necessarily garbage: it may belong to a pending
  or in-progress compaction. Stop writers and check pending trajectories first.
- Published eviction must coordinate the manifest, task map, rendered session
  memory, and any surviving references. It is not a standalone Chroma delete.
- Arc merge/split/reassignment requires a reviewed migration, including record
  identity and reference changes. Do not just overwrite `arc_id` metadata.
- A new `muted`, `retired`, or `superseded_by` metadata field is only an annotation
  unless retrieval/session rendering implements its semantics. This skill adds
  no such harness behavior.

## Guard Coverage and Limits

The guard checks content hashes, source fingerprints, source paths, session/arc
publication, task-map membership, missing references, and pending trajectory files
for sessions represented in the bank. Missing sessions fail closed. It does not
validate prose fidelity, discover unknown session roots, detect every external
writer, find an entirely deleted session's records without a prior inventory,
or prove the semantic quality of embeddings. `writers_stopped=True` is an
operator assertion, not a cross-process lock honored by the harness.

Use the explicit sessions root that produced these records. Snapshots copy the
bank and the associated dump/curated files, and contain private conversation data;
keep them local. No automatic backup expiry. Removing a Chroma record does not
erase session copies, exports, or backups.
