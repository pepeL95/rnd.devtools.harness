
# Maintenance Recipes

Run with the harness environment. Set `skill_dir`, `bank`, `sessions`, and any
output paths to inspected absolute paths; do not copy placeholders blindly.
No network is needed for audit/pruning. Re-indexing may download the configured
local model or call a hosted provider: confirm the model, data destination, and cost.

## Inspect

```sh
chroma browse turn-trajectories-v1 --local --path /absolute/workspace/.quasipilot/.trajectorybank
python /absolute/skill/scripts/bank_guard.py /absolute/workspace/.quasipilot/.trajectorybank --sessions /absolute/sessions
```

Check the installed `chroma --help`; the native CLI supplies browsing, copying,
server operation, and vacuuming, not a complete document-editing/re-indexing CLI.
Vacuum is storage maintenance, not semantic eviction or re-embedding.
Do not start a browser/server on the database during an offline mutation.

For SDK recipes below, load the bundled module without installing a new package:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(skill_dir / "scripts"))
from bank_guard import inspect_bank, backup, prune_unreferenced, reindex

report = inspect_bank(bank, sessions)
```

The audit lists record IDs, their arcs, publication, size, and integrity issues.
For full documents use `chromadb.PersistentClient(path=str(bank)).get_collection(
"turn-trajectories-v1", embedding_function=None).get(ids=reviewed_ids,
include=["documents", "metadatas"])`. Never use `get_or_create_collection` to
inspect an assumed existing bank. The audit digest is a reviewed-state precondition,
not evidence of human approval.

## Prune Confirmed Unreferenced Records

Review exact IDs from the audit. Close writers before taking a fresh audit and
approving deletion. The helper refuses referenced records, missing sessions,
pending compactions, mismatched state, and an existing backup destination.

```python
prune_unreferenced(
    bank, sessions, ids=reviewed_ids, expected_digest=report["digest"],
    backup_dir=Path("/absolute/backups/before-prune"), writers_stopped=True,
)
```

The helper preserves raw/curated sessions and verifies the remaining bank. It
does not provide published-record eviction or treat a stale source as permission
to discard a published record. Never delete raw source events to hide corruption.

## Re-index Without Changing Evidence

```python
candidate = Path("/absolute/staging/reindexed-bank")  # Must not exist.
reindex(bank, sessions, candidate, writers_stopped=True)
```

This uses the configured `get_default_embedding_model()` and
`DEFAULT_EMBEDDING_IDENTITY`, reads old documents even if the current configured
identity differs, and writes a fresh collection with unchanged records/metadata.
It checks hashes/references, vector count/dimensionality/finite values, and a query
probe. Errors leave the live bank untouched; any partial candidate is disposable,
not production-ready. Do not switch on the basis of record count alone.

Compare representative reasoning queries against the old and candidate banks.
Recalibrate thresholds for a different model; the current `0.72` is not portable.
Query vectors must come from the same model as each respective index.

After approval, stop all readers/writers, snapshot the old bank with `backup`,
exit any Python processes holding Chroma clients, and replace the live directory
with the validated candidate using an explicit reviewed filesystem operation.
Retain the old directory for rollback. Keep the harness embedding configuration
consistent with the activated candidate; changing config first may prevent normal
harness startup until the migration is complete. Rerun the audit and a harness
retrieval smoke test after restart. There is deliberately no automatic activation.

## Documents and Arc Reorganization

Make an offline snapshot using `backup(bank, sessions, output,
writers_stopped=True)`. It contains `bank/` and `sessions/`; source-path metadata
still points to the original raw files. Treat this as a restore backup, not a new
independent workspace. For migration experiments, explicitly rebase source paths
to its copied sessions before auditing the copy; never change the raw contents.

Use a task-specific SDK script for the approved migration. Preserve original
evidence, generate coherent replacement identities and vectors, and update affected
session task maps, manifests, and rendered memory together. Annotate authored
corrections honestly rather than presenting them as original trajectory synthesis.
An integrity audit cannot judge whether the rewritten memory is faithful: review
before/after documents and exercise a subsequent compaction on an isolated copy.
Do not activate a migration with unresolved references or failed validation.

## References

- [Chroma CLI](https://docs.trychroma.com/cli/install)
- [Chroma Python client](https://docs.trychroma.com/reference/python)
- [Collection and embedding migrations](https://cookbook.chromadb.dev/core/collections/)
- Installed harness: `core/trajectory_bank/{store,retrieval,evidence,defaults}.py`
  and `core/compaction/arcs.py`.
