---
name: trajectory-bank-curation
description: Inspect and curate Quasipilot trajectory banks with Chroma, including re-indexing, eviction, provenance checks, and session task-arc maintenance.
---

# Trajectory Bank Curation

Use Chroma's CLI for browsing and its Python SDK for data operations. This skill
is a maintenance workspace, not harness middleware or an autonomous forgetting policy.
Make semantic decisions with the engineer; use the bundled guards for integrity.

## Start Here

1. Use the Python interpreter from the harness runtime (`quasipilot-harness` in
   this repository). Resolve the nearest `.quasipilot/.trajectorybank` from the
   target cwd with `core.utilities.workspace.workspace_directory`. Do not assume
   the skill directory is the workspace. Sessions normally live separately at
   `~/.quasipilot/sessions`; custom roots must be supplied explicitly.
2. Read [the data contract](references/contract.md) before proposing changes.
   Run `python <skill-dir>/scripts/bank_guard.py <bank-path> --sessions <sessions-root>`.
   This audits without embedding calls or creating a missing bank. Nonzero exit
   means integrity issues require investigation, not automatic deletion.
3. Inspect the relevant documents and arcs, then agree on exact targets and intent.
   Prefer preserving useful failures and constraints over age-based deletion.
4. Read [maintenance recipes](references/maintenance.md) for the chosen operation.
   Close harness processes and Chroma servers accessing this bank before mutation;
   do not curate it from a run that is actively using it. Use another workspace
   with bank access disabled for the curation agent, or let the engineer execute.

## Apply and Verify

`scripts/bank_guard.py` also exports guarded `backup`, `prune_unreferenced`, and
`reindex` functions. Import it in a short operation-specific Python script; there
is no separate CLI product to learn. Re-indexing builds a new candidate directory,
never switches the live bank. Pruning requires reviewed IDs and a matching audit
digest, creates a backup, and rejects published records and pending compactions.

For document rewrites or arc regrouping, work on a backup copy and implement the
specific migration with Chroma's SDK and the harness session APIs. Do not bypass
the data contract to make a validator pass. Get approval for the concrete migration;
this skill does not grant blanket permission to delete or rewrite memories.

Report the applied targets, backup/candidate paths, audit result, retrieval checks,
and unresolved issues. Do not claim metadata changes affect recall unless the
harness actually consumes those fields. Keep raw historical evidence unchanged.
