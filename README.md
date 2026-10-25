# Quasipilot Harness

Lightweight LangChain coding agent harness.

Harness instructions come from `.quasipilot/SYSTEM.md` in the working directory,
falling back to `~/.quasipilot/SYSTEM.md`. The local file replaces the home file;
if neither exists, the run fails. The selected content and source are snapshotted
per turn and recorded as system events: both raw and curated sessions keep only
the latest version, preserved through compaction. Instructions
stay fixed during a turn, including live steering, without duplicate model context.

If present, `AGENTS.md` in the working directory is appended to the model's system
prompt and refreshed at each new turn (not during a turn or live steering).
Both session files retain only the latest system entry, unchanged through
compaction. Removing or clearing the file replaces its entry with empty content
and stops injection on the next turn. Instruction version history is the client's
responsibility, not the harness's.
Missing or blank files initially add nothing;
unreadable files raise an error. No parent, home, or nested directories are searched.

The current implementation includes the core harness, driver agent, compaction,
session persistence, telemetry, and the Textual CLI.

## Development

```bash
./setup.sh
conda activate quasipilot-harness
python -m pytest
```

To synchronize an existing developer environment with the exact package
versions validated by the team:

```bash
./sync-env.sh
```

The script activates `quasipilot-harness`, installs the editable project and
pinned agent stack, validates dependencies, and runs the test suite. Pass a
different environment name when needed: `./sync-env.sh my-environment`.

The validated stack is DeepAgents 0.7.23, LangChain 1.4.4, LangGraph 1.2.14,
ChromaDB 1.5.9, and the Google GenAI adapter 4.4.1. When upgrading it, update the minimum
versions in `pyproject.toml` and exact pins in `sync-env.sh` together, then
run the synchronization script in `quasipilot-harness`.

Git preserves the executable permission, but if it was stripped when copying or
extracting the repository, restore it before running the script:

```bash
chmod +x sync-env.sh
./sync-env.sh
```

If you want a completely fresh environment, remove and recreate it:

```bash
conda env remove -n quasipilot-harness
./setup.sh
conda activate quasipilot-harness
python -m pytest
```

The compaction module loads `GOOGLE_API_KEY` from the repository `.env` file
when using the default Gemini-backed LangChain adapter.

Run the CLI with:

```bash
quasipilot
```

Use `/preferences` from an idle prompt to asynchronously evolve the workspace's
free-form `.quasipilot/PREFERENCES.md` from completed turns in the active session.
An optional focus can guide the detached review, for example:

```text
/preferences focus on testing and commit preferences
```

Preferences follow the SYSTEM.md/AGENTS.md session lifecycle: they are read at each
new turn, held stable through that turn and live steering, and persisted as one
latest system entry in both raw and curated sessions. The entry includes the
existing precedence guidance and survives compaction without duplicating
model context or appearing in chat history. Clearing or removing PREFERENCES.md
clears its entry on the next turn; an initially missing/blank file adds nothing.
Unreadable preferences raise an error instead of silently dropping instructions.

`PREFERENCES.md` replaces `DEVPROFILE.md`, and `/preferences` replaces `/devprofile`.
For an existing workspace, rename `.quasipilot/DEVPROFILE.md` to
`.quasipilot/PREFERENCES.md` before starting the CLI. A legacy-only file raises a
migration error rather than silently losing preferences. If both exist, the new
file wins, including when empty; neither file is automatically merged or rewritten.

Configure steering and executable hooks in global or workspace
`.quasipilot/.hooks/hooks.toml`. See the [hooks v1 reference](core/hooks/docs/v1.md)
for triggers, matching, and completion-checklist examples.

The local filesystem toolset remains `read_file`, `make_file`, `edit_file`, and
`execute`. `make_file` creates files with initial contents and refuses existing
paths; discovery and search use shell commands through `execute`. Skills retain
the harness's explicit `read_skill` activation.

Configure reasoning, compaction, and trajectory recall in the nearest workspace's `.quasipilot/config.toml`
(defaults shown):

```toml
[reasoning]
eagerness = "high" # low, medium, or high (default)

[compaction.session]
trigger_tokens = 80000
keep_last_turns = 2
max_critic_loops = 1

[compaction.trajectory]
trigger_every_turns = 1

[trajectory_bank]
enabled = false
automatic_min_similarity = 0.72
search_min_similarity = 0.35
automatic_results = 2
search_results = 5
automatic_tokens = 1200
search_tokens = 3000
```

CLI startup creates this file with defaults if it is missing, or appends entirely
missing top-level sections. Existing sections, values, and comments are preserved;
omitted keys within existing sections use in-memory defaults. Invalid files are
reported without modification. The CLI and Python driver read settings when building the agent;
the Python driver does not create the file.
Restart the CLI after editing it. Workspace discovery checks cwd, then parents;
it does not inherit home configuration or merge parent files. Missing settings
use the defaults above; invalid TOML, unknown keys, and invalid values raise an error.
An explicit `DriverAgentConfig(reasoning_eagerness=...)` overrides the file.
Reasoning eagerness controls the harness's tool, not the model's native thinking budget.
Compaction thresholds must be positive integers; retained turns and critic loops
may be zero. Explicitly supplied session or trajectory compaction coordinators
retain their own policies. Larger trajectory intervals delay memory synthesis;
the default preserves a trajectory after every completed turn.

At the highest reasoning setting (`high`), the next model request requires
`reasoning` after every completed action-tool batch, including failures and single
calls, using named tool choice (the model adapter must support it). Checkpoints
use first-person introspective prose as durable memory, favoring high-signal
findings and concrete keywords without a fixed template or length quota.
Reasoning-only batches restore normal tool choice; lower settings are unchanged.

Session compaction extracts rich task episodes from newly evicted conversation, then
writes a cumulative prose arc using those episodes, original evidence, and prior memory.
It preserves tasks, concrete progress, constraints, findings, and next
actions with source references, without a JSON schema, fixed headings, or turn assignments.
A bounded fidelity review checks for lost context, not formatting. If review attempts
run out, a warning is logged and the latest nonempty revision is used. The harness
retains an evidence index independently of the prose; archival and stale-session checks remain.
Extraction uses the compactor model (one additional call), without new configuration.
The [session-arc pipeline](core/compaction/README.md) runs independently of archival.
Its optional evidence-provider interface lets storage middleware supply references
and persist episodes without changing the generation prompts or review loop.
The CLI start event shows estimated tokens in the context selected for replacement,
excluding retained turns and pinned instructions. Completion shows `before -> injected`
tokens and the savings (or growth). Both sides estimate restored message content with
`cl100k_base` (characters/4 fallback), including tool calls and the injected memory wrapper,
but excluding duplicate storage fields, provider framing, and tool schemas. These are
comparable content estimates, not provider-reported request usage.

Structured `compaction.start` / `compaction.end` events live in the existing per-session
`~/.quasipilot/telemetry/YYYY/MM/DD/<session-id>.jsonl` and `.logs/compaction/` logs.
`context_tokens` records compacted, retained, injected, before/after, and saved tokens,
plus the estimation basis. Before/after totals describe the non-pinned compaction
snapshot, not newer turns arriving while compaction runs. Trajectory compaction records
the same completed breakdown in its telemetry and `.logs/trajectory/` logs.
Separately, `token_usage` records aggregate model input/output/total tokens and call
counts across extraction, synthesis, review, and revisions. `reported_calls` and
`estimated_calls` distinguish provider usage from fallback estimates. Its legacy
`source_tokens` field remains the serialized-event policy estimate, not the CLI delta.
Provider usage can include reasoning and cached input; retries not exposed by the
provider are not independently counted. Trigger thresholds are unchanged.

Trajectory-bank middleware is disabled by default. Set `[trajectory_bank] enabled = true`
to enable offloading and recall. Existing explicit settings are preserved.
Optional archival uses the nearest `.quasipilot/.trajectorybank` (cwd, then parents).
Disabling it does not change compaction semantics. New archives store task episodes,
not duplicate per-turn records; existing archives remain unchanged and available through
exact references, the `trajectory_bank` tool, and conservative post-reasoning
recall. Enabled CLI startup initializes or reopens the database; the local
embedding model still downloads only on first embedding use.
See [trajectory bank v1](core/trajectory_bank/docs/v1.md) for configuration,
thresholds, provenance checks, and evaluation commands. It is enabled by default;
set `[trajectory_bank] enabled = false` in workspace `config.toml` and restart to disable it.
If you have the old `.trajectorybank/config.toml`, move its settings into this
section and remove that old config file (not the database). Legacy files raise
a migration error rather than being silently ignored.
Developers can change the embedding default in
[`core/trajectory_bank/defaults.py`](core/trajectory_bank/defaults.py), using any
LangChain `Embeddings` implementation. Update its compatibility identity when
changing models; existing vectors require an explicit reindex.

For manual bank maintenance, install the versioned curation skill globally:

```sh
mkdir -p ~/.quasipilot/skills
rsync -a --exclude='__pycache__' --exclude='*.pyc' skills/trajectory-bank-curation/ ~/.quasipilot/skills/trajectory-bank-curation/
```

`SkillsMiddleware` discovers this directory automatically; `read_skill` loads its
instructions, references, and guarded Chroma SDK helpers. No additional middleware
or bank changes are needed. See [the skill](skills/trajectory-bank-curation/SKILL.md)
for offline audit, backup, pruning, and staged re-indexing. Re-run the copy after
updating the repository; installation does not change the bank itself.
