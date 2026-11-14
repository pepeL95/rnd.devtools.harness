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

The validated stack is DeepAgents 0.7.38, LangChain 1.5.0 (Core 1.8.0),
LangGraph 1.2.22, and the Google GenAI adapter 4.5.0 (SDK 3.3.0). When upgrading it, update the minimum
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

The optional [profile skill](skills/profile/SKILL.md) guides engineer-directed
preferences curation through this workflow. To make the versioned skill discoverable
globally (the repository's `skills/` directory is not searched automatically):

```bash
mkdir -p ~/.quasipilot/skills/profile
rsync -a skills/profile/ ~/.quasipilot/skills/profile/
```

Configure steering and executable hooks in global or workspace
`.quasipilot/.hooks/hooks.toml`. See the [hooks v1 reference](core/hooks/docs/v1.md)
for triggers, matching, and completion-checklist examples.

The local filesystem toolset remains `read_file`, `make_file`, `edit_file`, and
`execute`. `make_file` creates files with initial contents and refuses existing
paths; discovery and search use shell commands through `execute`. Skills retain
the harness's explicit `read_skill` activation.

For noisy commands, the agent can use `execute(command="pytest", output_mode="stash")`.
The default `auto` mode is unchanged. Stash waits for completion, captures combined
stdout/stderr to a private log in a session-specific directory under the nearest
workspace's `.quasipilot/artifacts/execute/`, and returns its path and exit code. Each log keeps
up to 10 MiB; excess output is drained and flagged as truncated. Failures include
the last 4 KiB of output, even if it falls beyond the saved prefix. Timeouts and
async cancellation terminate the command's process group on POSIX and retain the
partial log. Stash requires the local, non-virtual shell backend used by the driver.
Use targeted `rg`, `jq`, or `tail` queries to inspect logs rather than dumping them
back into context.

After a successful turn, the driver removes its session's logs only when
`completed_interaction_at - artifact_created_at > execute_retention_days` (7 by
default; 0 disables cleanup). Idle time alone never triggers deletion: there is
no startup or background sweep, and a resumed turn can read its old logs before
cleanup. Creation timestamps are embedded in filenames rather than inferred from
modification times. Active writers hold cross-process locks; cleanup skips locked
files and leaves other sessions, unowned legacy logs, and unknown files untouched.
Cleanup failures only warn; platforms without file locking retain logs. Preserve
important findings in session memory: logs are temporary and cleanup is not
reference-aware. Dormant sessions and unowned logs still require manual cleanup.

Configure reasoning and compaction in the nearest workspace's `.quasipilot/config.toml`
(defaults shown):

```toml
[reasoning]
eagerness = "high" # low, medium, or high (default)

[artifacts]
execute_retention_days = 7 # age relative to completed session interaction; 0 disables

[compaction]
offloading = ["lfs"] # ordered destinations; [] disables offloading

[compaction.session]
trigger_tokens = 80000
keep_last_turns = 2
max_critic_loops = 1

[compaction.trajectory]
trigger_every_turns = 1
```

CLI startup creates this file with defaults if it is missing, leaving existing
files and comments untouched. Omitted settings use in-memory defaults without
rewriting the file. The CLI and Python driver read settings when building the agent;
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

Trajectory compaction extracts a Markdown memory per selected turn: reasoning
lineage and decisive source excerpts, rather than model-generated JSON. Source
content is supplied with its original whitespace; live user corrections are included.
The final answer remains verbatim in the session and supplies task context, not
independent evidence of success. Larger intervals issue one
request per selected turn, not one batch request. A note must reduce the estimated
restored-context tokens for its turn; otherwise the original events stay in place.
When no turn shrinks, trajectory telemetry records a no-savings skip and model usage.
Empty or provider-reported
incomplete responses leave the source context intact; raw sessions and existing
memories are unchanged. Excerpt selection and semantic fidelity remain model-dependent.
See the [extraction evaluation](evals/trajectory_extraction_findings.md) for results and limits.

At the highest reasoning setting (`high`), the next model request requires
`reasoning` after every completed action-tool batch, including failures and single
calls, using named tool choice (the model adapter must support it). Checkpoints
use first-person introspective prose as durable memory, favoring high-signal
findings and concrete keywords without a fixed template or length quota.
Reasoning-only batches restore normal tool choice; lower settings are unchanged.

Session compaction extracts rich task episodes from newly evicted conversation, then
writes a cumulative prose arc using those episodes, original evidence, and prior memory.
The [session-arc pipeline](core/compaction/README.md) preserves tasks, concrete progress,
constraints, findings, and next actions without a rigid output schema. A bounded fidelity
review checks for lost context; on exhaustion, a warning is logged and the latest
nonempty revision is used. Extraction uses the compactor model, without new configuration.
By default, local-filesystem (`lfs`) offloading saves the final reviewed arc and the
exact intermediate task clusters, without extra model calls:

```text
~/.quasipilot/sessions/memories/<session-id>/
  arcs/<uuid>.md
  trajectories/<uuid>.md
```

Each compaction adds a matching UUID pair. Both files have YAML frontmatter with
`date: "yyyy-mm-dd-hh-mm-ss"` in UTC. Trajectories describe only newly evicted work;
arcs are cumulative. File references survive later compactions and can be read with
the existing filesystem tools. Custom session roots also relocate `memories/`.
Set `compaction.offloading = []` to disable new archives without changing the
arc pipeline or deleting existing files. Chroma/`trajectorybank` remains on its
feature branch, not a mode available on `main`.

Storage implementations subclass `BaseCompactionOffloadingStrategy`. Python callers
can pass one strategy or an ordered list to `Compactor(offloading=...)`; all receive
the same reviewed result. Stacks merge references and persist before session
publication. Any failure prevents publication, though earlier destinations may
already contain unreferenced archives. See the [strategy lifecycle](core/compaction/README.md#offloading-strategies).

The CLI start event shows estimated tokens selected for replacement. Completion shows
`before -> injected` tokens and savings or growth. These estimates count restored
message content, including tool calls and the memory wrapper, while excluding pinned
instructions, retained turns, and duplicate bookkeeping. Structured compaction telemetry
records that context breakdown separately from aggregate provider input/output tokens
and estimated-call counts. The trigger policy is unchanged.
