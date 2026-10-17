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
and the Google GenAI adapter 4.4.1. When upgrading it, update the minimum
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

Use `/devprofile` from an idle prompt to asynchronously evolve the workspace's
free-form `.quasipilot/DEVPROFILE.md` from completed turns in the active session.
An optional focus can guide the detached review, for example:

```text
/devprofile focus on testing and commit preferences
```

Configure steering and executable hooks in global or workspace
`.quasipilot/.hooks/hooks.toml`. See the [hooks v1 reference](core/hooks/docs/v1.md)
for triggers, matching, and completion-checklist examples.

The local filesystem toolset remains `read_file`, `make_file`, `edit_file`, and
`execute`. `make_file` creates files with initial contents and refuses existing
paths; discovery and search use shell commands through `execute`. Skills retain
the harness's explicit `read_skill` activation.

Configure reasoning and compaction in the nearest workspace's `.quasipilot/config.toml`
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

At the highest reasoning setting (`high`), the next model request requires
`reasoning` after every completed action-tool batch, including failures and single
calls, using named tool choice (the model adapter must support it). Checkpoints
use first-person introspective prose as durable memory, favoring high-signal
findings and concrete keywords without a fixed template or length quota.
Reasoning-only batches restore normal tool choice; lower settings are unchanged.

