# Quasipilot Harness

Lightweight LangChain coding agent harness.

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
