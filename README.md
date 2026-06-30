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

To upgrade the installed harness package and refresh dependencies in the
developer environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quasipilot-harness
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e ".[dev]"
python -m pytest
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
