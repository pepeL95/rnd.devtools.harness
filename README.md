# Quasipilot Harness

Lightweight LangChain coding agent harness.

The current implementation focuses on the core and driver-agent foundation from
`docs/specs/`. The CLI surface is intentionally not implemented yet.

## Development

```bash
conda activate quasipilot-harness
python -m pytest
```

The compaction module loads `GOOGLE_API_KEY` from the repository `.env` file
when using the default Gemini-backed LangChain adapter.
