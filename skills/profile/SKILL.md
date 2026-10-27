---
name: profile
description: Curate durable engineer requests, working preferences, and confirmed behavioral patterns in Quasipilot PREFERENCES.md. Use when reviewing learned preferences, recording a lasting correction, or maintaining the engineer's working profile.
---

# Engineer Profile

Treat the profile as the engineer-directed counterpart to repository `AGENTS.md`:
how this engineer wants work performed, not a repository manual or session diary.
The active file is `.quasipilot/PREFERENCES.md` under the harness runtime cwd.
Do not create `profile.md`, `PROFILE.md`, or a parallel global profile; those are
not loaded by the current mechanism. The skill directory is not the target cwd.
Current engineer requests and repository/system instructions take precedence.

## Learn What Should Persist

Capture reusable engineer instructions about communication, implementation taste,
quality, testing, commits, tools, environments, and collaboration. These are lenses,
not required sections. One explicit durable request is enough; repetition is not
required. "Always explain why in commit messages" is a preference; "commit this"
is a one-turn task, not standing permission to commit or push.

Use user-authored statements as evidence. Recurring patterns can suggest a question
to ask the engineer, but do not persist an inferred preference until confirmed.
Assistant behavior, reasoning, tool results, repository conventions, and silence
are not evidence of what the engineer wants. Exclude secrets, personality guesses,
task status, and codebase facts; keep repository instructions in `AGENTS.md`.

## Maintain Through the Existing Workflow

Do not rewrite the profile as a side effect of ordinary coding. From an idle CLI
prompt, `/preferences` reviews completed turns and merges learned instructions;
an optional focus can narrow the review, for example:

```text
/preferences focus on testing and commit-message preferences
```

This is a CLI slash command, not a shell command. If you cannot dispatch it, offer
the command to the engineer rather than claiming it ran or bypassing the workflow.

When operating with the scoped preferences-maintenance tools:

1. Call `inspect_session` and `read_preferences` first. Use `search_session` and
   `read_session_turns` to obtain exact user quotes from completed turns as needed.
2. Merge into the existing document. Preserve instructions not discussed in this
   session. Revise or remove one only when newer user evidence changes or retracts
   it; retain the engineer's qualifiers and scope. If nothing changed, do not write.
3. Write concise, actionable Markdown with descriptive `#` and `##` headings, not
   a fixed taxonomy, transcript, evidence log, or history of revisions.
4. Use `update_preferences` with the complete merged content, the revision returned
   by `read_preferences`, and supporting `{turn, quote}` evidence. On a conflict,
   reread and reconcile; never overwrite blindly. Follow the tool's empty-state
   instructions only for an absent profile with no learned preferences.
5. Report only confirmed changes, or that no update was warranted. Do not claim
   persistence after a rejected update or claim changes apply to the active turn.

The middleware reads the latest file at each new turn and holds it stable through
that turn and live steering. One latest system entry is kept in both raw and curated
sessions and survives compaction. Do not edit those session entries yourself.

If only legacy `.quasipilot/DEVPROFILE.md` exists, report the migration requirement:
rename it to `PREFERENCES.md` with the engineer's approval. If both exist, the new
file wins, even when empty; do not silently merge or discard either file.
