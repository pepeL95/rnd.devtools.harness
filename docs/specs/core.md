# Specifications for core/

We will be building the agent harness using Langchain


## middleware

This section defines all custom langchain middleware we will be creating for the harness. Most of will flow pretty intuitively from the `structure.md` specs (i.e. we enable them as agent/model lifecycle hooks leveraging their core implementations). Here we expand on some of the more intricate ones:

### session_dump.py

The SessionDumpMiddleware must build session artifacts and offload them into 2 places simultaneusly:

- ~/.quasipilot/sessions/dump/[session-id].jsonl # Full fidelity session history
- ~/.quasipilot/sessions/curated/[session-id].jsonl # Full fidelity, until compaction happens .. this is what the agent will see .. session dump doesnt have to worry about it, just keep appending new full fidelity items

This middleware works tightly with the session/ core, respecting data contracts and policies.

### session_load.py

The SessionLoadMiddleware loads session history into the agent's messages from `- ~/.quasipilot/sessions/curated/[session-id].jsonl`

- It formats is as langchain messages
- Works closely with the session/ core for contracts and policies

### system_prompt.py

A simple SystemPromptMiddleware to pass custom system prompts into the agent. Can be provided via SYSTEM.md file path or as a string

### runtime.py

Injects a runtime context session into the agent's system prompt. Context includes:

- cwd -> this should be probed often
- git dirty status
- git branch

Extensibility must be taken into account for runtime ctx additions later on.

## session

This is the session logic implementation core. Sessions must contain conversation history and runtime context.

**A basis for session elements is provided below (can expand):**

- timestamp
- type: turn_begin|turn_end|runtime|meta|user|reasoning|tool|tool_output|assistant
- turn: (1-based)
- payload:
    - model
    - turn
    - input/output contracts for messages/tools etc. Think how to best represent this
    - Anything else you think necessary (think about telemetry as well)
- Anything else you think necessary. But keep it lean

Ensure to use the folder structure to employ good engineering principles.

## compaction

- There is a GOOGLE_API_KEY already under .env -- we will use gemini models for evaluating our compaction module.

The compaction module will be a very detailed one. These are the core components of the design:

- A policy dictating when/how it will be triggered (by default it'll be trigger at 8000 tokens reached, keeping the previous k=5 turns as full fidelity).
- The compaction will be used to replace the [0, ...k) turns with the compacted memories in the `- ~/.quasipilot/sessions/curated/[session-id].jsonl`
- The module integrates closely with session core contracts

### Compaction Process

1) Semantic task segmentation: A fast step in which an LLM will segment the raw trajectory into meaningful episodes. Don't summarize yet — just identify boundaries.
Sample Prompt
```
You are reading a coding agent's trajectory. Your job is to segment it into
distinct episodes — coherent chunks of work with a clear intent.

An episode boundary occurs when:
- The agent shifts focus to a different sub-problem
- A significant failure causes a strategy change
- The user redirects the agent
- A discovery changes what the agent believes to be true

For each episode, output ONLY:
- EPISODE N: [one-line label]
- TURNS: [start] to [end]
- TRIGGER: what started this episode (user message / discovery / failure)
- APPARENT GOAL: what the agent seemed to be trying to do

Do not summarize content yet. Just identify the seams.
```

2) Step 2 — Synthesis: Receives the full raw trajectory + the segmentation map. Produces the full two-tier memory document in one pass.

Sample Prompt
```
You are producing a structured memory document for a coding agent.
You have access to the full session trajectory and a segmentation map.

This document serves two purposes simultaneously:
  1. RESUMPTION — enabling the agent to continue this specific task immediately
  2. ARCHIVAL — building durable knowledge about this codebase and task class

Produce the following sections in order:

━━━ EPISODIC MEMORY (task-specific, resumption-oriented) ━━━

ORIGINAL TASK
  The user's first message, preserved verbatim or near-verbatim.
  Do not paraphrase. This is ground truth.

USER DIRECTIVES
  All explicit instructions, corrections, constraints, and preferences 
  stated by the user across the session — including mid-task redirections.
  Format: "[near-verbatim quote or close paraphrase]" — turn N
  These override agent judgment. Miss none.

CURRENT STATE
  The state of the world at the moment of compaction:
  - Files created, modified, or deleted — with exact paths and one-line description of current state
  - Test status — which pass, which fail, which are skipped. Be specific.
  - Build / lint / type-check status if checked
  - Any external state changed (schemas, config, env, dependencies)
  - What the agent believes to be true right now vs. what is confirmed

WORK COMPLETED
  Concrete, irreversible progress. Past tense. Specific.
  - "Implemented [what] in [exact path]"
  - "Determined that [X] does not work because [mechanism]"
  Not: "Made progress on auth." Yes: "Moved JWT validation into middleware.ts:42,
  replacing inline checks that were inconsistently applied across 3 routes."

FAILED APPROACHES
  Every dead end, with its mechanism of failure. This section prevents re-exploration.
  For each:
  - ATTEMPT: what was tried
  - OUTCOME: what happened
  - CAUSE: why it failed — be mechanistic, not vague
  - VERDICT: abandoned / superseded / partially salvageable

OPEN PROBLEMS
  What is unsolved at the moment of compaction.
  Include: partial reasoning, live hypotheses, things the agent was mid-thought on.
  Distinguish confirmed blockers from suspected ones.

IMPLICIT TASKS DISCOVERED
  Sub-goals the agent pursued that were never explicitly requested but were
  necessary or valuable. For each:
  - What it was
  - Why it emerged
  - What was produced or learned

NEXT STEPS
  The agent's concrete, prioritized plan going forward.
  Specific enough to act on immediately without re-reading the trajectory.
  If there's genuine ambiguity about what to do next, name it explicitly —
  do not collapse uncertainty into false confidence.

━━━ SEMANTIC MEMORY (durable, transferable across tasks) ━━━

CODEBASE CHARACTERISTICS
  Facts about this system that aren't obvious from reading the code.
  Implicit contracts, surprising behaviors, load-bearing assumptions,
  performance characteristics, API quirks, coupling that isn't visible
  from file structure. Each entry should be a transferable fact, not
  a task-specific observation.

TASK-APPROACH PAIRS
  For each distinct class of task encountered this session:

  TASK CLASS: [generalizable description — not the specific instance]
  EFFECTIVE APPROACH: [what worked and why — specific enough to replicate]
  PITFALLS: [what to avoid and the mechanism of failure]
  CONFIDENCE: [high / medium / low]
    high   = observed working, clear causal structure, repeatable
    medium = worked once, causation plausible but not proven
    low    = single observation, unclear causation, treat as hypothesis

GENERALIZABLE INSIGHTS
  Lessons that transcend this codebase — about the problem domain,
  the tech stack, or the class of task. Only include if the evidence
  is clear. Mark speculative ones explicitly.
  Each insight: [observation] — [why it generalizes] — [confidence]

━━━ HANDOFF ━━━

SESSION NARRATIVE
  2-3 paragraphs. A senior engineer's honest account of this session —
  what was attempted, what was learned, what the shape of the remaining
  work looks like. Written for a reader who needs to understand the
  session quickly without reading the memory document in full.
  Be candid about uncertainty, partial progress, and open risk.
```

3) Step 3 — Critic loop .. Enforcing guidelines: Reads the GUIDELINES.md, full trajectory being compacted and compacted trajectories in order to provide further directives til' convergence or until loop hyperparameter capped

Sample Prompt
```
You are a critic evaluating a compacted memory document against the original
agent trajectory that produced it.

You have access to both. Your job is to find what the document got wrong,
missed, or misrepresented — not to rewrite it, only to flag it precisely.

Evaluate against each criterion below. For each violation found:
  - RULE: which criterion was violated
  - LOCATION: where in the document the problem appears (or should appear)
  - EVIDENCE: quote or reference the trajectory turn that proves the violation
  - FIX: the specific change needed — concrete, not directional

─── EPISODIC FIDELITY ───

[ ] The original user task is verbatim or near-verbatim — not paraphrased or compressed
[ ] Every user correction or mid-task directive appears in USER DIRECTIVES
[ ] Every file mentioned has its exact path — no vague references
[ ] Every failed approach has a stated mechanism, not just an outcome
[ ] CURRENT STATE reflects the actual state at compaction, not an earlier point
[ ] OPEN PROBLEMS includes partial reasoning, not just problem labels
[ ] NEXT STEPS are specific enough to act on without re-reading the trajectory
[ ] No implicit task that the agent visibly pursued is missing from IMPLICIT TASKS

─── SEMANTIC FIDELITY ───

[ ] Every TASK-APPROACH PAIR has a confidence rating
[ ] No insight rated high-confidence is evidenced only once in the trajectory
[ ] No insight rated high-confidence lacks a stated causal mechanism
[ ] CODEBASE CHARACTERISTICS contains only facts, not task-specific observations
[ ] GENERALIZABLE INSIGHTS are marked speculative where evidence is thin

─── OMISSION CHECK ───
  Read the trajectory episode by episode against the memory document.
  Flag any episode whose substance — decisions made, discoveries, failures — 
  is not represented anywhere in the document.

─── DISTORTION CHECK ───
  Flag any claim in the document that is contradicted by, or not supported by,
  the trajectory. Pay particular attention to:
  - Outcomes stated as successes that were actually partial or ambiguous
  - Failure causes that are vague where the trajectory shows a clear mechanism
  - Confident assertions where the agent was visibly uncertain

After all violations, output:

CRITIQUE SUMMARY
  VIOLATIONS FOUND: N
  SEVERITY: [blocking / moderate / minor] — blocking means the document
    would cause the agent to resume incorrectly or repeat a known failure
  RECOMMENDED ACTION: [revise targeted sections / full rewrite / approve as-is]
```

4) Step 4 — Revision loop .. ingests critic's directives and acts upon them

Sample Prompt
```
You are revising a memory document based on a critic's findings.

You have: the original memory document, the critic's violation report,
and the original trajectory for reference.

Rules:
- Address every violation marked blocking or moderate
- Address minor violations unless doing so would require restructuring
  sections the critic approved
- Do not restructure or rewrite sections that passed critique
- Do not introduce new content beyond what the violations call for
- For each change made, note at the bottom which violation it resolves

After revision, append:

REVISION LOG
  [VIOLATION N] → [what was changed and where]

If any violation cannot be resolved without a full rewrite of a section,
flag it rather than attempting a patch — output:
  ESCALATION: [section] requires full rewrite because [reason]
```

How they chain:

Raw trajectory
    ↓ Prompt 1 — segmentation (fast, parallel-safe)
Episode map
    ↓ Prompt 2 — synthesis (trajectory + map as input)
Draft memory document
    ↓ Prompt 3 — critic (trajectory + draft as input)
Violation report
    ↓ Prompt 4 — revision (draft + violations + trajectory)
Revised document
    ↓ [if critic flagged blocking violations] loop Prompt 3 → 4
    ↓ [else] done
Final memory document

### Injection Format for agent

Once finalized the agent sees this as a user message:

```
[MEMORY RESTORE]
The following document was produced by a compaction process over your previous
session trajectory. It was not written by you. Treat it as accurate.
The episodic sections orient you to resume the current task.
The semantic sections reflect durable knowledge about this codebase.

[paste final memory document]

[END MEMORY RESTORE]

Your task continues below.
```

## telemetry

Record useful metrics (exceptions, token usage (tiktoken), compaction triggers, etc...)
We want to record as much useful information as possible to uses as training data for evolving the harness

## utilities

Useful, reusable utilities (e.g. formatting, authentication, etc...) to keep core logic clean
