SEGMENTATION_PROMPT = """You are reading a coding agent's trajectory. Your job is to segment it into
distinct episodes - coherent chunks of work with a clear intent.

An episode boundary occurs when:
- The agent shifts focus to a different sub-problem
- A significant failure causes a strategy change
- The user redirects the agent
- A discovery changes what the agent believes to be true
- They may not be linear - the agent might return to a previous episode after a detour

For each episode, output ONLY:
- EPISODE N: one-line label
- TURNS: use interval notation e.g. [1-5, 8, 12-15], indicating which turns belong to this episode
- TRIGGER: what started this episode (user message / discovery / failure)
- APPARENT GOAL: what the agent seemed to be trying to do

Do not summarize content yet. Just identify the boundaries."""

SYNTHESIS_PROMPT = """You are producing a structured memory document for a coding agent.
You have access to the full session trajectory and a segmentation map.

This document serves two purposes simultaneously:
1. RESUMPTION - enabling the agent to continue this specific task immediately
2. ARCHIVAL - building durable knowledge about this codebase and task class

Output style:
- Prefer semantic compression over transcript replay.
- Do not preserve long verbatim spans unless a literal phrase is itself load-bearing.
- Favor causal mechanisms, constraints, and decision-shaping facts over chronology.
- Distinguish confirmed facts from inference when certainty matters.
- Write like a senior engineer offloading judgment, not like a meeting scribe.
- Format the entire document as markdown with clear parent-child hierarchy.
- Use markdown headings consistently so the document is scannable by both humans and agents.

Produce the following sections in order:

## Episodic Memory

### Task Requirement Synthesis
  This section should read like a bank of cohesive episodic memories, one memory per semantic task.
  Synthesize all explicit instructions, corrections, constraints, preferences, and meaningful redirections.
  Group semantically linked work into the same task memory when the task stayed conceptually continuous.
  Split only when the session genuinely changed goals, assumptions, or problem frames.
  Do not quote the whole prompt back. Compress to the operative requirements, tradeoffs, discoveries, and why the task mattered.
  Format each semantic task as its own markdown subsection:
  #### [short task title]
  - FULL-FIDELITY REF: [turn interval(s) or dump reference for this task memory]
  - TASK TIMESTAMPS: [start timestamp -> end timestamp in ISO-like form, or the most faithful available range from session events]
  - TASK REQUEST SYNTHESIS: [what was being asked, constrained, or redirected]
  - TASK EXECUTION SYNTHESIS: [what happened in this task, what changed, what was learned, and what state transition matters]
  - PRIORITY SIGNALS: [constraints, corrections, discoveries, or preferences that governed execution]
  - OPEN LOOP: [what about this task still matters for continuation, if anything]
  Each task memory should feel cohesive: a future agent should understand the task's shape, execution arc, and remaining edge without reconstructing the transcript.
  Use timestamps to make the next agent temporally aware of when the task happened relative to the preserved tail.

### Current State
  Files created, modified, or deleted with exact paths, test/build status, external state,
  and what is confirmed versus believed.
  Prefer the state that would matter if the next agent resumed cold.

### Work Completed
  Concrete irreversible progress, in past tense, with exact paths.
  Focus on meaningful state transitions and decisions, not a step-by-step timeline.

### Failed Approaches
  The goal is not to list attempts, but to extract reasoning that prevents re-exploration.
  For each failed or abandoned line of attack, include:
  - APPROACH
  - FAILURE MECHANISM
  - REUSABLE LESSON
  - STATUS: abandoned / superseded / maybe salvageable

### Open Problems
  What remains unsolved, including partial reasoning and live hypotheses.
  Separate confirmed blockers from active hypotheses when both exist.

### Implicit Tasks Discovered
  Necessary sub-goals that emerged, why they emerged, and what was produced or learned.

### Next Steps
  Concrete prioritized next steps specific enough to execute without rereading the trajectory.
  These should read like an actionable continuation plan, not generic advice.

## Semantic Memory

### Codebase Characteristics
  Durable facts about this system, implicit contracts, load-bearing assumptions, and non-obvious runtime behaviors.
  Exclude transient task-local facts unless they expose a lasting contract.

### Task-Approach Pairs
  For each task class: TASK CLASS, EFFECTIVE APPROACH, PITFALLS, CONFIDENCE.
  Emphasize why the approach worked, not just what was done.

### Generalizable Insights
  Lessons that generalize beyond this codebase, with confidence.
  Each insight should carry an observation, a mechanism, and why it transfers.

## Handoff

### Session Narrative
  2-3 paragraphs. Be candid about uncertainty, partial progress, and open risk.
  This is a curved continuation into the retained full-fidelity turns: orient the next agent so the preserved tail feels like a natural continuation rather than a hard context jump.
  Emphasize the live edge of the work, why the preserved turns matter, and what lens the next agent should carry into them."""

CRITIC_PROMPT = """You are a critic evaluating a compacted memory document against the original
agent trajectory that produced it.

Find what the document got wrong, missed, or misrepresented. Do not rewrite it.
For each violation found:
- RULE: which criterion was violated
- LOCATION: where in the document the problem appears or should appear
- EVIDENCE: quote or reference the trajectory turn that proves the violation
- FIX: the specific change needed

Evaluate these criteria:
- The document uses coherent markdown hierarchy, and TASK REQUIREMENT SYNTHESIS is organized into subsections, one per clustered semantic task, each with a full-fidelity reference and task timestamps.
- The memory is useful for resumption: CURRENT STATE, OPEN PROBLEMS, and NEXT STEPS together let a new agent continue without rereading the transcript.
- The memory maximizes signal over noise: it captures durable mechanisms, non-obvious contracts, reusable lessons, and the live continuation edge rather than replaying chronology.

After all violations, output:

CRITIQUE SUMMARY
  VIOLATIONS FOUND: N
  SEVERITY: [blocking / moderate / minor]
  RECOMMENDED ACTION: [revise targeted sections / full rewrite / approve as-is]"""

REVISION_PROMPT = """You are revising a memory document based on a critic's findings.

Rules:
- Address every violation marked blocking or moderate
- Address minor violations unless doing so would require restructuring sections that passed critique
- Do not introduce new content beyond what the violations call for
- Preserve the memory document section order
- Push the revision toward higher signal density and lower transcript mimicry.

After revision, append:

REVISION LOG
  [VIOLATION N] -> [what was changed and where]

If any violation cannot be resolved without a full rewrite, output:
ESCALATION: [section] requires full rewrite because [reason]"""
