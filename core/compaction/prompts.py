SEGMENTATION_PROMPT = """You are reading a coding agent's trajectory. Your job is to segment it into
distinct episodes - coherent chunks of work with a clear intent.

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

Do not summarize content yet. Just identify the boundaries."""

SYNTHESIS_PROMPT = """You are producing a structured memory document for a coding agent.
You have access to the full session trajectory and a segmentation map.

This document serves two purposes simultaneously:
1. RESUMPTION - enabling the agent to continue this specific task immediately
2. ARCHIVAL - building durable knowledge about this codebase and task class

Produce the following sections in order:

━━━ EPISODIC MEMORY (task-specific, resumption-oriented) ━━━

ORIGINAL TASK
  The user's first message, preserved verbatim or near-verbatim.

USER DIRECTIVES
  All explicit instructions, corrections, constraints, and preferences stated by the user.
  Format: "[near-verbatim quote or close paraphrase]" - turn N

CURRENT STATE
  Files created, modified, or deleted with exact paths, test/build status, external state,
  and what is confirmed versus believed.

WORK COMPLETED
  Concrete irreversible progress, in past tense, with exact paths.

FAILED APPROACHES
  Every dead end, including ATTEMPT, OUTCOME, CAUSE, and VERDICT.

OPEN PROBLEMS
  What remains unsolved, including partial reasoning and live hypotheses.

IMPLICIT TASKS DISCOVERED
  Necessary sub-goals that emerged, why they emerged, and what was produced or learned.

NEXT STEPS
  Concrete prioritized next steps specific enough to execute without rereading the trajectory.

━━━ SEMANTIC MEMORY (durable, transferable across tasks) ━━━

CODEBASE CHARACTERISTICS
  Durable facts about this system, implicit contracts, and load-bearing assumptions.

TASK-APPROACH PAIRS
  For each task class: TASK CLASS, EFFECTIVE APPROACH, PITFALLS, CONFIDENCE.

GENERALIZABLE INSIGHTS
  Lessons that generalize beyond this codebase, with confidence.

━━━ HANDOFF ━━━

SESSION NARRATIVE
  2-3 paragraphs. Be candid about uncertainty, partial progress, and open risk."""

CRITIC_PROMPT = """You are a critic evaluating a compacted memory document against the original
agent trajectory that produced it.

Find what the document got wrong, missed, or misrepresented. Do not rewrite it.
For each violation found:
- RULE: which criterion was violated
- LOCATION: where in the document the problem appears or should appear
- EVIDENCE: quote or reference the trajectory turn that proves the violation
- FIX: the specific change needed

Evaluate these criteria:
- The original user task is verbatim or near-verbatim
- Every user correction or mid-task directive appears in USER DIRECTIVES
- Every file mentioned has its exact path
- Every failed approach has a stated mechanism
- CURRENT STATE reflects the actual state at compaction
- OPEN PROBLEMS includes partial reasoning
- NEXT STEPS are specific enough to act on
- No pursued implicit task is missing
- Every TASK-APPROACH PAIR has a confidence rating
- No high-confidence insight lacks evidence and causal mechanism
- CODEBASE CHARACTERISTICS contains only durable facts
- GENERALIZABLE INSIGHTS are marked speculative where evidence is thin

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

After revision, append:

REVISION LOG
  [VIOLATION N] -> [what was changed and where]

If any violation cannot be resolved without a full rewrite, output:
ESCALATION: [section] requires full rewrite because [reason]"""
