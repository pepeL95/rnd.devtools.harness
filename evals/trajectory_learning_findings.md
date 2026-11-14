# Trajectory Synthesis Iterations: 2026-11-14

## Current Scope

Trajectory compaction is first-person synthesis of the run, not extraction of
generalizable memories. Preserve investigation, evidence, changes in understanding,
and high-value verbatim snippets. A grounded forward-looking reflection is useful;
standing behavioral rules and claims about how tasks generally behave are not the
objective. The earlier transfer-oriented framing below is historical, not the
current acceptance criterion.

The system prompt and request footer now ask for trajectory synthesis. New outputs
use `[TRAJECTORY SYNTHESIS]` delimiters. Internal persistence identifiers remain
unchanged so existing sessions require no migration. Regression verification:
545 tests and 14 subtests pass in `quasipilot-harness`, including literal snippet
preservation by the wrapper and compatibility with older session records.

Direct review of `20261114T164750Z` found first-person analysis of the specific
startup investigation, exact diagnostic and tabbed diff excerpts, and no repeated
INFO logs or broad frequency claim. It still invents an initial environment-isolation
step and proposes checking credential formatting without evidence that this was
pending work. Restored context is 560 -> 409 tokens. The semantics are better aligned
with the requested scope, not demonstrated to be fully faithful or leaner than the
prior prompt (339 tokens in this pair). No further mechanism was introduced.

## First-Person Instructions

The prompt itself now opens with "I must synthesize the following completed
trajectory" and consistently uses first-person instructions. The request footer
uses the same perspective. Scope, model, and mechanics are unchanged; no examples
were added. This adopts the requested phrasing, not a demonstrated fidelity gain.

Direct review of the noisy-log pair in `20261114T165933Z` found the exact diagnostic,
complete tabbed diff including its hunk header, and literal mocked-test result in
the candidate. Repetitive INFO output was removed. It still overstated regression
coverage and proposed unrequested authentication work. Context was 560 -> 437 tokens,
versus 427 for the prior system prompt. Both legs used the new first-person footer;
this one-case comparison does not isolate the footer change or establish a general
improvement. Four model calls; 545 tests and 14 subtests pass in `quasipilot-harness`.

## Paired-Example Experiment

One good/bad startup example was appended to the `76a613d` prompt, with the good
example last. Its source was a local `api-key`/`api_key` mismatch before a provider
request, an exact tabbed diff, and three passing mocked tests. The bad example
invented frequency, a standing rule, and verified authentication; the good example
preserved the hypothesis, decisive evidence, exact patch, and unverified live auth.
The example was explicitly labeled as style guidance, not facts to import.

I directly reviewed four different trajectories, their drafts, and continuations.
No automated semantic scores or judges. Model, effort, input serialization, and
delimiters were identical between legs; only the appended example differed. These
are familiar development cases, not a held-out benchmark. The startup case used
to teach the prompt was not used to judge it.

| Case | Original context | Without example | With example |
| --- | ---: | ---: | ---: |
| Cancellation | 351 | 351 (skipped) | 351 (skipped) |
| Stale snapshot | 363 | 363 (skipped) | 363 (skipped) |
| Captured sanitization | 1,979 | 1,333 | 990 |
| Captured partial investigation | 2,735 | 950 | 910 |

These are estimated restored-context tokens, not scores. Both dense synthetic
cases retained their source because neither draft saved context. Their continuation
answers therefore cannot demonstrate synthesis quality. I inspected the rejected
drafts in the recorded calls as well as the published captured-turn syntheses.

- Cancellation: the example shortened the draft while retaining the failed wrapper,
  native async cancellation, same-turn state, side effects, and mocked/live boundary.
  It still omitted the uncommitted status and explicit interruption-role distinction.
- Stale snapshot: the example removed the invented comment from the quoted excerpt
  and the baseline's suggested cleanup utility. But it invented an earlier attempt
  to force a merge and omitted the observed exact-retrieval rejection. Shorter did
  not mean a more faithful trajectory.
- Sanitization: the example explicitly acknowledged that helper implementations
  had not been observed, unlike baseline, and avoided speculative truncation work.
  However, it still asserted a diagnostic-only, agent-remediated sanitization
  process that the partial read does not establish. This is a real improvement in
  one boundary, not a resolved fidelity problem.
- Partial investigation: the smaller draft still asserted prompt and test changes
  without a diff or test-file read. It lost baseline's explicit statement that tests
  had not been run, substituting a weaker statement about unverified live behavior.
  Both versions overclaim what the limited inspection established.

No startup-specific facts appeared in these unrelated drafts. The example added
223 reported input tokens per synthesis; synthesis output fell by 597 tokens across
the four pairs. Including continuations, this experiment used 16 recorded calls,
all with responses: 28,175 input tokens and 6,220 output tokens. Only previously
approved captured turns were sent to Gemini.

Decision: do not adopt this paired example. It improved brevity and one evidence
boundary, but not fidelity consistently enough to justify replacing the prompt.
The runtime remains unchanged; no new evaluation or runtime machinery was added.
The exact prompts, raw drafts, sources, and continuations remain in local bundles
`20261114T165445Z`, `20261114T165509Z`, and `20261114T165549Z` under
`eval_results/trajectory_compaction_ab/`. Regression suite: 545 tests and 14 subtests
passed in `quasipilot-harness`; this is mechanical verification, not semantic evidence.

## Earlier Simplified Contract

The subsequent simplification adopts a shorter prompt: first-person synthesis of
what was learned, plus high-value verbatim snippets for future turns. No new stages,
schemas, validators, model changes, or configuration. This is a simpler behavioral
contract, not a claim that the model's fidelity problems are solved.

Direct review of the final noisy-log sample (`20261114T164036Z`) found first-person
introspection, no repeated INFO output, and the exact tabbed patch. Restored context
fell from 560 to 340 estimated tokens, versus 246 for baseline. The memory still
overgeneralizes from one failure and introduces `AttributeError`, which was not in
the source. The continuation correctly distinguishes the new HTTP 401 from the old
local key mismatch. This supports the requested output shape, not a general quality
win or better compression than baseline.

The initial simplified prompt was also inspected on the captured sanitization turn
(`20261114T163949Z`); it retained the known unsupported claims and reconstructed
excerpts. The final copy-exactly clarification was rechecked on the noisy-log case,
not on that captured turn. The first noisy-log run is `20261114T163917Z`. These three
paired runs used 12 model calls; outputs are in the existing local eval bundles.

## Earlier Comparison

Neither prompt rewrite from the earlier follow-up was adopted; at that point the
runtime prompt remained the one in `89fab24`. Neither demonstrated better memory. This does
not establish that the baseline is adequate: its captured-code memory also invents
behavior. Model, request format, Markdown parsing, and publication safeguards are
unchanged. No additional runtime calls, configuration, or semantic validators.

The objective is retained learning: decisive evidence -> changed understanding ->
conditional next decision. It is not a shorter activity report or an inventory of
quotations. Keep failed approaches when their causes prevent repeating a mistake;
keep literal snippets when they let the agent act without reconstructing details.
Do not teach unsupported general rules from a successful local fix.

## Empirical Review

I read the sources, generated memories, and follow-up answers directly. No keyword
scores, automatic judges, or semantic pass/fail assertions were used. The existing
runner only generated and recorded outputs, usage, and restored-context estimates.
These are labeled development examples, not a blind or held-out evaluation.

Three existing probes now ask for decisions in related situations, rather than
primarily asking the agent to repeat the old diagnosis. Sources were not changed.
The synthesis model does not see the probe. The continuation receives the retained
user/final messages and memory, or the original events if the size guard skips it.
The approved captured sanitization turn was sent to Gemini; no new captures were
introduced. All calls used the existing Flash-Lite/minimal model configuration.

The first candidate emphasized causal learning, applicability, and minimal evidence.
One shorter revision made the private future-self purpose more explicit and asked
the model to correct earlier overclaims in retrospect. Both exact prompts and their
responses remain in the local, gitignored experiment bundles listed below.

| Case | Source context | Baseline after | Candidate after | Direct judgment |
| --- | ---: | ---: | ---: | --- |
| Noisy startup, then HTTP 401 | 560 | 246 | 270 | Both drop noise and retain the patch; no candidate advantage |
| Offloading lesson applied to plugin migration | 423 | 401 | 423 | Candidate expanded and was skipped; not successful compression |
| Captured sanitization, first candidate | 1,979 | 839 | 824 | Both teach unsupported behavior |
| Captured sanitization, shorter revision | 1,979 | 839 | 997 | Still unsupported, larger, and reconstructs code |

Counts are estimated restored context, including retained messages, not API usage.

- **Startup:** Both memories discard the repeated INFO lines and retain the tabbed
  key-change excerpt. Both continuations reject reapplying that fix to HTTP 401.
  The candidate adds an unsupported assurance that no live credentials were exposed;
  its extra text is not useful learning. Bundle `20261114T161856Z`.
- **Configuration:** The baseline retains the field-presence distinction but conflates
  resolution and startup backfill, and reformats code with an invented comment. Its
  continuation omits the explicit-default-list test, the closest counterexample to
  the proposed equality check. The candidate continuation includes that case, but
  it received the original trajectory because the generated memory was too large.
  That is not evidence for better compression. Bundle `20261114T161926Z`.
- **Partial read:** The source ends at line 100 inside `quality_issues`; the artifact
  detector and sanitizer implementations are not shown. Both candidates nevertheless
  claim sanitization is non-destructive and requires agent/manual remediation. The
  shorter revision also invents comments and elisions in a code block. Continuations
  reject the proposed blanket safety rule, but still rely on invented mechanics.
  A cautious-sounding answer is not enough. Bundles `20261114T161941Z` and
  `20261114T162109Z`.

This follow-up used 16 recorded calls, all returning responses: 24,568 reported input
tokens and 5,216 output tokens. These totals are separate from the earlier extraction
experiments. Bundles live under `eval_results/trajectory_compaction_ab/`.

## What Useful Memory Would Look Like

This is a manually written design example, not model output, a required template,
or an exact-match grading target:

> I inspected only lines 1-100 of `core/compaction/quality.py`. In `quality_issues`,
> `_has_compaction_artifacts(memory_document)` leads to an appended `QualityIssue`
> with `rule="compaction_process_artifacts"`. This establishes an issue-reporting
> call site, not how artifacts are detected or removed. My earlier explanation
> overreached: I have not seen the sanitizer or detector implementation, so I cannot
> treat cleanup as non-destructive or agent-driven. Before changing cleanup, I need
> those definitions, not another review of the heading list. A clean report alone
> would not establish that valuable content survived.

For a successful configuration turn, the equivalent learning is that field presence
and value are different evidence: explicit `[]` and explicit default-valued lists
must win over legacy fallback. The exact predicate and `ensure_config` backfill
behavior matter because materializing defaults can erase that distinction. Routine
test chatter does not substitute for this causal link.

The next useful experiment should test a focused evidence-to-learning example in
the prompt, not add more prohibitions or a runtime scoring system. It must still be
reviewed on another task to avoid teaching one memorized answer. This follow-up does
not establish reduced re-exploration or improved end-to-end coding performance.

Verification in `quasipilot-harness`: 544 tests and 14 subtests passed. Tests verify
mechanics, not the semantic quality of these memories.
