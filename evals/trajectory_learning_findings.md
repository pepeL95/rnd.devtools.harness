# Agent-First Working Memory: 2026-11-14

## Simplified Contract

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
