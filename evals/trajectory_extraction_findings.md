# Trajectory Extraction: 2026-11-14

## Initial Evaluation Decision

The initial decision was to keep this on `feature/trajectory-extraction`, not promote
it as a demonstrated semantic-fidelity improvement. Markdown removes model-output
JSON parsing, and literal input enables better excerpts, but extraction still
overclaims on captured evidence. The branch is a tested foundation, not a solved
catastrophic-forgetting mechanism.

The user subsequently requested promotion to main. That delivery decision does not
change these findings; later prompt experiments and their limits are recorded in
[the synthesis iteration notes](trajectory_learning_findings.md).

The production model selection is unchanged: `gemini-3.1-flash-lite`, minimal effort.
There is no new configuration, critic loop, structural rubric, or session migration.
Session-arc compaction and offloading are untouched.

## Implemented Contract

- One Markdown response per selected turn; code owns the turn ID. No JSON envelope,
  required headings, paragraph quota, or mandatory `live_edge` field.
- Event text reaches the model without nested JSON escaping; metadata remains JSON.
  All user corrections and the final-answer anchor are included. Claims are labeled
  as claims, not independent evidence.
- The prompt stays close to the established synthesis instructions, with explicit
  attention to literal excerpts, reasoning lineage, and verification boundaries.
- Only a strictly smaller estimated restored context replaces a turn. Dense turns
  stay intact. An all-skipped batch records its model usage and `no_context_reduction`
  in telemetry without publishing a pending rewrite.
- Empty, reasoning-only, or provider-reported incomplete responses preserve the
  original batch. Existing raw history, memory wrappers, pinned instructions, and
  stale-snapshot protections remain in place.

At the default interval, synthesis still costs one request per completed turn.
Larger intervals now require one request per selected turn instead of one batch
request. A no-savings skip still incurs synthesis cost; it saves no context tokens.

## Evaluation

I directly reviewed sources, memories, and continuation answers across five existing
synthetic cases, three captured turns, and two new synthetic extraction cases.
These are labeled development comparisons, not blind or held-out statistical tests.
The probe receives only retained context. If compaction is skipped, that includes
the original internal events; a skip is never counted as a compression success.
Continuation answers are a proxy, not measured end-to-end coding performance.

The baseline adapter reproduces the old prompt, input, and JSON output contract for
single-turn cases. All observed baseline notes were smaller than their source, so
the shared no-growth guard did not change their publication outcomes. Saved bundles
include raw outputs, prompts/hashes, provider usage, and sources under the gitignored
`eval_results/trajectory_compaction_ab/` directory.

### Final Mechanism Checks

| Case | Source context | Baseline after | Candidate after | Judgment |
| --- | ---: | ---: | ---: | --- |
| Noisy startup log | 560 | 224 | 246 | Useful extraction; exact tabbed diff retained, slightly larger than baseline |
| Dense tabbed implementation | 413 | 310 | 413 | Correctly skipped; original evidence retained, no compression claimed |
| Captured sanitization | 1,979 | 700 | 839 | Semantic failure remains in both; candidate is not an improvement |

Counts estimate restored message content using `cl100k_base`, not provider framing.

- **Noisy log:** Both continuations diagnose the local `api-key`/`api_key` mismatch
  and avoid claiming live authentication was tested. The candidate drops repeated
  progress logs while retaining the exact tabbed before/after diff, source location,
  test command, and three passing mocked tests. Context falls 56%, versus 60% for
  baseline; the additional 22 tokens carry source evidence. Bundle `20261114T154722Z`.
- **Dense code:** The model preserved literal tabs but added unobserved comments to
  its proposed code block and unnecessary prose. The size guard rejected the note.
  This is evidence preservation by retaining source, not proof that the model can
  copy arbitrary excerpts reliably. The continuation still normalized tabs inside
  the regex, so exact downstream reuse is not guaranteed. Bundle `20261114T154528Z`.
- **Sanitization:** The source stops inside `quality_issues` at line 100; it does not
  show the sanitization implementation. Both memories and probes nevertheless claim
  that sanitization is non-destructive and relies on issue remediation. The candidate
  also changes a relative path to `/core/compaction/quality.py`. Its smaller context
  is not a semantic success. Bundle `20261114T154752Z`.

The final prompt's earlier cancellation and stale-snapshot outputs were replayed
locally through the new guard, with prompt equality checked. Both are skipped:
351 -> 351 and 363 -> 363 tokens respectively. Their unguarded memories grew the
context to a combined 872 tokens and added unnecessary follow-up. This replay used
no new model calls and makes no claim about fresh continuation behavior.
Source bundle: `20261114T153937Z`.

### Rejected Experiments

- A broad evidence-first prompt produced larger notes and retained false certainty
  on partial reads. One stale-snapshot continuation incorrectly authorized resumption
  from unpublished records. Bundles `20261114T151939Z`, `20261114T152101Z`,
  `20261114T152205Z`, `20261114T152334Z`, `20261114T153200Z`, `20261114T153203Z`.
- Withholding the final answer did not fix overclaiming. Decoded input did preserve
  the complete tabbed source block in one pilot, unlike baseline, but did not repair
  evidence calibration. Bundles `20261114T152756Z`, `20261114T152759Z`,
  `20261114T152945Z`, `20261114T152948Z`.
- A shorter rewritten prompt invented a source function and unsupported follow-up.
  Two requests also hit the provider's per-minute quota; those runs are incomplete,
  not quality wins. Bundles `20261114T153539Z`, `20261114T153542Z`, `20261114T153558Z`.
- Gemini 3.6 Flash at minimal effort did not resolve the semantic failure; the adapter
  ignored its temperature setting, so this was a model/sampling-package comparison,
  not an isolated reasoning ablation. Bundles `20261114T152546Z`, `20261114T152550Z`.
- Flash-Lite at low effort increased memory from 496 baseline tokens to 1,812 on two
  captured cases without establishing fidelity. Bundle `20261114T153805Z`.
- Gemini 3.8 Flash at low effort timed out on both captured syntheses under the
  existing timeout. No quality conclusion follows. Bundle `20261114T154142Z`.

Across all experiments: 149 recorded calls, 145 responses, three failed syntheses
and one failed probe; reported usage totals 227,503 input and 41,636 output tokens.
SDK retries and server-side work on failed requests are not independently metered.
No JSON syntax failures occurred in this small live set, so it does not estimate
their production frequency. The Markdown path removes that parser by construction.

## Verification And Next Boundary

`quasipilot-harness`: 544 tests and 14 subtests pass. Regressions cover turn ownership,
multi-turn usage, pending-file reload, literal whitespace round trips, live corrections,
incomplete output, original-session preservation, old-memory compatibility, mixed
size-guard outcomes, and truthful skipped-run evaluation accounting.

Do not infer semantic correctness from parseability, smaller size, or passing unit
tests. A next experiment should investigate source-selected, deterministically copied
excerpts rather than further prompt expansion. Any such design must also address
unsupported prose; exact quotations alone will not fix evidence interpretation.
