# Trajectory Compaction A/B: 2026-10-27

Historical synthetic-set review. The [captured-session follow-up](trajectory_compaction_transfer_findings.md)
found unresolved evidence-fidelity failures and did not approve promotion.

Referee judgment: prefer the candidate overall in these five examples, with a mixed
result on stale-snapshot recovery. Production remains unchanged pending a focused
refinement and a small captured-session check, not a larger scoring framework.

This direct source-backed review supersedes the original keyword-count verdict.
Equivalent wording is not lost knowledge; retaining a method name is less useful
than retaining its decisive condition. The earlier scores obscured both distinctions.

## Setup

Five synthetic turns in `trajectory_compaction_set.json`; no production session data.
Both variants use `gemini-3.1-flash-lite`, temperature 0, minimal thinking, identical
events and parser, and one synthesis call. A continuation call receives only the
memory and retained user/final messages, not raw events. Order alternates. The frozen
baseline matches production at `219849e`. Cases used for tuning are not held out.

I compared every source, both memories, and both continuation answers directly.
This is a labeled review by the candidate's author, not an independent blind trial.
Text-only continuation answers indicate resumability, not measured coding success.
Final bundle: `eval_results/trajectory_compaction_ab/20261027T220818Z/` (gitignored).
The runner now generates `comparison.md`; it does not assign semantic scores.

## Case Judgments

- **Whitespace: candidate.** Both explain tabs versus spaces and reject fuzzy
  normalization. The candidate also preserves the exact replacement literals, full
  `conda run -n quasipilot-harness python -m pytest tests/test_edit_diagnostics.py`
  command, and 18 passing tests. Both retain the mixed-indentation follow-up.
- **Cancellation: candidate, modestly.** Both distinguish failed synchronous-worker
  cancellation from native async cancellation and preserve `paused_turn_id` plus the
  mock-only verification limit. The candidate additionally retains `active_task.cancel()`,
  muted assistant introspection, uncommitted state, and non-rollback of completed tool
  effects. Its extra tokens carry useful state, not merely longer prose.
- **Explicit offloading: candidate.** The actual `model_fields_set` predicate survives,
  as does explicit `[]`. This is more actionable than a method name. The baseline
  invents possible merge work. Both blur runtime resolution versus persisted backfill
  and omit pushed status and the precise unrelated `low`/`minimal` test mismatch.
- **Stale snapshot: mixed.** The candidate better explains a correction changing the
  same turn and the captured-event-ID/current-prefix check. Exclusion from publication
  correctly conveys nonauthority. But the baseline explicitly preserves discard-and-
  resnapshot recovery, absent from the candidate memory. Conversely, the baseline
  invents orphan cleanup, which its continuation promotes to a background task. The
  candidate avoids that invention, though its continuation imprecisely describes the
  resnapshot source. Both lose newer-turn/pinned-instruction preservation guarantees.
- **Blocked embedding: candidate.** It keeps the model identities, 384/768 mismatch,
  read-only boundary, and separate-index continuation in fewer tokens. It explicitly
  preserves that no provider request was dispatched. Both identify the safe broad
  direction but omit explicit activation; neither establishes retrieval quality.

## Cost and Next Step

| Metric, summed over five turns | Baseline | Candidate |
| --- | ---: | ---: |
| Original restored-context estimate | 1,817 | 1,817 |
| Resulting restored-context estimate | 1,268 | 1,244 |
| Memory tokens, including wrapper/live edge | 994 | 972 |
| Provider synthesis input tokens | 6,443 | 8,638 |
| Provider synthesis output tokens | 1,196 | 1,176 |
| Synthesis / continuation failures | 0 / 0 | 0 / 0 |

The candidate saves 2.2% of memory tokens and 1.9% of resulting context but costs
34.1% more synthesis input. The quality preference is semantic, not a token-count win.
Next: preserve conditional recovery and safety invariants even for completed tasks,
without inventing pending work. Recheck the mixed case and a few captured turns
before promotion. Do not add another judge model or keyword gate.

No new model requests were needed for this review. Historical bundles remain intact:
the pilot `20261027T134002Z`, two-repeat pass `20261027T215754Z`, malformed-JSON
iteration `20261027T220429Z`, and final pass `20261027T220818Z`. Together these used
42 synthesis and 41 continuation requests; SDK retries are not independently counted.
Earlier identical temperature-zero repeats are not independent statistical evidence.

Final prompt SHA-256:

- Baseline: `19e259f3349b14645dbeca5b9b3bd0990b72b23b754b530572d8919347fc75eb`
- Candidate: `cab9a717d7056566286de8f84ec9ecaed3f39feafba66de8c43612776bd9a743`
