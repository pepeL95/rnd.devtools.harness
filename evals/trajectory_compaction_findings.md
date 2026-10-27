# Trajectory Compaction A/B: 2026-10-27

Decision: retain the production prompt. The candidate is experimental, not promoted.

## Setup

Five synthetic turns in `trajectory_compaction_set.json`; no production session data.
Both variants use `gemini-3.1-flash-lite`, temperature 0, minimal thinking, identical
events and parser, and one synthesis call. A separate evaluation-only continuation
call receives only the compressed memory and retained user/final messages. Neither
model receives the checks or answers. Order alternates. The frozen baseline matches
the production prompt at commit `219849e`.

Each case has five regex-group preservation checks and three continuation checks.
These measure anchor coverage, not semantic correctness or actual avoided tool calls.
Manual source comparisons supplement them. Cases used for tuning are not held out.

## Final Paired Pass

Bundle: `eval_results/trajectory_compaction_ab/20261027T220818Z/` (local, gitignored).
One paired run per case; no synthesis or probe failures.

| Metric, summed over five turns | Baseline | Candidate |
| --- | ---: | ---: |
| Original restored-context estimate | 1,817 | 1,817 |
| Resulting restored-context estimate | 1,268 | 1,244 |
| Memory tokens, including wrapper/live edge | 994 | 972 |
| Preservation checks | 11/25 | 13/25 |
| Memory-only continuation checks | 6/15 | 8/15 |
| Provider synthesis input tokens | 6,443 | 8,638 |
| Provider synthesis output tokens | 1,196 | 1,176 |

| Case | Memory tokens A/B | Preservation A/B | Continuation A/B |
| --- | ---: | ---: | ---: |
| Exact whitespace | 190 / 184 | 2 / 3 | 2 / 2 |
| Same-turn cancellation | 200 / 228 | 3 / 4 | 1 / 3 |
| Explicit offloading | 212 / 209 | 1 / 2 | 1 / 2 |
| Stale snapshot | 203 / 199 | 2 / 1 | 2 / 1 |
| Blocked embedding | 189 / 152 | 3 / 3 | 0 / 0 |

The candidate saves 2.2% of memory tokens, 1.9% of resulting context, and costs 34.1%
more synthesis input tokens. This is not a demonstrated production-quality win.

## Observations and Iterations

- The final candidate retains the exact named-environment test invocation in the
  whitespace case and the actual `model_fields_set` condition in the config case.
  It separates the failed worker wrapper from native async cancellation and retains
  uncommitted state and the non-rollback boundary.
- The stale-snapshot note loses the explicit resnapshot continuation. Both variants
  omit some newer-turn/pinned-instruction guarantees. The baseline invents a future
  orphan-cleanup task; the candidate avoids that but does not retain every useful fact.
- Both omit pushed status and the `low`/`minimal` baseline-failure details in the
  config case, and explicit activation and some verification limits in the embedding
  case. Smaller notes alone are not adequate evidence for adoption.
- Regex checks under-credit valid paraphrases: "excluded from publication" does not
  match "unpublished", and a correct condition can survive without its method name.
  They also cannot detect all wrong causal relationships. Scores are coverage proxies.
- An earlier candidate (`20261027T215754Z`, two repeats) used 16.8% fewer memory
  tokens, but conflated an omitted offloading field with an absent compaction block
  and blurred the failed cancellation approach with the implementation. Repeats at
  temperature zero were identical, not independent statistical confirmation.
- The next candidate (`20261027T220429Z`) produced invalid JSON in one of five cases.
  Explicit quote/backslash/newline escaping fixed that case in the final pass, but
  one successful pass does not establish a general formatting guarantee.
- The initial one-case pilot is `20261027T134002Z`. Across four experiments there
  were 42 synthesis requests and 41 continuation requests; the malformed synthesis
  skipped its probe. SDK-hidden retries are not independently counted. Early bundles
  retain notes/errors; the final runner also retains raw calls and counts failed
  compactions as zero coverage with unchanged context.

Next experiment: protect exact conditions, reproducible verification, publication
state, and conditional recovery steps before polishing prose. Keep checks fixed,
audit causal claims against sources, and validate on held-out captured turns before
promotion. The separate live-steering input omission remains outside this prompt-only
experiment; instructions cannot preserve a directive absent from the request.

Final prompt SHA-256:

- Baseline: `19e259f3349b14645dbeca5b9b3bd0990b72b23b754b530572d8919347fc75eb`
- Candidate: `cab9a717d7056566286de8f84ec9ecaed3f39feafba66de8c43612776bd9a743`
