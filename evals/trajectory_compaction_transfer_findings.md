# Trajectory Memory Transfer Check: 2026-10-27

Decision: do not promote the experimental prompt, increase production reasoning
effort, or change production serialization. Smaller notes were not consistently
better memory. This qualifies the earlier synthetic-only preference for the candidate.

## Method

The objective was accurate continuation with less context and less rediscovery,
not phrase coverage. I read each source, both memories, and continuation answers.
Three native turns (2, 3, 4) from `c2d0dc29c65a43ac94484c9c24fe0b23.jsonl` contrast
partial investigation, code explanation, and a completed environment check. Their
events, turn numbers, and boundaries were preserved, not reconstructed as clean
synthetic traces. The existing five synthetic cases served as development probes.

The model remained `gemini-3.1-flash-lite`, temperature 0. Prompts shared identical
input and alternated order. The effort ablation changed synthesis only; probes
remained at minimal effort. The format ablation changed representation, not evidence.
These are labeled, author-reviewed examples from one captured session, not a blind
benchmark or statistical estimate. Once inspected and used to refine prompts, the
captured cases became development data. No held-out success is claimed.

## Findings

- **Recovery improved in the shorter prompt's synthetic test.** It retained discard-
  and-resnapshot recovery without the baseline's invented background cleanup task.
  But cancellation lost role/presentation semantics and invented a file association;
  embedding migration lost the separate-candidate-index instruction. Shorter was mixed.
- **Captured investigation exposed evidence inflation.** Only `git status` and the
  first 50 lines of two files were read; no diff or tests were inspected/run. Yet
  memories described test updates as validating new behavior. The final calibrated
  candidate still said the test suite was updated to validate those behaviors.
- **Captured sanitization exposed false closure.** The output stopped inside
  `quality_issues` at line 100. It established diagnostic checks, not the complete
  sanitization implementation. Both legs asserted that sanitization was not a
  destructive filter. The candidate said the mechanism was "fully defined" by the
  rule set; its continuation then reasoned from that unsupported premise.
- **The environment case compressed well.** Both retained the observed active marker,
  environment location, and nonpersistent activation concern. The shorter candidate
  used 119 versus 172 memory tokens, without an important observed semantic loss.
- **More effort was not a remedy here.** Low effort still overclaimed and returned
  turn 3 for a turn-2 request once. The existing parser rejected it; unchanged context
  was counted rather than treating the failed compaction as savings.
- **Native event objects did not fix fidelity.** Both legs retained the same unsupported
  conclusions; candidate wording even claimed observation of unread quality/test changes.
  Its synthesis input grew from 9,090 to 9,268 tokens across the two cases. Escaping
  alone was not demonstrated to be the bottleneck.

## Evidence and Cost

Bundles below live under `eval_results/trajectory_compaction_ab/` (gitignored).
Each retains source events, prompts/hashes, raw responses, usage, and `comparison.md`.

| Bundle | Intervention | Memory tokens baseline / candidate |
| --- | --- | ---: |
| `20261027T223905Z` | Rich prompt + recovery; three captured turns | 668 / 699 |
| `20261027T224334Z` | Short evidence-first prompt; same captured turns | 668 / 425 |
| `20261027T224433Z` | Short prompt; five synthetic regressions | 994 / 837 |
| `20261027T225333Z` | Short prompt, low synthesis effort; two captured turns | 438 / 127 + one failure |
| `20261027T225753Z` | Rich prompt + calibration example; two captured turns | 496 / 413 |
| `20261027T230113Z` | Same calibrated prompt, native event objects | 467 / 364 |

The short prompt reduced memory by 36.4% on the three captured turns and 15.8% on
the synthetic set, but preserved false certainty and lost useful anchors. Those are
not quality wins. The low-effort run's 127 tokens exclude its failed synthesis and
must not be read as a compression ratio; the original failed turn remained intact.

These trials made 34 synthesis and 33 continuation requests, without extra judge
calls; one SDK-alias setup error made no requests. Hidden provider retries are not
independently counted. The calibration example did not solve evidence fidelity, so
further prompt expansion was stopped. The rejected candidate file was subsequently
removed; its exact prompts remain in experiment bundles and Git history.

Provider-reported synthesis usage totaled 106,036 input and 8,301 output tokens;
continuation usage is recorded separately in each bundle.

The next useful isolated experiment is a stronger synthesis model on these same two
failure cases, keeping the probe fixed. That is a hypothesis, not an implemented
model upgrade or a claim that more compute will solve it. Production stays unchanged.
