# Evaluations

## Trajectory Compaction A/B

Five short synthetic coding turns cover whitespace diagnosis, same-turn cancellation,
config precedence, stale publication, and a blocked embedding migration. Each has
a memory-only continuation question. A separate captured set selects three unmodified
turns from the checked-in session JSONL. The assistant referees semantic quality
directly against the source;
there is no regex rubric, numeric semantic score, or additional judge-model call.

```sh
conda activate quasipilot-harness
python evals/trajectory_compaction_ab.py
```

This uses the configured trajectory model and incurs 20 model requests: two variants,
five cases, one repeat, with one continuation probe per synthesis. Source events and
continuation questions are shared; request order alternates. The baseline reproduces
the old nested-JSON input, final-response inclusion, and JSON output contract. The
candidate uses current production extraction with decoded content and Markdown output.
`--case-id exact_whitespace --repeats 1` runs a four-request pilot. The baseline is
frozen in `trajectory_compaction_baseline.txt`; optional `--candidate /path/to/prompt.txt`
overrides the Markdown prompt without changing production. `--baseline-format markdown`
compares two prompts using the current mechanism instead of the legacy JSON adapter.

Run the captured-session comparison from the repository root (12 requests):

```sh
python evals/trajectory_compaction_ab.py \
  --fixture evals/trajectory_compaction_captured_set.json
```

These commands send fixture contents to the configured provider. Confirm authorization
before using captured sessions. The separate `trajectory_extraction_set.json` is a
synthetic tabs/regex/code-reuse and noisy-startup-log set (eight requests, or four
with `--case-id literal_tabbed_patch` or `--case-id noisy_log_extraction`).

`--synthesis-thinking low` changes only synthesis effort; the probe retains its
configured default. `--baseline-model` and `--candidate-model` override individual
synthesis models without changing the probe; `--candidate-thinking low` isolates
candidate effort. These are eval-only ablations, not
harness configuration. Compare a single changed factor against its saved control.

Timestamped `eval_results/trajectory_compaction_ab/` bundles retain fixtures, prompt
texts/hashes, source events, raw requests/responses, memories, continuation answers,
provider usage, and restored-context token estimates. `comparison.md` brings each
source, both memories, and continuation answers together for direct review.
Failures retain unchanged session context and cause a nonzero exit. A failed synthesis
has no continuation answer. Provider retries hidden by the SDK are not independently
metered. Read each pair against its source: which preserves the causal mechanism,
useful implementation details, uncertainty, and safe next action? Cite concrete
omissions or inventions, accept equivalent wording, and allow mixed verdicts.
Continuation answers are a resumability proxy, not measured coding-task performance.
Repeated temperature-zero outputs are not independent statistical evidence. Track
format failures and per-case weaknesses, not just smaller notes.
No-savings skips are counted separately, retain their original context-token count,
and give that retained source to the continuation probe; zero memory tokens do not
mean a successful compression. Inspect raw recorded synthesis output for skipped notes.
See [the initial synthetic findings](trajectory_compaction_findings.md) and
[the captured-session follow-up](trajectory_compaction_transfer_findings.md).
The rejected candidate was removed: smaller memories still overclaimed what the
source established. Its prompts remain in saved experiment bundles and Git history.
The current Markdown mechanism and remaining fidelity issues are reviewed in
[the extraction findings](trajectory_extraction_findings.md).

## Compaction Semantic Richness

The compaction semantic-richness eval measures whether the compacted context
preserves information an agent needs to resume accurately:

- user directives and corrections
- exact paths, commands, commits, and environment names
- failed approaches with mechanisms
- current state and validation status
- durable codebase facts with confidence markers

Scoring checks expected facts, not headings or document structure.

Run the full eval:

```sh
python evals/compaction_semantic_richness.py --critic-loops 1
```

Run a single case after a provider quota reset:

```sh
python evals/compaction_semantic_richness.py \
  --case-id setup_script_dependency_bootstrap \
  --critic-loops 1
```

Every run writes a new timestamped directory under
`eval_results/compaction_semantic_richness/`. The runner does not delete or
overwrite prior results.

## Empirical Session Rewrites

Run full-session compaction on a captured session:

```sh
python evals/compaction_empirical.py \
  --input evals/compaction/c2d0dc29c65a43ac94484c9c24fe0b23.jsonl
```

Run trajectory compression on a captured session:

```sh
python evals/trajectory_empirical.py \
  --input evals/trajectories/c2d0dc29c65a43ac94484c9c24fe0b23.jsonl
```

Each run writes a timestamped bundle under `eval_results/` containing:

- `source.jsonl`
- `rewritten.jsonl`
- `metrics.json`
- `summary.md`

Compaction runs also emit `memory.md` and extracted task episodes in `segmentation.md`, while trajectory
compression runs emit `syntheses.json`.
