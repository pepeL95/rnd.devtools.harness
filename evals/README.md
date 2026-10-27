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
python evals/trajectory_compaction_ab.py --candidate /path/to/proposed-prompt.txt
```

This uses the configured trajectory model and incurs 20 model requests: two prompts,
five cases, one repeat, with one continuation probe per synthesis. Both variants use
identical inputs, temperature, and thinking settings; request order alternates.
`--case-id exact_whitespace --repeats 1` runs a four-request pilot. The baseline is
frozen in `trajectory_compaction_baseline.txt`; required `--candidate` selects an experimental
prompt without changing production. No extra review stage or input-selection change
is introduced into the harness.

Run the captured-session comparison from the repository root (12 requests):

```sh
python evals/trajectory_compaction_ab.py --candidate /path/to/proposed-prompt.txt \
  --fixture evals/trajectory_compaction_captured_set.json
```

`--synthesis-thinking low` changes only synthesis effort; the probe retains its
configured default. `--structured-events` tests native event objects rather than
nested serialized strings for both prompts. These are eval-only ablations, not
harness configuration. Compare a single changed factor against its saved control;
neither alternative demonstrated a fidelity improvement in the recorded trials.

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
See [the initial synthetic findings](trajectory_compaction_findings.md) and
[the captured-session follow-up](trajectory_compaction_transfer_findings.md).
The rejected candidate was removed: smaller memories still overclaimed what the
source established. Its prompts remain in saved experiment bundles and Git history.
Production synthesis instructions have not changed.

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
