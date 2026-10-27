# Evaluations

## Trajectory Compaction A/B

Five short synthetic coding turns cover whitespace diagnosis, same-turn cancellation,
config precedence, stale publication, and a blocked embedding migration. Each has
a memory-only continuation question. The fixtures are not captured production
sessions. The assistant referees semantic quality directly against the source;
there is no regex rubric, numeric semantic score, or additional judge-model call.

```sh
conda activate quasipilot-harness
python evals/trajectory_compaction_ab.py --repeats 2
```

This uses the configured trajectory model and incurs 40 model requests: two prompts,
five cases, two repeats, with one continuation probe per synthesis. Both variants use
identical inputs, temperature, and thinking settings; request order alternates.
`--case-id exact_whitespace --repeats 1` runs a four-request pilot. The baseline is
frozen in `trajectory_compaction_baseline.txt`; `--candidate` selects an experimental
prompt without changing production. No extra review stage or input-selection change
is introduced into the harness.

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
See [the initial A/B findings](trajectory_compaction_findings.md). The candidate
remains experimental; production synthesis instructions have not changed. The
semantic review supersedes the original keyword-count interpretation.

## Trajectory Bank Retrieval

Run the real local embedding model against positive and close-topic negative
queries (first use downloads the model):

```sh
python evals/trajectory_bank_retrieval.py
```

Add `--live-compaction` to also exercise the configured compaction LLM, real
Chroma archival, session publication, reopening, and exact retrieval on synthetic
turns. This optional mode uses provider credentials and incurs model usage.
Results go to `eval_results/trajectory_bank_retrieval/<timestamp>/metrics.json`;
live runs also retain source/rewritten sessions, the bank, and `memory.md`.
The small synthetic set is a smoke gate, not a production threshold calibration.

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
