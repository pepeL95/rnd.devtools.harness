# Evaluations

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

Compaction runs also emit `memory.md` and an empty legacy `segmentation.md`
(there is no separate segmentation stage), while trajectory
compression runs emit `syntheses.json`.
