# Evaluations

## Compaction Semantic Richness

The compaction semantic-richness eval measures whether the compacted context
preserves information an agent needs to resume accurately:

- user directives and corrections
- exact paths, commands, commits, and environment names
- failed approaches with mechanisms
- current state and validation status
- durable codebase facts with confidence markers

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
