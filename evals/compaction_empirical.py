from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import ensure_dir, load_session, render_events, render_turn_table, timestamp_slug, write_json, write_jsonl
from core.compaction.compactor import Compactor
from core.compaction.policy import CompactionPolicy
from core.compaction.token_counter import TokenCounter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run empirical full-session compaction on a captured session JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source session JSONL.")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/compaction_empirical"))
    parser.add_argument("--trigger-tokens", type=int, default=1, help="Override trigger threshold for evaluation.")
    parser.add_argument("--keep-last-turns", type=int, default=5)
    parser.add_argument("--critic-loops", type=int, default=1)
    args = parser.parse_args()

    events = load_session(args.input)
    token_counter = TokenCounter()
    token_estimate = token_counter.count_events(events)
    compactor = Compactor(
        policy=CompactionPolicy(
            trigger_tokens=args.trigger_tokens,
            keep_last_turns=args.keep_last_turns,
            max_critic_loops=args.critic_loops,
        )
    )
    result = compactor.compact(events, token_estimate=token_estimate)

    run_dir = ensure_dir(args.out_dir / args.input.stem / timestamp_slug())
    write_jsonl(run_dir / "source.jsonl", events)
    write_jsonl(run_dir / "rewritten.jsonl", result.events)
    (run_dir / "memory.md").write_text(result.memory_document, encoding="utf-8")
    (run_dir / "segmentation.md").write_text(result.segmentation, encoding="utf-8")
    write_json(
        run_dir / "metrics.json",
        {
            "input": str(args.input),
            "token_estimate_before": result.token_estimate_before,
            "compacted_event_count": result.compacted_event_count,
            "retained_event_count": result.retained_event_count,
            "revisions": result.revisions,
            "critique_count": len(result.critiques),
            "output_event_count": len(result.events),
        },
    )
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Compaction Empirical Result",
                "",
                f"- Input: `{args.input}`",
                f"- Token estimate before: `{result.token_estimate_before}`",
                f"- Compacted event count: `{result.compacted_event_count}`",
                f"- Retained event count: `{result.retained_event_count}`",
                f"- Revisions: `{result.revisions}`",
                "",
                "## Rewritten Event Table",
                "",
                render_turn_table(result.events),
                "",
                "## Rewritten Events",
                "",
                render_events(result.events),
                "",
                "## Memory Document",
                "",
                result.memory_document or "_No compaction output produced._",
            ]
        ),
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
