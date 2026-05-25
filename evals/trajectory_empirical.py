from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.common import ensure_dir, load_session, render_events, render_turn_table, timestamp_slug, write_json, write_jsonl
from core.trajectory.compactor import TrajectoryCompactor
from core.trajectory.policy import TrajectoryCompactionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run empirical trajectory compression on a captured session JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source session JSONL.")
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/trajectory_empirical"))
    parser.add_argument("--batch-size", type=int, default=2, help="Turn batch size N.")
    args = parser.parse_args()

    events = load_session(args.input)
    compactor = TrajectoryCompactor(policy=TrajectoryCompactionPolicy(trigger_every_turns=args.batch_size))
    result = compactor.compact(events)

    run_dir = ensure_dir(args.out_dir / args.input.stem / timestamp_slug())
    write_jsonl(run_dir / "source.jsonl", events)
    write_jsonl(run_dir / "rewritten.jsonl", result.events)
    write_json(
        run_dir / "syntheses.json",
        {
            "turn_syntheses": [
                {"turn": item.turn, "synthesis": item.synthesis, "live_edge": item.live_edge}
                for item in result.turn_syntheses
            ]
        },
    )
    write_json(
        run_dir / "metrics.json",
        {
            "input": str(args.input),
            "batch_size": args.batch_size,
            "compacted_turns": result.compacted_turns,
            "compacted_event_count": result.compacted_event_count,
            "output_event_count": len(result.events),
        },
    )
    synthesis_section = "\n\n".join(
        [
            f"### Turn {item.turn}\n\n{item.synthesis}\n\nLive edge: {item.live_edge}"
            for item in result.turn_syntheses
        ]
    )
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Trajectory Empirical Result",
                "",
                f"- Input: `{args.input}`",
                f"- Batch size: `{args.batch_size}`",
                f"- Compacted turns: `{result.compacted_turns}`",
                f"- Compacted internal event count: `{result.compacted_event_count}`",
                "",
                "## Rewritten Event Table",
                "",
                render_turn_table(result.events),
                "",
                "## Rewritten Events",
                "",
                render_events(result.events),
                "",
                "## Turn Syntheses",
                "",
                synthesis_section or "_No trajectory compaction output produced._",
            ]
        ),
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
