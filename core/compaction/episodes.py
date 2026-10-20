from __future__ import annotations

import re


def split_task_episodes(text: str) -> list[str]:
    """Use task headings as soft boundaries; preserve unstructured output intact."""
    episodes, lines = [], []
    fence = ""
    seen_heading = False
    for line in text.strip().splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            delimiter = marker[1]
            if not fence:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = ""
        if not fence and re.match(r"^##\s+\S", line):
            if seen_heading:
                episodes.append("\n".join(lines).strip())
                lines = []
            seen_heading = True
        lines.append(line)
    if lines:
        episodes.append("\n".join(lines).strip())
    return episodes
