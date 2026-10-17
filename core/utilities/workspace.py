from pathlib import Path


def workspace_directory(cwd: Path) -> Path:
    """Find the nearest workspace marker, falling back to the working directory."""
    cwd = cwd.expanduser().resolve()
    home = Path.home().resolve()
    for parent in (cwd, *cwd.parents):
        # The home marker stores global configuration, not a parent workspace.
        if parent == home and parent != cwd:
            continue
        marker = parent / ".quasipilot"
        if marker.is_dir():
            return marker
    return cwd / ".quasipilot"
