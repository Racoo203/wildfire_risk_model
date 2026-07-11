"""Figure manifest: every viz/ call appends an entry here, so the results
dashboard (Phase 5) can browse figures without re-globbing the filesystem
or guessing at filenames from strings (Section 9)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def append_to_manifest(figures_dir, path, category, generated_by, season=None, params=None):
    manifest_path = Path(figures_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing manifest at {manifest_path}; starting fresh.")
            entries = []

    rel_path = str(Path(path).relative_to(figures_dir)) if _is_relative(path, figures_dir) else str(path)

    # Upsert: a rerun of the same figure replaces its old entry instead of
    # duplicating it — the manifest reflects current state, not a log.
    entries = [e for e in entries if e["path"] != rel_path]
    entries.append({
        "path": rel_path,
        "category": category,
        "season": season,
        "generated_by": generated_by,
        "params": params or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    manifest_path.write_text(json.dumps(entries, indent=2))


def _is_relative(path: Path, base: Path) -> bool:
    try:
        Path(path).relative_to(base)
        return True
    except ValueError:
        return False