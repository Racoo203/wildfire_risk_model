"""Figure manifest: every viz/ call appends an entry here, so the results
dashboard (Phase 5) can browse figures without re-globbing the filesystem
or guessing at filenames from strings (Section 9)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def append_to_manifest(
    figures_dir: Path,
    path: Path,
    category: str,
    generated_by: str,
    season: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one figure's metadata to figures/manifest.json (created if absent)."""
    manifest_path = Path(figures_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Could not parse existing manifest at {manifest_path}; starting fresh.")
            entries = []

    entries.append({
        "path": str(Path(path).relative_to(figures_dir)) if _is_relative(path, figures_dir) else str(path),
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