"""Reset per-FOV annotator metadata so review can start from scratch."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from . import dendrite_link_store

_ANNOTATOR_FILES = (
    "dendrite_links.json",
    "dendrite_links_wide.csv",
    "dendrite_link_progress.json",
    "lineage_decisions.json",
    "spine_review_progress.json",
    "spine_registry_wide.csv",
    "oof_segments.json",
    "ignored_spines.json",
    "timepoint_selection.json",
)

_DENDRITE_LINK_FILES = frozenset({
    "dendrite_links.json",
    "dendrite_links_wide.csv",
    "dendrite_link_progress.json",
})


def reset_fov_annotator(
    respan: Path,
    fov: int,
    *,
    backup: bool = True,
    keep_dendrite_links: bool = False,
) -> dict:
    """Clear saved lineage / OOF / review progress for one FOV.

    When keep_dendrite_links=True, dendrite link JSON/CSV and link progress are preserved.
    spine_catalog.csv is always kept (rebuilt from detection CSVs on load if stale).
    """
    meta = dendrite_link_store.annotator_meta_dir(respan, fov)
    if not meta.is_dir():
        return {
            "fov": fov,
            "removed": [],
            "kept": [],
            "backup_dir": "",
            "message": f"No annotator data for FOV {fov} — already fresh.",
        }

    removed: List[str] = []
    kept: List[str] = []
    failed: List[str] = []
    backup_dir = ""
    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = str(meta.parent / f"fov{fov}_backup_{stamp}")
        shutil.copytree(meta, backup_dir)

    for name in _ANNOTATOR_FILES:
        if keep_dendrite_links and name in _DENDRITE_LINK_FILES:
            if (meta / name).is_file():
                kept.append(name)
            continue
        path = meta / name
        if path.is_file():
            try:
                path.unlink()
                removed.append(name)
            except OSError as exc:
                failed.append(f"{name} ({exc})")

    if meta.is_dir() and not any(meta.iterdir()):
        meta.rmdir()

    mode = "spine review reset (dendrite links kept)" if keep_dendrite_links else "full annotator reset"
    msg = (
        f"FOV {fov} {mode}: {len(removed)} file(s) removed."
        + (f" Kept: {', '.join(kept)}." if kept else "")
        + (f" Backup: {backup_dir}" if backup_dir else "")
    )
    if failed:
        msg += f" Could not delete (close Excel/app): {', '.join(f.name if hasattr(f, 'name') else str(f) for f in failed)}."
        # failed entries are strings like "file (err)"
        failed_names = [f.split(" (", 1)[0] for f in failed]
        msg += f" Locked: {', '.join(failed_names)}."
    return {
        "fov": fov,
        "removed": removed,
        "kept": kept,
        "failed": failed,
        "backup_dir": backup_dir,
        "message": msg,
    }
