#!/usr/bin/env python3
"""Open Explorer only for folders newly created by bootstrap (skip existing)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend.animal_config import (
    bootstrap,
    list_newly_created_folders,
    load_config,
    resolve_paths,
)


def _open(path: Path) -> None:
    print(f"Open (new): {path}")
    os.startfile(str(path))  # noqa: S606


def main() -> int:
    cfg = load_config()
    if not cfg.workspace:
        print("ERROR: set workspace in config/annotator.json")
        return 1

    try:
        report = bootstrap(cfg)
        paths = resolve_paths(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Animal: {cfg.animal_id or '?'}")
    print(f"Workspace: {paths.get('workspace')}")
    print(f"Respan: {paths.get('respan_root')}")
    if report.created:
        print(f"Created {len(report.created)} folder(s) under respan/")
        for rel in report.created[:20]:
            print(f"  + {rel}")
        if len(report.created) > 20:
            print(f"  ... +{len(report.created) - 20} more")
    else:
        print("Layout OK — all folders already exist (no Explorer windows).")

    folders = list_newly_created_folders(report)
    if not folders:
        print("Skipping Explorer — nothing new to show.")
        return 0

    print(f"Opening {len(folders)} newly created folder(s)...")
    for folder in folders:
        _open(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
