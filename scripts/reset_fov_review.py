#!/usr/bin/env python3
"""Reset spine review for one FOV (optionally keep dendrite links)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend import animal_config, fov_reset


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset FOV annotator review data")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "annotator.json"),
    )
    parser.add_argument("--fov", type=int, default=1)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--keep-dendrite-links",
        action="store_true",
        help="Preserve dendrite_links.json and link progress",
    )
    args = parser.parse_args()

    cfg = animal_config.load_config(args.config)
    respan = animal_config.require_respan(cfg)
    result = fov_reset.reset_fov_annotator(
        respan,
        int(args.fov),
        backup=not args.no_backup,
        keep_dendrite_links=bool(args.keep_dendrite_links),
    )
    print(result["message"])
    if result.get("removed"):
        print("Removed:", ", ".join(result["removed"]))
    if result.get("kept"):
        print("Kept:", ", ".join(result["kept"]))
    if result.get("failed"):
        print("Failed:", result["failed"])
    if result.get("backup_dir"):
        print("Backup:", result["backup_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
