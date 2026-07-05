#!/usr/bin/env python3
"""Build global spine catalog (S_* IDs) for one FOV from detection CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend import animal_config, animal_layout, spine_catalog_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Build spine_catalog.csv for a FOV")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "annotator.json"),
        help="Path to annotator.json",
    )
    parser.add_argument("--fov", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if catalog exists")
    args = parser.parse_args()

    cfg = animal_config.load_config(args.config)
    if not cfg.workspace:
        print("ERROR: set workspace in config")
        return 1
    respan = animal_config.require_respan(cfg)
    fov = int(args.fov or cfg.default_fov or 1)
    available = animal_layout.discover_available_timepoints(respan, fov)
    selected = animal_config.resolve_active_timepoints(cfg, available)
    inv = animal_layout.build_fov_inventory(
        respan, fov, animal_id=cfg.animal_id, selected_timepoints=selected
    )
    csv_by_tp = {
        tp.name: Path(tp.csv_path)
        for tp in inv.timepoints
        if tp.csv_path
    }
    tps = [tp.name for tp in inv.timepoints if tp.csv_path]
    catalog, path, created = spine_catalog_store.ensure_catalog(
        respan,
        fov,
        animal_id=cfg.animal_id,
        timepoint_names=tps,
        csv_by_tp=csv_by_tp,
        rebuild=bool(args.rebuild),
    )
    print(f"Catalog: {path}")
    print(f"Rows: {len(catalog.rows)}  ({'created' if created else 'loaded'})")
    for tp in tps:
        n = sum(1 for r in catalog.rows if r["timepoint"] == tp)
        print(f"  {tp}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
