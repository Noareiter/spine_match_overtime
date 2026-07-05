#!/usr/bin/env python3
"""Create a minimal local demo workspace (10 spines, 3 timepoints) for logic testing."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORKSPACE = PROJECT_ROOT / "results" / "demo_spine_logic_lab"
RESPAN = WORKSPACE / "respan"
ANIMAL_ID = "DEMO10"
FOV = 1
TIMEPOINTS = ("pre-droplet", "end-droplet", "end-lever")
N_SPINES = 10
RNG = random.Random(42)


def _spine_table() -> pd.DataFrame:
    """Same spine IDs and coordinates at every timepoint (phase-1-only demo)."""
    rows = []
    for sid in range(1, N_SPINES + 1):
        dendrite_id = "1" if sid <= 5 else "2"
        rows.append(
            {
                "spine_id": sid,
                "x": round(120.0 + sid * 72.0, 3),
                "y": round(180.0 + (sid % 5) * 95.0, 3),
                "z": float((sid % 4) * 3 + 1),
                "dendrite_id": dendrite_id,
                "spine_vol": round(RNG.uniform(0.5, 4.0), 3),
                "head_vol": round(RNG.uniform(0.2, 2.0), 3),
            }
        )
    return pd.DataFrame(rows)


def _write_tiff(path: Path, spines: pd.DataFrame) -> None:
    h, w, z_planes = 600, 900, 8
    stack = np.zeros((z_planes, h, w), dtype=np.float32)
    stack += np.random.uniform(0.02, 0.08, size=stack.shape).astype(np.float32)
    for _, row in spines.iterrows():
        zi = int(np.clip(round(float(row["z"])), 0, z_planes - 1))
        xi = int(np.clip(round(float(row["x"])), 2, w - 3))
        yi = int(np.clip(round(float(row["y"])), 2, h - 3))
        stack[zi, yi - 2 : yi + 3, xi - 2 : xi + 3] += 0.85
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, stack)


def _write_dendrite_links(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    links = [
        {
            "link_id": "link_dendrite_1",
            "members": {tp: ["1"] for tp in TIMEPOINTS},
            "notes": "Demo dendrite 1 (spines 1-5)",
        },
        {
            "link_id": "link_dendrite_2",
            "members": {tp: ["2"] for tp in TIMEPOINTS},
            "notes": "Demo dendrite 2 (spines 6-10)",
        },
    ]
    payload = {
        "animal_id": ANIMAL_ID,
        "fov": str(FOV),
        "timepoint_names": list(TIMEPOINTS),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "links": links,
    }
    (meta_dir / "dendrite_links.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (meta_dir / "dendrite_link_progress.json").write_text(
        json.dumps({"visited_link_ids": [], "updated_at": payload["updated_at"]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print(f"Creating demo workspace at:\n  {WORKSPACE}\n")
    base_spines = _spine_table()
    for tp in TIMEPOINTS:
        tp_dir = RESPAN / tp
        tables = tp_dir / "Tables"
        tables.mkdir(parents=True, exist_ok=True)
        df = base_spines.copy()
        csv_path = tables / f"fov{FOV}_detected_spines_demo.csv"
        df.to_csv(csv_path, index=False)
        tiff_path = tp_dir / f"fov{FOV}.tif"
        _write_tiff(tiff_path, df)
        print(f"  {tp}: {len(df)} spines -> {csv_path.name}")

    _write_dendrite_links(RESPAN / "_annotator" / f"fov{FOV}")
    # clean annotator state except dendrite links
    annotator = RESPAN / "_annotator" / f"fov{FOV}"
    for name in (
        "lineage_decisions.json",
        "spine_review_progress.json",
        "spine_registry_wide.csv",
        "oof_segments.json",
        "ignored_spines.json",
        "timepoint_selection.json",
    ):
        p = annotator / name
        if p.is_file():
            try:
                p.unlink()
            except OSError as exc:
                print(f"  Warning: could not remove {name} ({exc})")

    print("\nDone. (Phase 1 only — identical spines 1-10 at all timepoints)")
    print("Launch: start_demo_annotator.bat")


if __name__ == "__main__":
    main()
