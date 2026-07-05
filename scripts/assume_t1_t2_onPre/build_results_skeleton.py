#!/usr/bin/env python3
"""
Pre-build the results/ folder skeleton expected by infer_baseline_bridged_pairs.py.

Point it at a respan folder (set in build_results_config.txt). It discovers the
FOVs from the timepoint Tables and creates, for every FOV:

    <respan>/results/fovN/<comparison>/

for all 10 chronological comparison folders. This gives manual annotation exports
and the inference run the exact layout they expect, so nothing fails on a missing
or misspelled folder.

Comparison folder names and timepoint order are imported from
infer_baseline_bridged_pairs.py (single source of truth).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Source of truth for timepoint order + canonical comparison folder names.
from infer_baseline_bridged_pairs import TP_ORDER, comparison_folder_name

CONFIG_NAME = "build_results_config.txt"
DETECTED_SPINES_RE = re.compile(r"^fov(\d+).*detected_spines.*\.csv$", re.IGNORECASE)


def chronological_comparisons() -> list[str]:
    """All 10 T1<T2 comparison folder names, in chronological order."""
    out: list[str] = []
    seen: set[str] = set()
    for i, t_a in enumerate(TP_ORDER):
        for t_b in TP_ORDER[i + 1 :]:
            name = comparison_folder_name(t_a, t_b)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def read_config(config_path: Path) -> dict[str, str]:
    """Parse simple 'key = value' lines (# comments and blanks ignored)."""
    cfg: dict[str, str] = {}
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip().lower()] = val.strip().strip('"')
    return cfg


def discover_fovs_from_respan(respan: Path) -> list[int]:
    """Find FOV numbers from <respan>/<timepoint>/Tables/fovN_detected_spines*.csv."""
    fovs: set[int] = set()
    for child in respan.iterdir():
        if not child.is_dir() or child.name.lower() == "results":
            continue
        tables = child / "Tables"
        if not tables.is_dir():
            continue
        for csv in tables.glob("*.csv"):
            m = DETECTED_SPINES_RE.match(csv.name)
            if m:
                fovs.add(int(m.group(1)))
    return sorted(fovs)


def main() -> int:
    here = Path(__file__).resolve().parent
    config_path = here / CONFIG_NAME
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}")
        print(f"       Create it with a line like: respan_path = E:\\...\\GP04\\respan")
        return 1

    cfg = read_config(config_path)
    respan_raw = cfg.get("respan_path") or cfg.get("respan")
    if not respan_raw or "xxx" in respan_raw.lower():
        print(f"ERROR: set a real 'respan_path = ...' in {config_path.name}")
        return 1

    respan = Path(respan_raw).expanduser()
    if not respan.is_dir():
        print(f"ERROR: respan path not found: {respan}")
        return 1

    # FOVs: explicit override in config, else auto-discover from timepoint Tables.
    if cfg.get("fovs"):
        fovs = sorted({int(x) for x in re.split(r"[\s,]+", cfg["fovs"].strip()) if x})
    else:
        fovs = discover_fovs_from_respan(respan)
    if not fovs:
        print(
            "ERROR: no FOVs found.\n"
            "       Looked for <respan>/<timepoint>/Tables/fovN_detected_spines*.csv.\n"
            f"       Set 'fovs = 1 2 3' in {config_path.name} to create them manually."
        )
        return 1

    comparisons = chronological_comparisons()
    results_root = respan / "results"

    print(f"respan:       {respan}")
    print(f"results root: {results_root}")
    print(f"FOVs:         {fovs}")
    print(f"comparisons:  {len(comparisons)} per FOV")
    print()

    created = 0
    for fov in fovs:
        fov_dir = results_root / f"fov{fov}"
        for comp in comparisons:
            comp_dir = fov_dir / comp
            if comp_dir.is_dir():
                continue
            comp_dir.mkdir(parents=True, exist_ok=True)
            created += 1
            print(f"  + {comp_dir.relative_to(respan)}")

    print()
    print(f"Done. Created {created} new folder(s); {len(fovs) * len(comparisons)} expected total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
