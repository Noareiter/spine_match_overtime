#!/usr/bin/env python3
"""Repair GP04 pre-fix annotation data: reopen trailing-blank lineages and fix last_seen_tp.

This script identifies and optionally repairs two categories of issues in data
produced by the annotation server before the bug-fix commits (bc59948, 0b93f0e):

1. Trailing-blank lineages (279 across 6 fovs, ~22.6%): a lineage has a real match
   at some earlier timepoint, then every later per_tp[tp] is {spine_id: null,
   fate: null, source: "cleared"|"returned_to_pool"} instead of fate: "lost".
   Repair: remove the lineage's anchor ID from spine_review_progress.json's
   reviewed_ids, and reset phase_index to 0 so Space-key navigation surfaces them.

2. Wrong last_seen_tp (37 across 6 fovs): a lineage-level summary field doesn't
   match the true last timepoint with a real match. These lineages are otherwise
   fully decided (no per-tp re-review needed).
   Repair: recompute and correct the field directly from per-tp data.

Usage:
  python scripts/repair_gp04_prefix_data.py <base_annotator_dir>
      (dry-run: report what would be changed, don't write anything)

  python scripts/repair_gp04_prefix_data.py <base_annotator_dir> --apply
      (execute: backup, apply all repairs, save files)

Example:
  python scripts/repair_gp04_prefix_data.py "E:/Noa/Pons - layer 5/Imaging/GP04/try/respan/_annotator" --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the app's own modules for loading/saving
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spine_annotator_backend import spine_lineage_store


def find_trailing_blank_lineages(
    lineages: List[dict]
) -> List[dict]:
    """Find lineages with trailing per_tp entries that are all blank (spine_id: null, fate: null)."""
    results: List[dict] = []
    for lin in lineages:
        per_tp = lin.get("per_tp") or {}
        lineage_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if not lineage_key:
            continue

        # Get timepoint_names from this lineage (not global)
        timepoint_names = lin.get("timepoint_names") or []
        if not timepoint_names:
            continue

        # Find the true last timepoint with a real match
        last_match_tp: Optional[str] = None
        for tp in timepoint_names:
            td = per_tp.get(tp) or {}
            if str(td.get("spine_id") or "").strip():
                last_match_tp = tp

        if last_match_tp is None:
            # No matches at all, skip
            continue

        # Check if all TPs after last_match_tp are blank
        trailing_blank_tps: List[str] = []
        for tp in timepoint_names[timepoint_names.index(last_match_tp) + 1 :]:
            td = per_tp.get(tp) or {}
            sid = str(td.get("spine_id") or "").strip()
            fate = str(td.get("fate") or "").strip()
            if not sid and not fate:
                trailing_blank_tps.append(tp)

        if trailing_blank_tps:
            results.append(
                {
                    "lineage_key": lineage_key,
                    "last_match_tp": last_match_tp,
                    "trailing_blank_tps": trailing_blank_tps,
                    "stored_last_seen_tp": str(lin.get("last_seen_tp") or ""),
                }
            )

    return results


def find_wrong_last_seen_tp(
    lineages: List[dict]
) -> List[dict]:
    """Find lineages where stored last_seen_tp doesn't match the true last match tp."""
    results: List[dict] = []
    for lin in lineages:
        per_tp = lin.get("per_tp") or {}
        lineage_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if not lineage_key:
            continue

        # Get timepoint_names from this lineage
        timepoint_names = lin.get("timepoint_names") or []
        if not timepoint_names:
            continue

        # Compute true last_seen_tp
        true_last_seen_tp: Optional[str] = None
        for tp in timepoint_names:
            td = per_tp.get(tp) or {}
            if str(td.get("spine_id") or "").strip():
                true_last_seen_tp = tp

        stored_last_seen_tp = str(lin.get("last_seen_tp") or "").strip()

        if true_last_seen_tp and stored_last_seen_tp != true_last_seen_tp:
            results.append(
                {
                    "lineage_key": lineage_key,
                    "true_last_seen_tp": true_last_seen_tp,
                    "stored_last_seen_tp": stored_last_seen_tp,
                }
            )

    return results


def generate_dry_run_report(
    base_path: Path,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], Dict[str, int]]:
    """Scan all fovs, return {fov: [...issues...]} for trailing blanks, wrong last_seen_tp, and counts."""
    trailing_blanks_by_fov: Dict[str, List[dict]] = {}
    wrong_last_seen_by_fov: Dict[str, List[dict]] = {}
    reviewed_id_removals: Dict[str, int] = {}

    # Explicitly find fov directories (not via glob to avoid Windows path issues)
    fov_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("fov")]
    for fov_dir in sorted(fov_dirs):
        fov_name = fov_dir.name

        dec_path = fov_dir / "lineage_decisions.json"
        if not dec_path.is_file():
            continue

        data = json.loads(dec_path.read_text(encoding="utf-8-sig"))
        lineages = data.get("lineages") or []

        if not lineages:
            continue

        # Find trailing blanks
        trailing = find_trailing_blank_lineages(lineages)
        if trailing:
            trailing_blanks_by_fov[fov_name] = trailing

            # Count how many would be removed from reviewed_ids
            prog_path = fov_dir / "spine_review_progress.json"
            if prog_path.is_file():
                prog = json.loads(prog_path.read_text(encoding="utf-8"))
                reviewed_ids = set(prog.get("reviewed_ids") or [])
                removals = sum(
                    1 for item in trailing if item["lineage_key"] in reviewed_ids
                )
                reviewed_id_removals[fov_name] = removals

        # Find wrong last_seen_tp (excluding those already in trailing_blanks)
        trailing_keys = set(item["lineage_key"] for item in trailing)
        wrong = [item for item in find_wrong_last_seen_tp(lineages)
                 if item["lineage_key"] not in trailing_keys]
        if wrong:
            wrong_last_seen_by_fov[fov_name] = wrong

    return trailing_blanks_by_fov, wrong_last_seen_by_fov, reviewed_id_removals


def print_dry_run_report(
    trailing_blanks: Dict[str, List[dict]],
    wrong_last_seen: Dict[str, List[dict]],
    reviewed_id_removals: Dict[str, int],
) -> None:
    """Pretty-print the dry-run report."""
    print("=" * 80)
    print("DRY-RUN REPAIR REPORT — GP04 Pre-Fix Data")
    print("=" * 80)
    print()

    total_trailing = sum(len(items) for items in trailing_blanks.values())
    total_wrong_last_seen = sum(len(items) for items in wrong_last_seen.values())

    print(f"Category 1: Trailing-blank lineages (needs re-review)")
    print(f"  Total across all fovs: {total_trailing}")
    print()

    for fov_name in sorted(trailing_blanks.keys()):
        items = trailing_blanks[fov_name]
        removals = reviewed_id_removals.get(fov_name, 0)
        print(f"  {fov_name}: {len(items)} lineages")
        print(f"    -> {removals} anchor IDs to remove from reviewed_ids")
        for item in items[:3]:  # Show first 3 as examples
            print(
                f"      {item['lineage_key']}: last_match={item['last_match_tp']}, "
                f"blank_at={', '.join(item['trailing_blank_tps'])}"
            )
        if len(items) > 3:
            print(f"      ... and {len(items) - 3} more")
        print()

    print(f"Category 2: Wrong last_seen_tp (auto-correct, no re-review)")
    print(f"  Total across all fovs: {total_wrong_last_seen}")
    print()

    for fov_name in sorted(wrong_last_seen.keys()):
        items = wrong_last_seen[fov_name]
        print(f"  {fov_name}: {len(items)} lineages")
        for item in items[:3]:  # Show first 3 as examples
            print(
                f"      {item['lineage_key']}: "
                f"stored={item['stored_last_seen_tp']!r} → true={item['true_last_seen_tp']!r}"
            )
        if len(items) > 3:
            print(f"      ... and {len(items) - 3} more")
        print()

    print("=" * 80)
    print(f"Summary:")
    print(f"  Trailing-blank lineages (need reopening): {total_trailing}")
    print(f"  Wrong last_seen_tp (auto-correct): {total_wrong_last_seen}")
    print(f"  Total lineages affected: {total_trailing + total_wrong_last_seen}")
    print()
    print("To apply these repairs, run with --apply flag:")
    print("  python scripts/repair_gp04_prefix_data.py <base_path> --apply")
    print("=" * 80)


def apply_repairs(
    base_path: Path,
    trailing_blanks: Dict[str, List[dict]],
    wrong_last_seen: Dict[str, List[dict]],
) -> None:
    """Apply all repairs: backup, modify files, save."""
    # Backup first
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = base_path.parent / f"{base_path.name}_backup_{timestamp}"
    print(f"Backing up to {backup_path}...")
    shutil.copytree(base_path, backup_path)
    print(f"  [OK] Backup complete")
    print()

    # Apply repairs per fov
    for fov_name in sorted(trailing_blanks.keys() | wrong_last_seen.keys()):
        fov_dir = base_path / fov_name
        if not fov_dir.is_dir():
            continue

        print(f"Repairing {fov_name}...")

        # Load current state
        dec_path = fov_dir / "lineage_decisions.json"
        prog_path = fov_dir / "spine_review_progress.json"

        data = json.loads(dec_path.read_text(encoding="utf-8-sig"))
        lineages = data.get("lineages") or []

        prog = (
            json.loads(prog_path.read_text(encoding="utf-8"))
            if prog_path.is_file()
            else spine_lineage_store.load_progress(fov_dir.parent, int(fov_name[3:]))
        )

        # Track modifications
        reviewed_ids_removed = 0
        last_seen_tp_corrected = 0

        # Category 1: Reopen trailing-blank lineages (progress file only)
        if fov_name in trailing_blanks:
            reviewed_ids = set(prog.get("reviewed_ids") or [])
            for item in trailing_blanks[fov_name]:
                if item["lineage_key"] in reviewed_ids:
                    reviewed_ids.discard(item["lineage_key"])
                    reviewed_ids_removed += 1

            prog["reviewed_ids"] = sorted(reviewed_ids)
            prog["phase_index"] = 0
            prog["last_pre_spine_index"] = 0

        # Category 2: Fix wrong last_seen_tp (lineage_decisions.json)
        if fov_name in wrong_last_seen:
            by_key = {
                lin.get("lineage_key") or lin.get("pre_spine_id"): lin
                for lin in lineages
            }
            for item in wrong_last_seen[fov_name]:
                lin = by_key.get(item["lineage_key"])
                if lin:
                    lin["last_seen_tp"] = item["true_last_seen_tp"]
                    last_seen_tp_corrected += 1

        # Save changes
        if reviewed_ids_removed > 0:
            prog["updated_at"] = datetime.now(timezone.utc).isoformat()
            prog_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
            print(f"  [OK] {reviewed_ids_removed} lineages reopened (removed from reviewed_ids, reset phase)")

        if last_seen_tp_corrected > 0:
            data["lineages"] = lineages
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            dec_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  [OK] {last_seen_tp_corrected} lineages' last_seen_tp corrected")

        if reviewed_ids_removed == 0 and last_seen_tp_corrected == 0:
            print(f"  (no changes needed)")

    print()
    print("=" * 80)
    print("Repairs complete. Backup available at:")
    print(f"  {backup_path}")
    print()
    print("Next: reload the animal in the app and verify:")
    print("  1. Reopened lineages appear via Space-key navigation with matches intact")
    print("  2. Previously-finished lineages are not re-surfaced")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair GP04 pre-fix annotation data (trailing-blank lineages, wrong last_seen_tp)",
    )
    parser.add_argument(
        "base_path",
        type=Path,
        help="Path to _annotator/ directory (e.g. E:/Noa/Pons - layer 5/Imaging/GP04/try/respan/_annotator)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs (backup first). Without this flag, only a dry-run report is printed.",
    )
    args = parser.parse_args()

    base_path = args.base_path.resolve()
    if not base_path.is_dir():
        print(f"Error: {base_path} not found or not a directory")
        return 1

    print(f"Scanning {base_path}...")
    print()

    trailing_blanks, wrong_last_seen, reviewed_id_removals = generate_dry_run_report(base_path)

    print_dry_run_report(trailing_blanks, wrong_last_seen, reviewed_id_removals)

    if args.apply:
        print()
        print("Applying repairs...")
        print()
        apply_repairs(base_path, trailing_blanks, wrong_last_seen)
        return 0
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
