"""Reconstruct the survival-analysis columns spine_registry_wide.csv no longer
carries directly. These are derived classifications, not raw observations, so
they were moved out of the annotator's primary output (see CLAUDE_CODE_BRIEF.md
Task 4) -- the annotator's job is to record observations, interpretation is a
post-processing concern.

Reads lineage_decisions.json (the untrimmed, authoritative source -- unaffected
by the registry column removal) and reproduces exactly the same values the
registry builder used to compute inline: event_<tp>, formation_tp, lifecycle,
right_censored, censored_from_tp, n_timepoints_seen, n_timepoints_continuous,
and active_timepoints. Join the output to spine_registry_wide.csv on
lineage_id (or lineage_key) to get the full original column set back.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from . import spine_lineage_store as sls


def build_derived_rows(respan: Path, fov: int) -> Tuple[List[str], List[Dict[str, str]]]:
    data = sls.load_decisions(respan, fov)
    lineages: List[dict] = list(data.get("lineages") or [])
    timepoint_names = sls._ordered_union_timepoints(lineages)
    header = (
        ["lineage_id", "lineage_key"]
        + [f"event_{tp}" for tp in timepoint_names]
        + [
            "formation_tp",
            "lifecycle",
            "right_censored",
            "censored_from_tp",
            "n_timepoints_seen",
            "n_timepoints_continuous",
            "active_timepoints",
        ]
    )
    rows: List[Dict[str, str]] = []
    for lin in lineages:
        pre_id = str(lin.get("pre_spine_id", "") or "")
        lineage_key = str(lin.get("lineage_key") or pre_id or "")
        per_tp = dict(lin.get("per_tp") or {})
        row_tps = list(lin.get("timepoint_names") or timepoint_names)
        events_by_tp, formation_tp, _termination_tp, lifecycle = sls._derive_events_and_lifecycle(
            row_tps, per_tp
        )
        _first_seen_tp, _last_seen_tp, censored_from_tp, right_censored = sls._lineage_summary_fields(
            lin, per_tp, row_tps
        )
        if right_censored and lifecycle not in ("stable", "transient", "persistent_engram"):
            lifecycle = "right_censored"
        seen = 0
        continuous_n = 0
        for tp in timepoint_names:
            td = per_tp.get(tp) or {}
            status = sls.status_for_tp_data(td) if tp in per_tp else ""
            if status in ("matched", "manual", "new"):
                seen += 1
            if tp in per_tp and sls._lineage_continuous_present(td):
                continuous_n += 1
        row: Dict[str, str] = {
            "lineage_id": f"L_{lineage_key.replace(' ', '_')}",
            "lineage_key": lineage_key,
            "formation_tp": formation_tp,
            "lifecycle": lifecycle,
            "right_censored": "1" if right_censored else "0",
            "censored_from_tp": censored_from_tp,
            "n_timepoints_seen": str(seen),
            "n_timepoints_continuous": str(continuous_n),
            "active_timepoints": ";".join(row_tps),
        }
        for tp in timepoint_names:
            row[f"event_{tp}"] = str(events_by_tp.get(tp) or "") if tp in row_tps else ""
        rows.append(row)
    return header, rows


def write_derived_csv(respan: Path, fov: int, out_path: Path) -> Path:
    header, rows = build_derived_rows(respan, fov)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("respan", type=Path, help="Path to the respan/ directory")
    parser.add_argument("fov", type=int)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: _annotator/fovN/spine_registry_derived.csv)",
    )
    args = parser.parse_args()
    out = args.out or (sls._meta_dir(args.respan, args.fov) / "spine_registry_derived.csv")
    path = write_derived_csv(args.respan, args.fov, out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
