"""Whole-dendrite ignore markers, per timepoint — persisted per FOV.

Ignoring a dendrite (in the Phase 0 dendrite-link step) means every spine
currently sitting on that dendrite at that timepoint is excluded from any
match candidacy — never suggested/ranked as a candidate, never offered in
the orphan-review queue — and is listed in the merged ignored_spines.csv
under source "dendrite".

Unlike ignored_spine_store (a snapshot list of specific spine ids), this
store records the (timepoint, dendrite_id) pair itself. Which spines are
affected is resolved live wherever needed (via spine_ids_on_ignored_dendrites),
so a spine added to an already-ignored dendrite later (e.g. after a CSV
re-run) is automatically covered too.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from . import dendrite_link_store


def _path(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov) / "ignored_dendrites.json"


def load_ignored_dendrites(respan: Path, fov: int) -> Dict[str, Set[str]]:
    p = _path(respan, fov)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Set[str]] = {}
    for tp, ids in (data.get("by_tp") or {}).items():
        out[str(tp)] = {str(x) for x in ids if str(x).strip()}
    return out


def save_ignored_dendrites(
    respan: Path,
    fov: int,
    by_tp: Dict[str, Set[str]],
    *,
    animal_id: str = "",
) -> Path:
    p = _path(respan, fov)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "by_tp": {tp: sorted(ids) for tp, ids in sorted(by_tp.items()) if ids},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def add_ignored_dendrite(
    respan: Path,
    fov: int,
    *,
    animal_id: str = "",
    timepoint: str,
    dendrite_id: str,
) -> Dict[str, Set[str]]:
    current = load_ignored_dendrites(respan, fov)
    bucket = current.setdefault(str(timepoint), set())
    bucket.add(str(dendrite_id))
    save_ignored_dendrites(respan, fov, current, animal_id=animal_id)
    return current


def remove_ignored_dendrite(
    respan: Path,
    fov: int,
    *,
    timepoint: str,
    dendrite_id: str,
) -> Dict[str, Set[str]]:
    current = load_ignored_dendrites(respan, fov)
    bucket = current.get(str(timepoint))
    if bucket:
        bucket.discard(str(dendrite_id))
    save_ignored_dendrites(respan, fov, current)
    return current


def is_dendrite_ignored(by_tp: Dict[str, Set[str]], tp: str, dendrite_id: str) -> bool:
    return str(dendrite_id) in by_tp.get(str(tp), set())


def spine_ids_on_ignored_dendrites(
    by_tp: Dict[str, Set[str]],
    tp: str,
    spine_lookup_for_tp: Dict[str, dict],
) -> Set[str]:
    """Global ids at `tp` whose dendrite is currently marked ignored."""
    ignored = by_tp.get(str(tp), set())
    if not ignored:
        return set()
    return {
        gid
        for gid, rec in spine_lookup_for_tp.items()
        if str(rec.get("dendrite_id") or "") in ignored
    }
