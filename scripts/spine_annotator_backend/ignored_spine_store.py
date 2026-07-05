"""Spine IDs marked ignore (inside OOF regions) — persisted per FOV."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from . import dendrite_link_store


def _path(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov) / "ignored_spines.json"


def load_ignored(respan: Path, fov: int) -> Dict[str, Set[str]]:
    p = _path(respan, fov)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Set[str]] = {}
    for tp, ids in (data.get("by_tp") or {}).items():
        out[str(tp)] = {str(x) for x in ids if str(x).strip()}
    return out


def save_ignored(
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
        "by_tp": {tp: sorted(ids) for tp, ids in sorted(by_tp.items())},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def add_ignored(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    by_tp_add: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    current = load_ignored(respan, fov)
    for tp, ids in by_tp_add.items():
        bucket = current.setdefault(str(tp), set())
        for sid in ids:
            if str(sid).strip():
                bucket.add(str(sid))
    save_ignored(respan, fov, current, animal_id=animal_id)
    return current


def is_ignored(by_tp: Dict[str, Set[str]], tp: str, spine_id: str) -> bool:
    return str(spine_id) in by_tp.get(tp, set())
