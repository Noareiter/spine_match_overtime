"""Persist which timepoints are active for a given FOV (subset of the full series)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from . import dendrite_link_store


def _path(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov) / "timepoint_selection.json"


def load(respan: Path, fov: int) -> List[str]:
    p = _path(respan, fov)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [str(x).strip() for x in data.get("timepoints") or [] if str(x).strip()]


def save(respan: Path, fov: int, timepoints: List[str], *, animal_id: str = "") -> Path:
    p = _path(respan, fov)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "timepoints": list(timepoints),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def parse_timepoint_query(raw: str | None) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]
