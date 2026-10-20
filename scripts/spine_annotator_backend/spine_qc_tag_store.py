"""Per-spine-per-timepoint QC tags (artifact, ignore) — independent of lineages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from . import dendrite_link_store


def _path(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov) / "spine_qc_tags.json"


def load_tags(respan: Path, fov: int) -> Dict[str, Dict[str, str]]:
    """Load QC tags. Returns tp -> spine_id -> "artifact"|"ignore"."""
    p = _path(respan, fov)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for tp, tags in (data.get("by_tp") or {}).items():
        tp_str = str(tp)
        out[tp_str] = {}
        for spine_id, tag in (tags or {}).items():
            if tag in ("artifact", "ignore"):
                out[tp_str][str(spine_id)] = str(tag)
    return out


def save_tags(
    respan: Path,
    fov: int,
    by_tp: Dict[str, Dict[str, str]],
    *,
    animal_id: str = "",
) -> Path:
    """Save all QC tags for a FOV."""
    p = _path(respan, fov)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "by_tp": {tp: dict(tags) for tp, tags in sorted(by_tp.items())},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def save_tag(
    respan: Path,
    fov: int,
    *,
    timepoint: str,
    spine_id: str,
    tag: str,
    animal_id: str = "",
) -> None:
    """Set or overwrite a single QC tag."""
    current = load_tags(respan, fov)
    tp = str(timepoint)
    sid = str(spine_id)
    if tag in ("artifact", "ignore"):
        current.setdefault(tp, {})[sid] = tag
    else:
        if tp in current:
            current[tp].pop(sid, None)
    save_tags(respan, fov, current, animal_id=animal_id)


def clear_tag(
    respan: Path,
    fov: int,
    *,
    timepoint: str,
    spine_id: str,
    animal_id: str = "",
) -> None:
    """Remove a QC tag."""
    current = load_tags(respan, fov)
    tp = str(timepoint)
    sid = str(spine_id)
    if tp in current:
        current[tp].pop(sid, None)
        if not current[tp]:
            current.pop(tp, None)
    save_tags(respan, fov, current, animal_id=animal_id)


def get_tag(by_tp: Dict[str, Dict[str, str]], tp: str, spine_id: str) -> str:
    """Get tag for a spine, or empty string if none."""
    return str(by_tp.get(str(tp), {}).get(str(spine_id), ""))


def is_artifact(by_tp: Dict[str, Dict[str, str]], tp: str, spine_id: str) -> bool:
    """True if spine is tagged artifact at this TP."""
    return get_tag(by_tp, tp, spine_id) == "artifact"


def is_ignore(by_tp: Dict[str, Dict[str, str]], tp: str, spine_id: str) -> bool:
    """True if spine is tagged ignore at this TP."""
    return get_tag(by_tp, tp, spine_id) == "ignore"


def tagged_spines_at_tp(by_tp: Dict[str, Dict[str, str]], tp: str) -> Set[str]:
    """All spine IDs with any QC tag at this TP."""
    return set(by_tp.get(str(tp), {}).keys())
