"""In-session manual spine IDs (persisted via lineage_decisions / spine_registry_wide only)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set


def is_manual_id(spine_id: str) -> bool:
    return str(spine_id or "").strip().startswith("manual_")


def is_manual_entry(entry: dict) -> bool:
    sid = str(entry.get("spine_id") or "").strip()
    return is_manual_id(sid) or str(entry.get("source") or "") == "manual_added"


def _tp_slug(tp: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(tp).lower()).strip("_")
    return slug[:24] or "tp"


def make_manual_spine_id(timepoint: str, existing_ids: Iterable[str]) -> str:
    slug = _tp_slug(timepoint)
    prefix = f"manual_{slug}_"
    taken: Set[str] = {str(x).strip() for x in existing_ids if str(x).strip()}
    n = 1
    while True:
        sid = f"{prefix}{n:04d}"
        if sid not in taken:
            return sid
        n += 1


def to_lookup_row(
    *,
    spine_id: str,
    x: float,
    y: float,
    z: float,
    dendrite_id: str = "",
) -> dict:
    return {
        "spine_id": str(spine_id),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "dendrite_id": str(dendrite_id or ""),
    }


def collect_from_lineages(lineages: List[dict]) -> Dict[str, List[dict]]:
    """Rebuild manual spine lookup rows from saved lineage decisions."""
    by_tp: Dict[str, List[dict]] = {}
    seen: Set[tuple] = set()
    for lin in lineages or []:
        for tp, td in (lin.get("per_tp") or {}).items():
            td = dict(td or {})
            sid = str(td.get("spine_id") or "").strip()
            if not sid or not is_manual_entry(td):
                continue
            key = (str(tp), sid)
            if key in seen:
                continue
            seen.add(key)
            by_tp.setdefault(str(tp), []).append(
                to_lookup_row(
                    spine_id=sid,
                    x=float(td.get("x", 0)),
                    y=float(td.get("y", 0)),
                    z=float(td.get("z", 0)),
                    dendrite_id=str(td.get("dendrite_id") or ""),
                )
            )
    return by_tp
