"""QC phases: duplicate conflict detection, disposition ledger, unreviewed sweep."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import dendrite_link_store, ignored_spine_store, spine_catalog_store, spine_lineage_store

DISPOSITION_FILENAME = "spine_disposition.csv"
RESOLUTIONS_FILENAME = "duplicate_resolutions.json"

DISPOSITION_UNREVIEWED = "unreviewed"
DISPOSITION_DUPLICATE = "duplicate_unresolved"
DISPOSITION_POOL_IGNORE = "pool_ignore"
DISPOSITION_LINEAGE_MATCHED = "lineage_matched"
DISPOSITION_LINEAGE_NEW = "lineage_new"
DISPOSITION_LINEAGE_LOST = "lineage_lost"
DISPOSITION_LINEAGE_IGNORE = "lineage_ignore"
DISPOSITION_LINEAGE_ARTIFACT = "lineage_artifact"
DISPOSITION_RESOLVED = "duplicate_resolved"


def _meta_dir(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov)


def disposition_path(respan: Path, fov: int) -> Path:
    return _meta_dir(respan, fov) / DISPOSITION_FILENAME


def resolutions_path(respan: Path, fov: int) -> Path:
    return _meta_dir(respan, fov) / RESOLUTIONS_FILENAME


def conflict_id(timepoint: str, spine_id: str) -> str:
    return f"{timepoint}|{spine_id}"


def parse_queue_id(queue_id: str) -> Tuple[str, str]:
    parts = str(queue_id or "").split("|", 1)
    if len(parts) != 2:
        return "", str(queue_id or "")
    return parts[0], parts[1]


def load_duplicate_resolutions(respan: Path, fov: int) -> dict:
    p = resolutions_path(respan, fov)
    if not p.is_file():
        return {"resolutions": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"resolutions": dict(data.get("resolutions") or {})}


def save_duplicate_resolution(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    timepoint: str,
    spine_id: str,
    canonical_lineage_key: str,
) -> Path:
    p = resolutions_path(respan, fov)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_duplicate_resolutions(respan, fov)
    resolutions = dict(data.get("resolutions") or {})
    cid = conflict_id(timepoint, spine_id)
    resolutions[cid] = {
        "timepoint": timepoint,
        "spine_id": spine_id,
        "canonical_lineage_key": canonical_lineage_key,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "resolutions": resolutions,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _assignments_by_tp(lineages: List[dict], timepoint_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """tp -> spine_id -> [lineage_key, ...]."""
    out: Dict[str, Dict[str, List[str]]] = {tp: {} for tp in timepoint_names}
    for lin in lineages:
        key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if not key:
            continue
        per_tp = lin.get("per_tp") or {}
        for tp in timepoint_names:
            td = dict(per_tp.get(tp) or {})
            sid = str(td.get("spine_id") or "").strip()
            if not sid:
                continue
            out.setdefault(tp, {}).setdefault(sid, []).append(key)
    return out


def _lineages_by_key(lineages: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for lin in lineages:
        key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if key:
            out[key] = lin
    return out


def comparison_timepoint_for_lineages(
    lineages_by_key: Dict[str, dict],
    lineage_keys: List[str],
    timepoint_names: List[str],
) -> str:
    """Best TP to compare two lineages visually (first TP where both differ)."""
    if not timepoint_names:
        return ""
    keys = [str(k or "").strip() for k in lineage_keys if str(k or "").strip()]
    if len(keys) < 2:
        return timepoint_names[0]
    lin_a = lineages_by_key.get(keys[0]) or {}
    lin_b = lineages_by_key.get(keys[1]) or {}
    per_a = lin_a.get("per_tp") or {}
    per_b = lin_b.get("per_tp") or {}
    for tp in timepoint_names:
        sid_a = str((per_a.get(tp) or {}).get("spine_id") or "").strip()
        sid_b = str((per_b.get(tp) or {}).get("spine_id") or "").strip()
        if sid_a and sid_b and sid_a != sid_b:
            return tp
    for tp in timepoint_names:
        sid_a = str((per_a.get(tp) or {}).get("spine_id") or "").strip()
        sid_b = str((per_b.get(tp) or {}).get("spine_id") or "").strip()
        if sid_a and sid_b:
            return tp
    anchor_a = str(lin_a.get("pre_timepoint") or "").strip()
    anchor_b = str(lin_b.get("pre_timepoint") or "").strip()
    for tp in timepoint_names:
        if tp in (anchor_a, anchor_b):
            return tp
    return timepoint_names[0]


def find_duplicate_conflicts(
    respan: Path,
    fov: int,
    timepoint_names: List[str],
) -> List[dict]:
    data = spine_lineage_store.load_decisions(respan, fov)
    lineages = list(data.get("lineages") or [])
    by_key = _lineages_by_key(lineages)
    resolutions = load_duplicate_resolutions(respan, fov).get("resolutions") or {}
    by_tp = _assignments_by_tp(lineages, timepoint_names)
    conflicts: List[dict] = []
    seen: Set[str] = set()
    for tp in timepoint_names:
        for sid, keys in sorted((by_tp.get(tp) or {}).items()):
            uniq = sorted(set(keys))
            if len(uniq) < 2:
                continue
            cid = conflict_id(tp, sid)
            if cid in resolutions or cid in seen:
                continue
            seen.add(cid)
            comp_tp = comparison_timepoint_for_lineages(by_key, uniq, timepoint_names)
            conflicts.append(
                {
                    "conflict_id": cid,
                    "timepoint": tp,
                    "comparison_timepoint": comp_tp,
                    "spine_id": sid,
                    "lineage_keys": uniq[:2] if len(uniq) == 2 else uniq,
                    "lineage_count": len(uniq),
                }
            )
    return conflicts


def positions_from_lineage(
    lineage: Optional[dict],
    timepoint_names: List[str],
    spine_lookup: Dict[str, Dict[str, dict]],
) -> Dict[str, dict]:
    if not lineage:
        return {tp: {} for tp in timepoint_names}
    per_tp = dict(lineage.get("per_tp") or {})
    out: Dict[str, dict] = {}
    for tp in timepoint_names:
        td = dict(per_tp.get(tp) or {})
        sid = str(td.get("spine_id") or "").strip()
        lookup = spine_lookup.get(tp) or {}
        rec = lookup.get(sid) if sid else None
        x = td.get("x")
        y = td.get("y")
        z = td.get("z")
        if rec is not None:
            x = float(rec.get("x", x or 0))
            y = float(rec.get("y", y or 0))
            z = float(rec.get("z", z or 0))
        elif x is not None and y is not None:
            x, y = float(x), float(y)
            z = float(z or 0)
        else:
            x = y = z = None
        pos = {
            "spine_id": sid or None,
            "x": x,
            "y": y,
            "z": z,
            "fate": td.get("fate"),
            "source": str(td.get("source") or "saved"),
            "local_spine_id": str(td.get("local_spine_id") or (rec or {}).get("local_spine_id") or ""),
        }
        for k in ("artifact_mode", "removed_spine_id", "continuity", "decision_scope", "fate_locked"):
            if k in td and td[k] is not None:
                pos[k] = td[k]
        if rec is not None:
            pos["dendrite_id"] = str(rec.get("dendrite_id") or "")
        out[tp] = pos
    return out


def _disposition_from_status(status: str, fate: str) -> str:
    st = str(status or "").strip().lower()
    ft = str(fate or "").strip().lower()
    if st in ("ignored",) or ft == "ignore":
        return DISPOSITION_LINEAGE_IGNORE
    if st in ("artifact",) or ft == "artifact":
        return DISPOSITION_LINEAGE_ARTIFACT
    if st in ("lost",) or ft == "lost":
        return DISPOSITION_LINEAGE_LOST
    if st in ("new",) or ft == "new":
        return DISPOSITION_LINEAGE_NEW
    if st in ("matched", "manual", "stable"):
        return DISPOSITION_LINEAGE_MATCHED
    if st == "single_tp_only" and ft == "ignore":
        return DISPOSITION_LINEAGE_IGNORE
    return ""


def build_disposition_rows(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    timepoint_names: List[str],
    catalog_rows: Optional[List[dict]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
) -> Tuple[List[dict], dict]:
    if catalog_rows is None:
        cat_path = spine_catalog_store.catalog_path(respan, fov)
        if cat_path.is_file():
            with cat_path.open(newline="", encoding="utf-8-sig") as fh:
                catalog_rows = list(csv.DictReader(fh))
        else:
            catalog_rows = []
    if ignored_by_tp is None:
        ignored_by_tp = ignored_spine_store.load_ignored(respan, fov)

    data = spine_lineage_store.load_decisions(respan, fov)
    lineages = list(data.get("lineages") or [])
    resolutions = load_duplicate_resolutions(respan, fov).get("resolutions") or {}
    by_tp = _assignments_by_tp(lineages, timepoint_names)
    anchor_keys: Dict[str, Set[str]] = {tp: set() for tp in timepoint_names}
    for lin in lineages:
        anchor = str(lin.get("pre_timepoint") or "").strip()
        key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if anchor in anchor_keys and key:
            anchor_keys[anchor].add(key)

    rows: List[dict] = []
    unreviewed = 0
    duplicate_unresolved = 0
    for cat in catalog_rows:
        gid = str(cat.get("global_spine_id") or "").strip()
        tp = str(cat.get("timepoint") or "").strip()
        if not gid or tp not in timepoint_names:
            continue
        cid = conflict_id(tp, gid)
        keys = list(by_tp.get(tp, {}).get(gid) or [])
        disposition = DISPOSITION_UNREVIEWED
        lineage_key = ""
        status = ""
        fate = ""
        event = ""
        conflict = "no"
        reviewed = "no"

        if len(keys) > 1 and cid not in resolutions:
            disposition = DISPOSITION_DUPLICATE
            conflict = "yes"
            duplicate_unresolved += 1
        elif gid in (ignored_by_tp.get(tp) or set()):
            disposition = DISPOSITION_POOL_IGNORE
            reviewed = "yes"
        elif keys:
            primary = keys[0]
            if cid in resolutions:
                primary = str(resolutions[cid].get("canonical_lineage_key") or primary)
                disposition = DISPOSITION_RESOLVED
            lin = spine_lineage_store.get_lineage_by_key(respan, fov, primary)
            if lin:
                td = dict((lin.get("per_tp") or {}).get(tp) or {})
                status = spine_lineage_store.status_for_tp_data(td)
                fate = str(td.get("fate") or "")
                if str(td.get("spine_id") or "").strip() == gid:
                    mapped = _disposition_from_status(status, fate)
                    if mapped:
                        disposition = mapped
                        lineage_key = primary
                        reviewed = "yes"
                elif gid in anchor_keys.get(tp, set()):
                    td_anchor = dict((lin.get("per_tp") or {}).get(tp) or {})
                    status = spine_lineage_store.status_for_tp_data(td_anchor)
                    fate = str(td_anchor.get("fate") or "")
                    mapped = _disposition_from_status(status, fate)
                    disposition = mapped or DISPOSITION_LINEAGE_IGNORE
                    lineage_key = gid
                    reviewed = "yes"
        elif gid in anchor_keys.get(tp, set()):
            lin = spine_lineage_store.get_lineage_by_key(respan, fov, gid)
            if lin:
                td = dict((lin.get("per_tp") or {}).get(tp) or {})
                status = spine_lineage_store.status_for_tp_data(td)
                fate = str(td.get("fate") or "")
                mapped = _disposition_from_status(status, fate)
                disposition = mapped or DISPOSITION_LINEAGE_IGNORE
                lineage_key = gid
                reviewed = "yes"

        if disposition == DISPOSITION_UNREVIEWED:
            unreviewed += 1

        rows.append(
            {
                "animal_id": animal_id,
                "fov": str(fov),
                "global_spine_id": gid,
                "timepoint": tp,
                "local_spine_id": str(cat.get("local_spine_id") or ""),
                "disposition": disposition,
                "lineage_key": lineage_key,
                "status": status,
                "fate": fate,
                "event": event,
                "conflict": conflict,
                "reviewed": reviewed,
            }
        )

    summary = {
        "catalog_total": len(rows),
        "tagged": len(rows) - unreviewed,
        "unreviewed": unreviewed,
        "duplicate_unresolved": duplicate_unresolved,
    }
    return rows, summary


def rebuild_spine_disposition(
    respan: Path,
    fov: int,
    *,
    animal_id: str = "",
    timepoint_names: Optional[List[str]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
) -> Tuple[Path, dict]:
    data = spine_lineage_store.load_decisions(respan, fov)
    aid = str(animal_id or data.get("animal_id") or "")
    tps = list(
        timepoint_names
        or spine_lineage_store._ordered_union_timepoints(data.get("lineages") or [])
    )
    rows, summary = build_disposition_rows(
        respan,
        fov,
        animal_id=aid,
        timepoint_names=tps,
        ignored_by_tp=ignored_by_tp,
    )
    header = [
        "animal_id",
        "fov",
        "global_spine_id",
        "timepoint",
        "local_spine_id",
        "disposition",
        "lineage_key",
        "status",
        "fate",
        "event",
        "conflict",
        "reviewed",
    ]
    out = disposition_path(respan, fov)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out, summary


def unreviewed_queue(
    respan: Path,
    fov: int,
    timepoint_names: List[str],
    ignored_by_tp: Optional[Dict[str, set]] = None,
    *,
    disposition_rows: Optional[List[dict]] = None,
) -> List[dict]:
    rows = disposition_rows
    if rows is None:
        rows, _ = build_disposition_rows(
            respan,
            fov,
            animal_id="",
            timepoint_names=timepoint_names,
            ignored_by_tp=ignored_by_tp,
        )
    out: List[dict] = []
    for row in rows:
        if row.get("disposition") != DISPOSITION_UNREVIEWED:
            continue
        gid = str(row["global_spine_id"])
        tp = str(row["timepoint"])
        out.append(
            {
                "queue_id": conflict_id(tp, gid),
                "timepoint": tp,
                "global_spine_id": gid,
                "local_spine_id": str(row.get("local_spine_id") or ""),
            }
        )
    return out


def _disposition_cache_valid(respan: Path, fov: int) -> bool:
    meta = spine_lineage_store.paths(respan, fov)
    disp = disposition_path(respan, fov)
    if not disp.is_file() or not meta["decisions"].is_file():
        return False
    try:
        return disp.stat().st_mtime >= meta["decisions"].stat().st_mtime
    except OSError:
        return False


def _summary_from_disposition_rows(rows: List[dict]) -> dict:
    unreviewed = sum(1 for r in rows if r.get("disposition") == DISPOSITION_UNREVIEWED)
    duplicate_unresolved = sum(
        1 for r in rows if r.get("disposition") == DISPOSITION_DUPLICATE
    )
    tagged = len(rows) - unreviewed
    return {
        "catalog_total": len(rows),
        "tagged": tagged,
        "unreviewed": unreviewed,
        "duplicate_unresolved": duplicate_unresolved,
    }


def _load_disposition_rows_from_cache(respan: Path, fov: int) -> List[dict]:
    p = disposition_path(respan, fov)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def coverage_and_unreviewed(
    respan: Path,
    fov: int,
    timepoint_names: List[str],
    ignored_by_tp: Optional[Dict[str, set]] = None,
    *,
    animal_id: str = "",
    rebuild: bool = False,
) -> Tuple[List[dict], dict]:
    """Return disposition rows + summary, using on-disk cache when fresh."""
    if not rebuild and _disposition_cache_valid(respan, fov):
        rows = _load_disposition_rows_from_cache(respan, fov)
        if rows:
            return rows, _summary_from_disposition_rows(rows)
    rows, summary = build_disposition_rows(
        respan,
        fov,
        animal_id=animal_id,
        timepoint_names=timepoint_names,
        ignored_by_tp=ignored_by_tp,
    )
    header = [
        "animal_id",
        "fov",
        "global_spine_id",
        "timepoint",
        "local_spine_id",
        "disposition",
        "lineage_key",
        "status",
        "fate",
        "event",
        "conflict",
        "reviewed",
    ]
    out = disposition_path(respan, fov)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows, summary


def coverage_summary(
    respan: Path,
    fov: int,
    timepoint_names: List[str],
    ignored_by_tp: Optional[Dict[str, set]] = None,
    *,
    animal_id: str = "",
    disposition_rows: Optional[List[dict]] = None,
) -> dict:
    if disposition_rows is not None:
        return _summary_from_disposition_rows(disposition_rows)
    _, summary = coverage_and_unreviewed(
        respan,
        fov,
        timepoint_names,
        ignored_by_tp,
        animal_id=animal_id,
    )
    return summary


def apply_conflict_resolution(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    timepoint: str,
    spine_id: str,
    keep_lineage_key: str,
    timepoint_names: List[str],
    ignored_by_tp: Optional[Dict[str, set]] = None,
) -> dict:
    data = spine_lineage_store.load_decisions(respan, fov)
    lineages: List[dict] = list(data.get("lineages") or [])
    by_tp = _assignments_by_tp(lineages, timepoint_names)
    keys = list(by_tp.get(timepoint, {}).get(spine_id) or [])
    keep = str(keep_lineage_key).strip()
    if keep not in keys:
        raise ValueError(f"Lineage {keep} is not part of this conflict.")

    updated: List[str] = []
    removed: List[str] = []
    for lin in lineages:
        row_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if row_key == keep:
            continue
        if row_key not in keys:
            continue
        per_tp = dict(lin.get("per_tp") or {})
        td = dict(per_tp.get(timepoint) or {})
        conflict_sid = str(spine_id).strip()
        td_sid = str(td.get("spine_id") or "").strip()
        td_removed = str(td.get("removed_spine_id") or "").strip()
        if td_sid != conflict_sid and td_removed != conflict_sid:
            if not (
                td_sid == ""
                and td_removed == conflict_sid
                and str(td.get("fate") or "").strip()
            ):
                continue
        if str(td.get("fate") or "").strip() or str(td.get("artifact_mode") or "").strip():
            if td_sid == conflict_sid:
                td["spine_id"] = ""
                td["local_spine_id"] = ""
        else:
            td["spine_id"] = ""
            td["local_spine_id"] = ""
            td["source"] = spine_lineage_store.SOURCE_RETURNED_TO_POOL
        per_tp[timepoint] = td
        pre_id = str(lin.get("pre_spine_id") or "")
        anchor_tp = str(lin.get("pre_timepoint") or "")
        finalized, last_seen, _, censored, rc = spine_lineage_store.finalize_lineage_per_tp(
            per_tp,
            timepoint_names,
            pre_spine_id=pre_id,
            anchor_timepoint=anchor_tp,
            pre_tp=anchor_tp,
        )
        has_any = any(
            str((finalized.get(tp) or {}).get("spine_id") or "").strip()
            or str((finalized.get(tp) or {}).get("fate") or "").strip()
            for tp in timepoint_names
        )
        if not has_any:
            removed.append(row_key)
            continue
        lin["per_tp"] = {tp: dict(finalized.get(tp) or {}) for tp in timepoint_names}
        lin["last_seen_tp"] = last_seen
        lin["censored_from_tp"] = censored
        lin["right_censored"] = rc
        updated.append(row_key)

    if removed:
        lineages = [
            lin
            for lin in lineages
            if str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip() not in removed
        ]

    data["lineages"] = lineages
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta = spine_lineage_store.paths(respan, fov)
    meta["decisions"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_duplicate_resolution(
        respan,
        fov,
        animal_id=animal_id,
        timepoint=timepoint,
        spine_id=spine_id,
        canonical_lineage_key=keep,
    )
    spine_lineage_store.rebuild_registry_wide(respan, fov, animal_id=animal_id)
    disp_path, summary = rebuild_spine_disposition(
        respan,
        fov,
        animal_id=animal_id,
        timepoint_names=timepoint_names,
        ignored_by_tp=ignored_by_tp,
    )
    return {
        "keep_lineage_key": keep,
        "updated_lineages": updated,
        "removed_lineages": removed,
        "disposition_path": str(disp_path),
        "coverage": summary,
    }
