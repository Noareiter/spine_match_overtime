"""Longitudinal spine tracker: 5ֳ—2 grid (zoom + full FOV per timepoint).

Top row: high-magnification crops around each spine (or nearest region).
Bottom row: full FOV with XY pan (drag) and Z scroll; click recenters zoom.
Local registration runs when a T1 base spine is selected.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from . import animal_config, animal_layout, baseline_adapter, crop_service, dendrite_link_store
from . import (
    mtp_spine_matching,
    oof_segment_store,
    spine_lineage_store,
    manual_spine_store,
    ignored_spine_store,
    spine_catalog_store,
    spine_qc_store,
    spine_qc_tag_store,
    results_final_store,
)

router = APIRouter(prefix="/mtp/viewer", tags=["multi-timepoint-spine-viewer"])

LOCAL_REG_WINDOW_PX = 40.0
BLOCKING_FATES = frozenset({"lost"})
REVIEW_MODE_LINEAGE = "lineage"
REVIEW_MODE_TIMEPOINT = "timepoint"
# Timepoint mode only supports artifact/ignore — NEW/LOST are derived, never manually set
TIMEPOINT_FATES = frozenset({"artifact", "ignore"})


class _ViewerState:
    def __init__(self) -> None:
        self.animal_id: str = ""
        self.fov: int = 0
        self.respan_root: str = ""
        self.t1_timepoint: str = ""
        self.timepoint_names: List[str] = []
        self.files: Dict[str, Dict[str, str]] = {}
        self.spine_lookup: Dict[str, Dict[str, dict]] = {}
        self.spine_dfs: Dict[str, "pd.DataFrame"] = {}
        self.lineages: List[dict] = []
        self.dendrite_links: List[dict] = []
        self.registry_path: str = ""
        self.t1_spine_ids: List[str] = []
        self.spine_queue: List[dict] = []
        self.cross_dendrite_queue: List[dict] = []
        self.phase_queues: List[List[str]] = []
        self.queue_phase_index: int = 0
        self.queue_mode: str = "main"
        self.allow_cross_dendrite: bool = False
        self.active_t1_spine_id: str = ""
        self.positions: Dict[str, dict] = {}
        self.local_shifts: Dict[str, Tuple[float, float, float]] = {}
        self.active_link_id: str = ""
        self.oof_segments: List[dict] = []
        self.ignored_by_tp: Dict[str, set] = {}
        self.review_progress: dict = {}
        self.skip_orphan_phases: bool = False
        self.catalog: Optional[spine_catalog_store.SpineCatalog] = None
        self.catalog_path: str = ""
        self.review_mode: str = REVIEW_MODE_LINEAGE
        self._stacks: Dict[str, np.ndarray] = {}
        self.positions_b: Dict[str, dict] = {}
        self.conflicts_by_id: Dict[str, dict] = {}
        self.conflict_meta: Optional[dict] = None
        self.lineage_a_key: str = ""
        self.lineage_b_key: str = ""
        self.conflict_focus_row: str = "a"
        self.coverage_summary: dict = {}

    def reset(self) -> None:
        self.__init__()


_STATE = _ViewerState()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sort_spine_ids(ids: List[str]) -> List[str]:
    def key(s: str):
        s = str(s)
        if s.startswith(spine_catalog_store.GLOBAL_PREFIX):
            try:
                return (0, int(s[len(spine_catalog_store.GLOBAL_PREFIX) :]), "")
            except ValueError:
                pass
        try:
            return (1, float(s), "")
        except (TypeError, ValueError):
            return (2, 0.0, s)

    return sorted({str(i) for i in ids if str(i).strip()}, key=key)


def _float_or_none(val) -> Optional[float]:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _check_contiguity_would_violate(pos_map: Dict[str, dict], timepoint_names: List[str]) -> Optional[str]:
    """Check if current positions would violate contiguity if saved.

    Returns error message if gap would be created, None if OK.
    """
    if not timepoint_names:
        return None

    first_match_idx = None
    last_match_idx = None
    matched_indices = []

    for i, tp in enumerate(timepoint_names):
        td = pos_map.get(tp) or {}
        sid = str(td.get("spine_id") or "").strip()
        if sid:
            if first_match_idx is None:
                first_match_idx = i
            last_match_idx = i
            matched_indices.append(i)

    # No matches or single match is always OK
    if not matched_indices or len(matched_indices) == 1:
        return None

    # Check for gap
    for i in range(first_match_idx, last_match_idx + 1):
        if i not in matched_indices:
            gap_tp = timepoint_names[i]
            return f"Would create gap: spine matched at {timepoint_names[first_match_idx]} and {timepoint_names[last_match_idx]}, but not at {gap_tp}. Matched timepoints must be contiguous."

    return None


def _find_registry_csv(respan: Path, fov: int) -> Optional[Path]:
    meta = respan / "_annotator" / f"fov{fov}" / "spine_registry_wide.csv"
    if meta.is_file():
        return meta
    return None


def _parse_registry(path: Path, timepoint_names: List[str], *, fov: int) -> List[dict]:
    lineages: List[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("fov", "")).strip() not in ("", str(fov)):
                continue
            members: Dict[str, dict] = {}
            for tp in timepoint_names:
                sid = str(row.get(f"id_{tp}", "") or "").strip()
                if sid.startswith("bridge_"):
                    continue
                if sid and sid.lower() not in {"nan", "none"}:
                    members[tp] = {
                        "spine_id": sid,
                        "status": str(row.get(f"status_{tp}", "") or ""),
                        "x": _float_or_none(row.get(f"{tp}_x")),
                        "y": _float_or_none(row.get(f"{tp}_y")),
                        "z": _float_or_none(row.get(f"{tp}_z")),
                    }
            if not members:
                continue
            lineages.append(
                {
                    "lineage_id": str(row.get("lineage_id", "") or f"L{len(lineages)+1:05d}"),
                    "members": members,
                    "n_timepoints": len(members),
                }
            )
    return lineages


def _link_by_id(link_id: str) -> Optional[dict]:
    for link in _STATE.dendrite_links:
        if str(link.get("link_id", "")) == str(link_id):
            return link
    return None


def _globalize_queue_item(q: dict, pre_tp: str, mid_tp: str) -> dict:
    out = dict(q)
    cat = _STATE.catalog
    if not cat:
        return out
    out["pre_spine_id"] = cat.to_global(pre_tp, str(q.get("pre_spine_id") or ""))
    if mid_tp and q.get("mid_spine_id"):
        out["mid_spine_id"] = cat.to_global(mid_tp, str(q.get("mid_spine_id") or ""))
    out["local_pre_spine_id"] = cat.to_local(pre_tp, out["pre_spine_id"])
    return out


def _tiff_paths_map() -> Dict[str, str]:
    """Per-timepoint TIFF paths already resolved into _STATE.files at load time."""
    return {tp: info.get("tiff") or None for tp, info in _STATE.files.items()}


def _build_queues() -> None:
    if len(_STATE.timepoint_names) < 2:
        _STATE.spine_queue = []
        _STATE.cross_dendrite_queue = []
        return
    pre_tp = _STATE.timepoint_names[0]
    mid_tp = _STATE.timepoint_names[1]
    pre_df = _STATE.spine_dfs.get(pre_tp)
    mid_df = _STATE.spine_dfs.get(mid_tp)
    if pre_df is None or mid_df is None:
        return
    main, cross = mtp_spine_matching.build_pre_mid_queues(
        pre_df,
        mid_df,
        _STATE.dendrite_links,
        pre_tp=pre_tp,
        mid_tp=mid_tp,
        link_id=_STATE.active_link_id or None,
        tiff_paths=_tiff_paths_map(),
    )
    _STATE.spine_queue = [_globalize_queue_item(q, pre_tp, mid_tp) for q in main]
    _STATE.cross_dendrite_queue = [_globalize_queue_item(q, pre_tp, mid_tp) for q in cross]
    if _STATE.respan_root:
        _build_phase_queues(Path(_STATE.respan_root))
    _rebuild_spine_id_list()


def _all_pre_spine_ids() -> List[str]:
    pre_tp = _STATE.t1_timepoint or (_STATE.timepoint_names[0] if _STATE.timepoint_names else "")
    if _STATE.catalog:
        ids = list(_STATE.catalog.lookup_for_timepoint(pre_tp).keys())
    else:
        lookup = _STATE.spine_lookup.get(pre_tp) or {}
        ids = _sort_spine_ids(list(lookup.keys()))
    if _STATE.active_link_id:
        link = _link_by_id(_STATE.active_link_id)
        if link:
            allowed = {str(x) for x in link.get("members", {}).get(pre_tp, [])}
            if allowed:
                lookup = _STATE.spine_lookup.get(pre_tp) or {}
                ids = [
                    sid for sid in ids
                    if str(lookup.get(sid, {}).get("dendrite_id") or "") in allowed
                ]
    return _sort_spine_ids(ids)


def _orphan_spine_ids_at_tp(tp: str, respan: Path) -> List[str]:
    lookup = _STATE.spine_lookup.get(tp) or {}
    pending = _save_positions_snapshot() if _STATE.active_t1_spine_id and _STATE.positions else None
    claimed = spine_lineage_store.claimed_spine_ids_at_tp(
        respan,
        _STATE.fov,
        tp,
        exclude_lineage_key=str(_STATE.active_t1_spine_id or ""),
        pending_lineage=pending,
    )
    qc_tags = spine_qc_tag_store.load_tags(respan, _STATE.fov)
    tagged_spines = spine_qc_tag_store.tagged_spines_at_tp(qc_tags, tp)
    orphans = [
        sid for sid in _sort_spine_ids(list(lookup.keys()))
        if sid not in claimed
        and sid not in tagged_spines
        and not str(sid).startswith("bridge_")
        and not _is_spine_ignored(tp, sid)
    ]
    return orphans


def _resolve_lineage_key_for_active() -> str:
    """Map queue id (S_*, L_*, or tp|gid) to a lineage_key for claim exclusion."""
    active = str(_STATE.active_t1_spine_id or "").strip()
    if not active or "|" in active:
        return ""
    respan = _respan_path()
    for candidate in (active, f"L_{active}"):
        lin = spine_lineage_store.get_lineage_by_key(respan, _STATE.fov, candidate)
        if lin:
            return str(lin.get("lineage_key") or lin.get("pre_spine_id") or candidate)
    return active


def _claimed_for_matching(tp: str) -> set[str]:
    """Spine IDs at this TP claimed by other lineages (current edit uses live positions)."""
    if not _STATE.respan_root:
        return set()
    if _is_duplicate_phase():
        ours = _conflict_lineage_keys()
        claimed: set[str] = set()
        for lin in spine_lineage_store.load_decisions(_respan_path(), _STATE.fov).get(
            "lineages"
        ) or []:
            row_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
            if row_key in ours:
                continue
            td = dict((lin.get("per_tp") or {}).get(tp) or {})
            sid = str(td.get("spine_id") or "").strip()
            if sid:
                claimed.add(sid)
        return claimed
    if _is_unreviewed_phase():
        # Show full spine catalog; set-spine blocks picks already in other lineages.
        return set()
    pending = _save_positions_snapshot() if _STATE.active_t1_spine_id and _STATE.positions else None
    return spine_lineage_store.claimed_spine_ids_at_tp(
        _respan_path(),
        _STATE.fov,
        tp,
        exclude_lineage_key=_resolve_lineage_key_for_active(),
        pending_lineage=pending,
    )


def _all_assigned_spine_ids_at_tp(tp: str) -> set[str]:
    """Spine IDs at this TP already matched or fated in any lineage (for marker color)."""
    if not _STATE.respan_root:
        return set()
    respan = _respan_path()
    exclude = _resolve_lineage_key_for_active()
    pending = (
        _save_positions_snapshot()
        if _STATE.active_t1_spine_id and _STATE.positions
        else None
    )
    assigned: set[str] = set()
    for lin in spine_lineage_store.load_decisions(respan, _STATE.fov).get(
        "lineages"
    ) or []:
        row_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if exclude and row_key == exclude and pending is not None:
            td = dict(pending.get(tp) or {})
            for sid in (
                str(td.get("spine_id") or "").strip(),
                str(td.get("removed_spine_id") or "").strip(),
            ):
                if sid:
                    assigned.add(sid)
            continue
        assigned.update(spine_lineage_store._lineage_claim_ids_at_tp(lin, tp))
    return assigned


def _current_lineage_spine_ids() -> set[str]:
    if _is_duplicate_phase():
        return _lineage_spine_ids_from_map(_active_positions())
    return _lineage_spine_ids_from_map(_STATE.positions)


def _build_phase_queues(respan: Path) -> None:
    tps = _STATE.timepoint_names
    if not tps:
        _STATE.phase_queues = []
        return
    phase0: List[str] = []
    seen: set[str] = set()
    for q in _STATE.spine_queue:
        pid = str(q["pre_spine_id"])
        if pid not in seen and not _is_spine_ignored(tps[0], pid):
            seen.add(pid)
            phase0.append(pid)
    for sid in _all_pre_spine_ids():
        if sid not in seen and not _is_spine_ignored(tps[0], sid):
            seen.add(sid)
            phase0.append(sid)
    phases: List[List[str]] = [phase0]
    if not _STATE.skip_orphan_phases:
        for k in range(1, len(tps)):
            phases.append(_orphan_spine_ids_at_tp(tps[k], respan))
    conflicts = spine_qc_store.find_duplicate_conflicts(respan, _STATE.fov, tps)
    _STATE.conflicts_by_id = {c["conflict_id"]: c for c in conflicts}
    phases.append([c["conflict_id"] for c in conflicts])
    disp_rows, disp_summary = spine_qc_store.coverage_and_unreviewed(
        respan,
        _STATE.fov,
        tps,
        ignored_by_tp=_STATE.ignored_by_tp,
        animal_id=_STATE.animal_id,
    )
    _STATE.coverage_summary = disp_summary
    unreviewed = spine_qc_store.unreviewed_queue(
        respan,
        _STATE.fov,
        tps,
        ignored_by_tp=_STATE.ignored_by_tp,
        disposition_rows=disp_rows,
    )
    phases.append([u["queue_id"] for u in unreviewed])
    _STATE.phase_queues = phases


def _lineage_phase_count() -> int:
    if _STATE.skip_orphan_phases:
        return 1
    return max(1, len(_STATE.timepoint_names))


def _phase_kind_at(index: int) -> str:
    lc = _lineage_phase_count()
    if index < lc:
        return "lineage"
    if index == lc:
        return "duplicate"
    return "unreviewed"


def _current_phase_kind() -> str:
    return _phase_kind_at(_STATE.queue_phase_index)


def _is_duplicate_phase() -> bool:
    return _current_phase_kind() == "duplicate"


def _is_unreviewed_phase() -> bool:
    return _current_phase_kind() == "unreviewed"


def _review_phase_count() -> int:
    if _STATE.phase_queues:
        return len(_STATE.phase_queues)
    return 1 if _STATE.skip_orphan_phases else max(1, len(_STATE.timepoint_names))


def _current_anchor_timepoint() -> str:
    tps = _STATE.timepoint_names
    if not tps:
        return ""
    kind = _current_phase_kind()
    if kind == "duplicate":
        cid = str(_STATE.active_t1_spine_id or "")
        if not cid and _STATE.t1_spine_ids:
            cid = str(_STATE.t1_spine_ids[0])
        conf = _STATE.conflicts_by_id.get(cid) or {}
        return str(conf.get("timepoint") or tps[0])
    if kind == "unreviewed":
        qid = str(_STATE.active_t1_spine_id or "")
        if not qid and _STATE.t1_spine_ids:
            qid = str(_STATE.t1_spine_ids[0])
        tp, _ = spine_qc_store.parse_queue_id(qid)
        return tp or tps[0]
    if _STATE.skip_orphan_phases:
        return tps[0]
    idx = max(0, min(_STATE.queue_phase_index, len(tps) - 1))
    return tps[idx]


def _rebuild_spine_id_list() -> None:
    if _STATE.queue_mode == "cross":
        seen: set[str] = set()
        ids: List[str] = []
        for q in _STATE.cross_dendrite_queue:
            pid = str(q["pre_spine_id"])
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)
        _STATE.t1_spine_ids = ids
        return
    if not _STATE.phase_queues:
        _STATE.t1_spine_ids = [str(q["pre_spine_id"]) for q in _STATE.spine_queue]
        return
    idx = max(0, min(_STATE.queue_phase_index, len(_STATE.phase_queues) - 1))
    _STATE.t1_spine_ids = list(_STATE.phase_queues[idx])


def _qc_phase_indices() -> Tuple[int, int]:
    lc = _lineage_phase_count()
    return lc, lc + 1


def _pending_qc_counts() -> Tuple[int, int]:
    if not _STATE.phase_queues:
        return 0, 0
    dup_idx, unrev_idx = _qc_phase_indices()
    dup_n = len(_STATE.phase_queues[dup_idx]) if len(_STATE.phase_queues) > dup_idx else 0
    unrev_n = len(_STATE.phase_queues[unrev_idx]) if len(_STATE.phase_queues) > unrev_idx else 0
    return dup_n, unrev_n


def _snap_to_pending_qc_work() -> Optional[str]:
    """If QC phases were passed but work remains, rewind to the earliest pending one."""
    dup_idx, unrev_idx = _qc_phase_indices()
    if _STATE.queue_phase_index < dup_idx:
        return None
    dup_n, unrev_n = _pending_qc_counts()
    if dup_n > 0 and _STATE.queue_phase_index != dup_idx:
        _STATE.queue_phase_index = dup_idx
        _rebuild_spine_id_list()
        return f"Returned to duplicate conflicts ({dup_n} pending)."
    if dup_n == 0 and unrev_n > 0 and _STATE.queue_phase_index != unrev_idx:
        _STATE.queue_phase_index = unrev_idx
        _rebuild_spine_id_list()
        return f"Returned to unreviewed sweep ({unrev_n} pending)."
    return None


def _ensure_nonempty_phase(respan: Path) -> None:
    """Skip phases with no spines left to review."""
    max_phase = max(0, _review_phase_count() - 1)
    while True:
        _rebuild_spine_id_list()
        if _STATE.t1_spine_ids:
            return
        if _STATE.queue_phase_index >= max_phase:
            return
        _STATE.queue_phase_index += 1
        _build_phase_queues(respan)


def _advance_phase_if_complete(respan: Path, *, force_advance: bool = False) -> Optional[dict]:
    """Move to next timepoint orphan queue after finishing current phase."""
    idx = _STATE.queue_phase_index
    if idx >= _review_phase_count() - 1:
        return None
    if not force_advance:
        if not _STATE.t1_spine_ids:
            return None
        active = _STATE.active_t1_spine_id
        if active not in _STATE.t1_spine_ids:
            return None
        if _STATE.t1_spine_ids.index(active) < len(_STATE.t1_spine_ids) - 1:
            return None
    next_idx = idx + 1
    tps = _STATE.timepoint_names
    while next_idx < _review_phase_count():
        _STATE.queue_phase_index = next_idx
        _build_phase_queues(respan)
        _ensure_nonempty_phase(respan)
        anchor = _current_anchor_timepoint()
        if _STATE.t1_spine_ids:
            spine_lineage_store.save_progress(
                respan,
                _STATE.fov,
                index=0,
                reviewed_ids=list(_STATE.review_progress.get("reviewed_ids") or []),
                phase_index=next_idx,
                anchor_timepoint=anchor,
            )
            _STATE.review_progress = spine_lineage_store.load_progress(respan, _STATE.fov)
            kind = _phase_kind_at(next_idx)
            if kind == "duplicate":
                label = f"duplicate conflicts ({len(_STATE.t1_spine_ids)})"
            elif kind == "unreviewed":
                label = f"unreviewed spines ({len(_STATE.t1_spine_ids)})"
            else:
                label = f"remaining spines at {anchor} ({len(_STATE.t1_spine_ids)})"
            return {
                "phase_index": next_idx,
                "anchor_timepoint": anchor,
                "phase_kind": kind,
                "queue_count": len(_STATE.t1_spine_ids),
                "spine_ids": list(_STATE.t1_spine_ids),
                "coverage": dict(_STATE.coverage_summary or {}),
                "message": f"Phase {next_idx + 1}/{_review_phase_count()}: {label}",
            }
        next_idx += 1
    return None


def _queue_item(pre_spine_id: str) -> Optional[dict]:
    for q in _STATE.spine_queue:
        if str(q["pre_spine_id"]) == str(pre_spine_id):
            return q
    for q in _STATE.cross_dendrite_queue:
        if str(q["pre_spine_id"]) == str(pre_spine_id):
            return q
    return None


def _lineage_spine_ids_from_map(pos_map: Dict[str, dict]) -> set[str]:
    ids: set[str] = set()
    for pos in (pos_map or {}).values():
        sid = str(pos.get("spine_id") or "").strip()
        if sid:
            ids.add(sid)
    return ids


def _apply_conflict_edit_row(lineage_row: str = "") -> None:
    if not _is_duplicate_phase():
        return
    r = str(lineage_row or "").strip().lower()
    if r.startswith("b"):
        _STATE.conflict_focus_row = "b"
    elif r.startswith("a"):
        _STATE.conflict_focus_row = "a"


def _markers_for_panel(
    tp: str,
    bounds: dict,
    width: int,
    height: int,
    focus_pos: Optional[dict],
    *,
    anchor_pos: Optional[dict] = None,
    pos_map: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    if pos_map is None:
        pos_map = _STATE.positions
    return _markers_for_panel_positions(
        pos_map, tp, bounds, width, height, focus_pos, anchor_pos=anchor_pos
    )


def _markers_for_panel_positions(
    pos_map: Dict[str, dict],
    tp: str,
    bounds: dict,
    width: int,
    height: int,
    focus_pos: Optional[dict],
    *,
    anchor_pos: Optional[dict] = None,
) -> List[dict]:
    x0 = float(bounds.get("x0", 0))
    y0 = float(bounds.get("y0", 0))
    x1 = float(bounds.get("x1", 1))
    y1 = float(bounds.get("y1", 1))
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    focus_id = str((focus_pos or {}).get("spine_id") or "")
    markers: List[dict] = []
    lookup = _STATE.spine_lookup.get(tp) or {}
    assigned_ids = _all_assigned_spine_ids_at_tp(tp)
    for sid, rec in lookup.items():
        sid_str = str(sid)
        if manual_spine_store.is_manual_id(sid_str) and sid_str != focus_id:
            continue
        if _is_spine_ignored(tp, sid):
            continue
        if not _filter_oof_spine(rec, tp):
            continue
        x, y = float(rec["x"]), float(rec["y"])
        if not (x0 <= x < x1 and y0 <= y < y1):
            continue
        is_focus = sid_str == focus_id
        markers.append(
            {
                "spine_id": str(sid),
                "x": float((x - x0) / w * width),
                "y": float((y - y0) / h * height),
                "z": float(rec.get("z", 0)),
                "label": str(rec.get("label") or rec.get("local_spine_id") or sid),
                "role": "focus"
                if is_focus
                else ("manual" if str(sid).startswith("manual_") else "csv"),
                "assigned": sid_str in assigned_ids and not is_focus,
            }
        )
    if focus_pos and not focus_id:
        fx, fy = float(focus_pos["x"]), float(focus_pos["y"])
        if x0 <= fx < x1 and y0 <= fy < y1:
            markers.append(
                {
                    "spine_id": "",
                    "x": float((fx - x0) / w * width),
                    "y": float((fy - y0) / h * height),
                    "label": "?",
                    "role": "region",
                }
            )
    if anchor_pos:
        ax, ay = float(anchor_pos.get("x", 0)), float(anchor_pos.get("y", 0))
        fx = float((focus_pos or {}).get("x", ax))
        fy = float((focus_pos or {}).get("y", ay))
        if abs(ax - fx) > 0.5 or abs(ay - fy) > 0.5:
            if x0 <= ax < x1 and y0 <= ay < y1:
                markers.append(
                    {
                        "spine_id": "",
                        "x": float((ax - x0) / w * width),
                        "y": float((ay - y0) / h * height),
                        "label": "+",
                        "role": "anchor",
                    }
                )
    return markers


def _conflict_timepoint() -> str:
    """Timepoint used for side-by-side comparison in the UI."""
    conf = _STATE.conflict_meta or {}
    comp = str(conf.get("comparison_timepoint") or "").strip()
    if comp:
        return comp
    return str(conf.get("timepoint") or "").strip()


def _duplicate_detection_timepoint() -> str:
    """Timepoint where the duplicate spine_id was detected (for resolution)."""
    conf = _STATE.conflict_meta or {}
    return str(conf.get("timepoint") or "").strip()


def _marker_label_for_row(pos: dict, row: str, lineage_key: str) -> str:
    sid = str(pos.get("spine_id") or "").strip()
    local = str(pos.get("local_spine_id") or "").strip()
    lk = str(lineage_key or "").strip()
    parts = [row.upper()]
    if local:
        parts.append(f"loc {local}")
    if sid:
        parts.append(sid)
    elif pos.get("fate"):
        parts.append(str(pos.get("fate")).upper())
    if lk:
        parts.append(lk)
    return " · ".join(parts)


def _markers_for_conflict_tp(
    tp: str,
    bounds: dict,
    width: int,
    height: int,
    *,
    primary_row: str = "a",
) -> List[dict]:
    """At the conflict timepoint, show both lineage A and B assignments."""
    if not _is_duplicate_phase() or tp != _conflict_timepoint():
        return []
    x0 = float(bounds.get("x0", 0))
    y0 = float(bounds.get("y0", 0))
    x1 = float(bounds.get("x1", 1))
    y1 = float(bounds.get("y1", 1))
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    conf_sid = str((_STATE.conflict_meta or {}).get("spine_id") or "").strip()
    dup_tp = _duplicate_detection_timepoint()
    primary = str(primary_row or "a").strip().lower()
    markers: List[dict] = []
    rows = (
        ("a", _STATE.positions.get(tp), _STATE.lineage_a_key, "conflict_a"),
        ("b", _STATE.positions_b.get(tp), _STATE.lineage_b_key, "conflict_b"),
    )
    for row_id, pos, lk, peer_role in rows:
        if not pos:
            continue
        fx = pos.get("x")
        fy = pos.get("y")
        if fx is None or fy is None:
            continue
        fx, fy = float(fx), float(fy)
        if not (x0 <= fx < x1 and y0 <= fy < y1):
            continue
        sid = str(pos.get("spine_id") or "").strip()
        label = _marker_label_for_row(pos, row_id, lk)
        if tp == dup_tp and conf_sid and sid and sid != conf_sid:
            label = f"! {label}"
        role = "focus" if row_id == primary else peer_role
        markers.append(
            {
                "spine_id": sid,
                "x": float((fx - x0) / w * width),
                "y": float((fy - y0) / h * height),
                "label": label,
                "role": role,
            }
        )
    return markers


def _conflict_compare_center(
    tp: str,
    x: float,
    y: float,
    zoom_size: int,
) -> Tuple[float, float, int]:
    """Center zoom between A and B at conflict TP; widen if they are apart."""
    if not _is_duplicate_phase() or tp != _conflict_timepoint():
        return x, y, zoom_size
    pos_a = _STATE.positions.get(tp) or {}
    pos_b = _STATE.positions_b.get(tp) or {}
    ax, ay = pos_a.get("x"), pos_a.get("y")
    bx, by = pos_b.get("x"), pos_b.get("y")
    if ax is None or ay is None or bx is None or by is None:
        return x, y, zoom_size
    ax, ay, bx, by = float(ax), float(ay), float(bx), float(by)
    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    z = int(zoom_size)
    if dist > 15.0:
        z = max(z, int(dist * 2.5))
    return cx, cy, z


def _peer_conflict_markers(
    tp: str,
    bounds: dict,
    width: int,
    height: int,
    *,
    viewing_row: str,
) -> List[dict]:
    """Overlay the other lineage's marker at the comparison timepoint."""
    if not _is_duplicate_phase() or tp != _conflict_timepoint():
        return []
    use_b = str(viewing_row or "a").strip().lower().startswith("b")
    peer_map = _STATE.positions if use_b else _STATE.positions_b
    peer_key = _STATE.lineage_a_key if use_b else _STATE.lineage_b_key
    peer_row = "a" if use_b else "b"
    peer_role = "conflict_a" if use_b else "conflict_b"
    pos = peer_map.get(tp) or {}
    x0 = float(bounds.get("x0", 0))
    y0 = float(bounds.get("y0", 0))
    x1 = float(bounds.get("x1", 1))
    y1 = float(bounds.get("y1", 1))
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    fx = pos.get("x")
    fy = pos.get("y")
    if fx is None or fy is None:
        return []
    fx, fy = float(fx), float(fy)
    if not (x0 <= fx < x1 and y0 <= fy < y1):
        return []
    sid = str(pos.get("spine_id") or "").strip()
    label = _marker_label_for_row(pos, peer_row, peer_key)
    return [
        {
            "spine_id": sid,
            "x": float((fx - x0) / w * width),
            "y": float((fy - y0) / h * height),
            "label": label,
            "role": peer_role,
        }
    ]


def _markers_for_saved_lineage(
    tp: str,
    bounds: dict,
    width: int,
    height: int,
    focus_pos: Optional[dict],
) -> List[dict]:
    """Markers for conflict-phase read-only lineage row (no claimed-spine filtering)."""
    x0 = float(bounds.get("x0", 0))
    y0 = float(bounds.get("y0", 0))
    x1 = float(bounds.get("x1", 1))
    y1 = float(bounds.get("y1", 1))
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    markers: List[dict] = []
    if not focus_pos:
        return markers
    fx = focus_pos.get("x")
    fy = focus_pos.get("y")
    if fx is None or fy is None:
        return markers
    fx, fy = float(fx), float(fy)
    if not (x0 <= fx < x1 and y0 <= fy < y1):
        return markers
    sid = str(focus_pos.get("spine_id") or "")
    label = sid or str(focus_pos.get("fate") or "?").upper()
    markers.append(
        {
            "spine_id": sid,
            "x": float((fx - x0) / w * width),
            "y": float((fy - y0) / h * height),
            "label": label,
            "role": "focus",
        }
    )
    return markers


def _conflict_response_extra() -> dict:
    conf = _STATE.conflict_meta or {}
    return {
        "phase_kind": _current_phase_kind(),
        "conflict": conf,
        "lineage_a_key": _STATE.lineage_a_key,
        "lineage_b_key": _STATE.lineage_b_key,
        "conflict_focus_row": _STATE.conflict_focus_row,
        "positions_b": _positions_response_b(),
        "coverage": dict(_STATE.coverage_summary or {}),
    }


def _conflict_lineage_keys() -> set[str]:
    return {
        k
        for k in (
            str(_STATE.lineage_a_key or "").strip(),
            str(_STATE.lineage_b_key or "").strip(),
        )
        if k
    }


def _conflict_focus_is_b() -> bool:
    return _is_duplicate_phase() and str(
        _STATE.conflict_focus_row or "a"
    ).strip().lower().startswith("b")


def _active_positions() -> Dict[str, dict]:
    return _STATE.positions_b if _conflict_focus_is_b() else _STATE.positions


def _set_active_positions(positions: Dict[str, dict]) -> None:
    if _conflict_focus_is_b():
        _STATE.positions_b = positions
    else:
        _STATE.positions = positions


def _positions_response_b() -> Dict[str, dict]:
    if not _is_duplicate_phase():
        return {}
    positions = _apply_fate_rules(dict(_STATE.positions_b))
    enriched: Dict[str, dict] = {}
    active = str((_STATE.conflict_meta or {}).get("timepoint") or _current_anchor_timepoint())
    for tp in _STATE.timepoint_names:
        pos = dict(positions.get(tp) or {})
        pos["fate_options"] = _fate_options_for_tp(
            tp, positions, review_mode=_STATE.review_mode
        )
        pos["show_fate_dropdown"] = _show_fate_dropdown(tp, positions)
        pos["false_positive_eligible"] = _false_positive_eligible(tp, positions)
        pos["timepoint_fate_active"] = (
            _STATE.review_mode == REVIEW_MODE_TIMEPOINT and tp == active
        )
        enriched[tp] = pos
    return enriched


def _per_tp_snapshot(positions: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    cat = _STATE.catalog
    for tp in _STATE.timepoint_names:
        pos = dict(positions.get(tp) or {})
        sid = pos.get("spine_id")
        local_sid = ""
        if sid and cat:
            local_sid = cat.to_local(tp, str(sid))
        out[tp] = {
            "spine_id": sid,
            "local_spine_id": local_sid or None,
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
            "dendrite_id": pos.get("dendrite_id"),
            "source": pos.get("source"),
            "fate": pos.get("fate"),
            "decision_scope": pos.get("decision_scope"),
            "artifact_mode": pos.get("artifact_mode"),
            "removed_spine_id": pos.get("removed_spine_id"),
            "continuity": pos.get("continuity"),
        }
    return out


def _save_lineage_from_ui(lineage_key: str, positions: Dict[str, dict]) -> None:
    key = str(lineage_key or "").strip()
    if not key:
        return
    respan = _respan_path()
    lin = spine_lineage_store.get_lineage_by_key(respan, _STATE.fov, key)
    if not lin:
        return
    per_tp = _apply_fate_rules(_per_tp_snapshot(positions))
    anchor_tp = str(lin.get("pre_timepoint") or "")
    anchor_local = str(lin.get("pre_spine_id") or "")
    spine_lineage_store.save_decision(
        respan,
        _STATE.fov,
        animal_id=_STATE.animal_id,
        pre_spine_id=anchor_local,
        pre_timepoint=anchor_tp,
        timepoint_names=_STATE.timepoint_names,
        per_tp=per_tp,
        lineage_key=key,
        oof_segments=_STATE.oof_segments,
        ignored_by_tp=_STATE.ignored_by_tp,
        shifts=_STATE.local_shifts,
    )


def _save_conflict_edits() -> List[str]:
    saved: List[str] = []
    if _STATE.lineage_a_key:
        _save_lineage_from_ui(_STATE.lineage_a_key, _STATE.positions)
        saved.append(_STATE.lineage_a_key)
    if _STATE.lineage_b_key:
        _save_lineage_from_ui(_STATE.lineage_b_key, _STATE.positions_b)
        saved.append(_STATE.lineage_b_key)
    return saved


def _save_conflict_edits_row(row: str = "") -> List[str]:
    """Save one or both conflict lineages without resolving the duplicate."""
    r = str(row or "").strip().lower()
    if r.startswith("b"):
        if not _STATE.lineage_b_key:
            return []
        _save_lineage_from_ui(_STATE.lineage_b_key, _STATE.positions_b)
        return [_STATE.lineage_b_key]
    if r.startswith("a"):
        if not _STATE.lineage_a_key:
            return []
        _save_lineage_from_ui(_STATE.lineage_a_key, _STATE.positions)
        return [_STATE.lineage_a_key]
    return _save_conflict_edits()


def _reload_conflict_lineages() -> None:
    respan = _respan_path()
    lin_a = spine_lineage_store.get_lineage_by_key(
        respan, _STATE.fov, _STATE.lineage_a_key
    )
    lin_b = spine_lineage_store.get_lineage_by_key(
        respan, _STATE.fov, _STATE.lineage_b_key
    )
    _STATE.positions = _apply_fate_rules(
        spine_qc_store.positions_from_lineage(
            lin_a, _STATE.timepoint_names, _STATE.spine_lookup
        )
    )
    _STATE.positions_b = _apply_fate_rules(
        spine_qc_store.positions_from_lineage(
            lin_b, _STATE.timepoint_names, _STATE.spine_lookup
        )
    )


def _select_spine_response(**kwargs) -> SelectSpineResponse:
    base = {
        "t1_spine_id": _STATE.active_t1_spine_id,
        "positions": _positions_response(),
        "review_mode": _STATE.review_mode,
        "anchor_timepoint": _current_anchor_timepoint(),
    }
    if _is_duplicate_phase():
        base.update(_conflict_response_extra())
    base.update(kwargs)
    return SelectSpineResponse(**base)


def _respan_path() -> Path:
    return Path(_STATE.respan_root)


def _registry_mtime_iso(path: Optional[Path]) -> str:
    if not path or not path.is_file():
        return ""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _attach_results_final(respan: Path, out: dict, *, rebuild: bool = True) -> None:
    """Add all-FOV completion status; build Results final when every FOV is done."""
    cfg = animal_config.load_config()
    statuses, all_complete = results_final_store.all_fovs_status(respan, cfg)
    out["all_fovs_complete"] = all_complete
    out["fov_completion_status"] = statuses
    if not all_complete:
        out["pending_fovs"] = [s["fov"] for s in statuses if not s["complete"]]
        return
    if not rebuild:
        out["results_final_dir"] = str(results_final_store.results_final_dir(respan))
        return
    try:
        built = results_final_store.build_results_final(respan, cfg)
        build_msg = built.pop("message", "")
        out.update(built)
        if build_msg:
            out["results_final_message"] = build_msg
    except Exception as exc:
        out["results_final_error"] = str(exc)


def _apply_fate_rules(positions: Dict[str, dict]) -> Dict[str, dict]:
    tps = _STATE.timepoint_names
    out = spine_lineage_store.strip_bridge_fates(positions, tps)
    out = {k: dict(v) for k, v in out.items()}
    for i, tp in enumerate(tps):
        pos = dict(out.get(tp) or {})
        pos.setdefault("fate", None)
        pos["fate_locked"] = False
        if i > 0:
            prev = out.get(tps[i - 1]) or {}
            prev_fate = str(prev.get("fate") or "").lower()
            prev_scope = spine_lineage_store.decision_scope(prev)
            if prev_fate == "lost" and prev_scope != spine_lineage_store.DECISION_SCOPE_LOCAL:
                pos["fate_locked"] = True
        out[tp] = pos
    return out


def _all_prior_empty(positions: Dict[str, dict], idx: int) -> bool:
    """All earlier TPs: no match, no fate (NEW dropdown eligibility)."""
    tps = _STATE.timepoint_names
    for j in range(idx):
        prev = positions.get(tps[j]) or {}
        if str(prev.get("spine_id") or "").strip() or str(prev.get("fate") or "").strip():
            return False
    return True


def _false_positive_eligible(tp: str, positions: Dict[str, dict]) -> bool:
    """False+ only when this TP has a match and every other TP is clear (no match, no fate)."""
    pos = positions.get(tp) or {}
    if not str(pos.get("spine_id") or "").strip():
        return False
    if spine_lineage_store._is_false_positive_mark(pos):
        return False
    return spine_lineage_store.all_other_timepoints_cleared(
        positions, list(_STATE.timepoint_names), tp
    )


def _fate_options_for_tp(
    tp: str,
    positions: Dict[str, dict],
    *,
    review_mode: str = "",
) -> List[str]:
    mode = str(review_mode or _STATE.review_mode or REVIEW_MODE_LINEAGE).strip().lower()
    if mode != REVIEW_MODE_TIMEPOINT:
        return []
    active = _current_anchor_timepoint()
    if tp != active:
        return []
    pos = positions.get(tp) or {}
    if pos.get("fate_locked"):
        return []
    return ["artifact", "ignore"]


def _show_fate_dropdown(tp: str, positions: Dict[str, dict]) -> bool:
    if _STATE.review_mode != REVIEW_MODE_TIMEPOINT:
        return False
    active = _current_anchor_timepoint()
    if tp != active:
        return False
    pos = positions.get(tp) or {}
    if pos.get("spine_id"):
        return False
    return not pos.get("fate_locked")


def _is_spine_ignored(tp: str, spine_id: str) -> bool:
    sid = str(spine_id or "").strip()
    if _STATE.catalog and spine_catalog_store.is_global_id(sid):
        local = _STATE.catalog.to_local(tp, sid)
        if local and ignored_spine_store.is_ignored(_STATE.ignored_by_tp, tp, local):
            return True
    return ignored_spine_store.is_ignored(_STATE.ignored_by_tp, tp, sid)


def _sync_ignored_from_oof(respan: Path) -> None:
    """Rebuild ignored-spine list from current OOF segments."""
    by_tp = oof_segment_store.ignored_spines_by_timepoint(
        _STATE.oof_segments,
        list(_STATE.timepoint_names),
        _STATE.spine_lookup,
    )
    ignored_spine_store.save_ignored(
        respan, _STATE.fov, by_tp, animal_id=_STATE.animal_id
    )
    _STATE.ignored_by_tp = {tp: set(ids) for tp, ids in by_tp.items()}


def _oof_overlays_for_panel(tp: str, bounds: dict, width: int, height: int) -> List[dict]:
    x0 = float(bounds.get("x0", 0))
    y0 = float(bounds.get("y0", 0))
    x1 = float(bounds.get("x1", 1))
    y1 = float(bounds.get("y1", 1))
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    overlays: List[dict] = []

    def to_canvas(sx: float, sy: float) -> Tuple[float, float]:
        return float((sx - x0) / w * width), float((sy - y0) / h * height)

    for seg in _STATE.oof_segments:
        pre_tp = str(seg.get("pre_tp", ""))
        per_tp = seg.get("per_tp") or {}
        raw = per_tp.get(tp)
        if raw is None and tp == pre_tp:
            raw = seg.get("pre_shape") or seg.get("pre_bounds")
        sh = oof_segment_store.normalize_shape(raw)
        if not sh:
            continue
        bb = oof_segment_store.shape_bounds(sh)
        if not bb or bb["x1"] < x0 or bb["x0"] > x1 or bb["y1"] < y0 or bb["y0"] > y1:
            continue
        if sh["shape"] == "polygon":
            pts = [list(to_canvas(p[0], p[1])) for p in sh["points"]]
            overlays.append(
                {
                    "segment_id": str(seg.get("segment_id", "")),
                    "shape": "polygon",
                    "points": pts,
                }
            )
        else:
            bx0, by0 = float(sh["x0"]), float(sh["y0"])
            bx1, by1 = float(sh["x1"]), float(sh["y1"])
            overlays.append(
                {
                    "segment_id": str(seg.get("segment_id", "")),
                    "shape": "rect",
                    "x0": float(max(bx0, x0) - x0) / w * width,
                    "y0": float(max(by0, y0) - y0) / h * height,
                    "x1": float(min(bx1, x1) - x0) / w * width,
                    "y1": float(min(by1, y1) - y0) / h * height,
                }
            )
    return overlays


def _positions_response() -> Dict[str, dict]:
    positions = _apply_fate_rules(_STATE.positions)
    enriched: Dict[str, dict] = {}
    active = _current_anchor_timepoint()
    for tp in _STATE.timepoint_names:
        pos = dict(positions.get(tp) or {})
        pos["fate_options"] = _fate_options_for_tp(tp, positions)
        pos["show_fate_dropdown"] = _show_fate_dropdown(tp, positions)
        pos["false_positive_eligible"] = _false_positive_eligible(tp, positions)
        pos["timepoint_fate_active"] = (
            _STATE.review_mode == REVIEW_MODE_TIMEPOINT and tp == active
        )
        enriched[tp] = pos
    return enriched


def _save_positions_snapshot() -> Dict[str, dict]:
    """Strip UI-only fields; keep coords and ids for persistence."""
    out: Dict[str, dict] = {}
    cat = _STATE.catalog
    for tp, pos in _positions_response().items():
        sid = pos.get("spine_id")
        local_sid = ""
        if sid and cat:
            local_sid = cat.to_local(tp, str(sid))
        out[tp] = {
            "spine_id": sid,
            "local_spine_id": local_sid or None,
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
            "dendrite_id": pos.get("dendrite_id"),
            "source": pos.get("source"),
            "fate": pos.get("fate"),
            "decision_scope": pos.get("decision_scope"),
            "artifact_mode": pos.get("artifact_mode"),
            "removed_spine_id": pos.get("removed_spine_id"),
            "continuity": pos.get("continuity"),
        }
    return out


def _apply_catalog(respan: Path) -> str:
    """Build/load spine catalog and replace per-TP lookups with global S_* keys."""
    csv_by_tp = {
        tp: Path(_STATE.files[tp]["csv"])
        for tp in _STATE.timepoint_names
        if _STATE.files.get(tp, {}).get("csv")
    }
    info = spine_catalog_store.sync_catalog_for_fov(
        respan,
        _STATE.fov,
        animal_id=_STATE.animal_id,
        timepoint_names=list(_STATE.timepoint_names),
        csv_by_tp=csv_by_tp,
    )
    catalog = spine_catalog_store.load_catalog(respan, _STATE.fov)
    if catalog:
        _STATE.catalog = catalog
        for tp in _STATE.timepoint_names:
            _STATE.spine_lookup[tp] = catalog.lookup_for_timepoint(tp)
    _STATE.catalog_path = info.get("catalog_path", "")
    return str(info.get("message") or info.get("catalog_path") or "")


def _active_pre_dendrite_id() -> str:
    pre_tp = _STATE.t1_timepoint
    pre_id = _STATE.active_t1_spine_id
    lookup = _STATE.spine_lookup.get(pre_tp) or {}
    if pre_id in lookup:
        return str(lookup[pre_id].get("dendrite_id") or "")
    return ""


def _filter_oof_spine(rec: dict, tp: str, dendrite_id: str = "") -> bool:
    return not oof_segment_store.coords_in_oof(
        float(rec["x"]),
        float(rec["y"]),
        tp,
        _STATE.oof_segments,
        dendrite_id=dendrite_id or str(rec.get("dendrite_id") or ""),
    )


def _all_manual_ids_for_tp(tp: str, respan: Path, fov: int) -> List[str]:
    ids: List[str] = [sid for sid in (_STATE.spine_lookup.get(tp) or {}) if manual_spine_store.is_manual_id(sid)]
    data = spine_lineage_store.load_decisions(respan, fov)
    for lin in data.get("lineages") or []:
        td = (lin.get("per_tp") or {}).get(tp) or {}
        sid = str(td.get("spine_id") or "").strip()
        if sid and manual_spine_store.is_manual_entry(td):
            ids.append(sid)
    return ids


def _inject_manual_from_decisions(respan: Path, fov: int) -> int:
    data = spine_lineage_store.load_decisions(respan, fov)
    by_tp = manual_spine_store.collect_from_lineages(data.get("lineages") or [])
    count = 0
    for tp, rows in by_tp.items():
        if tp not in _STATE.spine_lookup:
            continue
        lookup = _STATE.spine_lookup[tp]
        for rec in rows:
            sid = str(rec["spine_id"])
            lookup[sid] = rec
            count += 1
    return count


def _filter_t1_spines_for_link(link_id: Optional[str]) -> None:
    t1_lookup = _STATE.spine_lookup.get(_STATE.t1_timepoint) or {}
    all_ids = _sort_spine_ids(list(t1_lookup.keys()))
    if not link_id:
        _STATE.t1_spine_ids = all_ids
        return
    link = _link_by_id(link_id)
    if not link:
        _STATE.t1_spine_ids = all_ids
        return
    allowed = {str(x) for x in link.get("members", {}).get(_STATE.t1_timepoint, [])}
    if not allowed:
        _STATE.t1_spine_ids = all_ids
        return
    _STATE.t1_spine_ids = [
        sid for sid in all_ids if str(t1_lookup[sid].get("dendrite_id") or "") in allowed
    ]


def _lineage_for_t1_spine(t1_spine_id: str) -> Optional[dict]:
    for lin in _STATE.lineages:
        t1 = _STATE.t1_timepoint
        mem = (lin.get("members") or {}).get(t1)
        if mem and str(mem.get("spine_id")) == str(t1_spine_id):
            return lin
    return None


def _linked_dendrites(src_tp: str, src_did: str, dst_tp: str) -> set[str]:
    out: set[str] = set()
    for link in _STATE.dendrite_links:
        members = link.get("members") or {}
        src_ids = {str(x) for x in members.get(src_tp, [])}
        if src_did in src_ids:
            for did in members.get(dst_tp, []):
                out.add(str(did))
    return out


def _nearest_spine(
    lookup: Dict[str, dict],
    x: float,
    y: float,
    z: float,
    *,
    tp: str,
    dendrite_ids: Optional[set[str]] = None,
) -> Optional[dict]:
    best: Optional[dict] = None
    best_d = float("inf")
    for sid, rec in lookup.items():
        did = str(rec.get("dendrite_id") or "")
        if dendrite_ids and did not in dendrite_ids:
            continue
        if not _filter_oof_spine(rec, tp, dendrite_id=did):
            continue
        d = float(
            np.hypot(float(rec["x"]) - x, float(rec["y"]) - y)
            + 0.25 * abs(float(rec["z"]) - z)
        )
        if d < best_d:
            best_d = d
            best = {
                "spine_id": sid,
                "x": float(rec["x"]),
                "y": float(rec["y"]),
                "z": float(rec["z"]),
                "dendrite_id": did,
            }
    return best


def _estimate_shift_at(
    anchor_rows: List[dict],
    x2: float,
    y2: float,
    z2: float,
) -> Tuple[float, float, float]:
    sum_w = 0.0
    sum_dx = sum_dy = sum_dz = 0.0
    for a in anchor_rows:
        d3 = float(
            np.sqrt((a["x2"] - x2) ** 2 + (a["y2"] - y2) ** 2 + (a["z2"] - z2) ** 2)
        )
        if d3 > LOCAL_REG_WINDOW_PX:
            continue
        w = 1.0 / (d3 + 1.0)
        sum_w += w
        sum_dx += w * float(a["dx"])
        sum_dy += w * float(a["dy"])
        sum_dz += w * float(a["dz"])
    if sum_w <= 0.0:
        return 0.0, 0.0, 0.0
    return float(sum_dx / sum_w), float(sum_dy / sum_w), float(sum_dz / sum_w)


def _build_anchor_rows(t1_spine_id: str, lineage: Optional[dict]) -> List[dict]:
    if not lineage:
        return []
    t1_tp = _STATE.t1_timepoint
    t1_lookup = _STATE.spine_lookup.get(t1_tp) or {}
    if t1_spine_id not in t1_lookup:
        return []
    r1 = t1_lookup[t1_spine_id]
    rows: List[dict] = []
    for tp, mem in (lineage.get("members") or {}).items():
        if tp == t1_tp:
            continue
        sid = str(mem.get("spine_id", ""))
        lookup = _STATE.spine_lookup.get(tp) or {}
        if sid not in lookup:
            continue
        r2 = lookup[sid]
        rows.append(
            {
                "tp": tp,
                "x2": float(r2["x"]),
                "y2": float(r2["y"]),
                "z2": float(r2["z"]),
                "dx": float(r1["x"]) - float(r2["x"]),
                "dy": float(r1["y"]) - float(r2["y"]),
                "dz": float(r1["z"]) - float(r2["z"]),
            }
        )
    return rows


def _resolve_positions(t1_spine_id: str) -> Dict[str, dict]:
    t1_tp = _STATE.t1_timepoint
    t1_lookup = _STATE.spine_lookup.get(t1_tp) or {}
    if t1_spine_id not in t1_lookup:
        raise ValueError(f"T1 spine '{t1_spine_id}' not found.")
    base = t1_lookup[t1_spine_id]
    bx, by, bz = float(base["x"]), float(base["y"]), float(base["z"])
    bd = str(base.get("dendrite_id") or "")
    lineage = _lineage_for_t1_spine(t1_spine_id)
    active_link = _link_by_id(_STATE.active_link_id) if _STATE.active_link_id else None
    anchor_rows = _build_anchor_rows(t1_spine_id, lineage)

    positions: Dict[str, dict] = {}
    shifts: Dict[str, Tuple[float, float, float]] = {}

    for tp in _STATE.timepoint_names:
        if tp == t1_tp:
            positions[tp] = {
                "spine_id": t1_spine_id,
                "x": bx,
                "y": by,
                "z": bz,
                "dendrite_id": bd,
                "source": "t1_base",
            }
            shifts[tp] = (0.0, 0.0, 0.0)
            continue

        lookup = _STATE.spine_lookup.get(tp) or {}
        mem = (lineage.get("members") or {}).get(tp) if lineage else None
        if not mem and active_link:
            link_ids = active_link.get("members", {}).get(tp, [])
            if link_ids:
                lookup = _STATE.spine_lookup.get(tp) or {}
                linked = _linked_dendrites(t1_tp, bd, tp) if bd else set(str(x) for x in link_ids)
                sx, sy, sz = _estimate_shift_at(anchor_rows, bx, by, bz)
                near = _nearest_spine(lookup, bx + sx, by + sy, bz + sz, tp=tp, dendrite_ids=linked or None)
                if near:
                    positions[tp] = {**near, "source": "link_nearest"}
                    continue
        if mem and str(mem.get("spine_id", "")) in lookup:
            sid = str(mem["spine_id"])
            rec = lookup[sid]
            positions[tp] = {
                "spine_id": sid,
                "x": float(rec["x"]),
                "y": float(rec["y"]),
                "z": float(rec["z"]),
                "dendrite_id": str(rec.get("dendrite_id") or ""),
                "source": "registry",
            }
            continue

        linked = _linked_dendrites(t1_tp, bd, tp) if bd else set()
        sx, sy, sz = _estimate_shift_at(anchor_rows, bx, by, bz)
        shifts[tp] = (sx, sy, sz)
        px, py, pz = bx + sx, by + sy, bz + sz
        near = _nearest_spine(lookup, px, py, pz, tp=tp, dendrite_ids=linked or None)
        if near:
            positions[tp] = {
                **near,
                "source": "nearest_dendrite" if linked else "nearest_coord",
            }
        else:
            positions[tp] = {
                "spine_id": None,
                "x": px,
                "y": py,
                "z": pz,
                "dendrite_id": "",
                "source": "coord_fallback",
            }

    _STATE.local_shifts = shifts
    return positions


def _stack_for(tp: str) -> np.ndarray:
    if tp in _STATE._stacks:
        return _STATE._stacks[tp]
    tiff = (_STATE.files.get(tp) or {}).get("tiff", "")
    if not tiff:
        raise HTTPException(status_code=400, detail=f"No TIFF for timepoint '{tp}'.")
    stack = baseline_adapter.load_stack(Path(tiff))
    arr = np.asarray(stack)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    _STATE._stacks[tp] = arr
    return arr


def _norm_u8(plane: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo = float(np.percentile(plane, 1.0))
    hi = float(np.percentile(plane, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((plane.astype(float) - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8), lo, hi


def _render_zoom(
    tp: str,
    x: float,
    y: float,
    z: int,
    *,
    size: int,
    z_plane: Optional[int],
    focus_pos: Optional[dict] = None,
    anchor_pos: Optional[dict] = None,
) -> dict:
    stack = _stack_for(tp)
    zp = int(z_plane) if z_plane is not None else int(round(float(z)))
    plane, meta = crop_service.xy_plane_at_stack_z(
        stack,
        z_plane=zp,
        x=float(x),
        y=float(y),
        width=int(size),
        height=int(size),
    )
    u8, lo, hi = _norm_u8(plane)
    zp_used = int(meta["source_bounds"]["z0"])
    bounds = {
        "x0": float(meta["source_bounds"]["x0"]),
        "y0": float(meta["source_bounds"]["y0"]),
        "x1": float(meta["source_bounds"]["x1"]),
        "y1": float(meta["source_bounds"]["y1"]),
    }
    w, h = int(plane.shape[1]), int(plane.shape[0])
    return {
        "mode": "zoom",
        "timepoint": tp,
        "z_plane": zp_used,
        "z_max": max(int(stack.shape[0]) - 1, 0),
        "width": w,
        "height": h,
        "pixels": u8.flatten().tolist(),
        "intensity_min": lo,
        "intensity_max": hi,
        "focus_x": float(x),
        "focus_y": float(y),
        "stack_bounds": bounds,
        "markers": _markers_for_panel(tp, bounds, w, h, focus_pos, anchor_pos=anchor_pos),
        "oof_overlays": _oof_overlays_for_panel(tp, bounds, w, h),
    }


def _render_fov(
    tp: str,
    center_x: float,
    center_y: float,
    *,
    z_plane: int,
    view_w: int,
    view_h: int,
    fov_span: float,
    focus_pos: Optional[dict] = None,
) -> dict:
    stack = _stack_for(tp)
    z_max = max(int(stack.shape[0]) - 1, 0)
    zp = int(np.clip(int(z_plane), 0, z_max))
    plane_full = np.asarray(stack[zp], dtype=float)
    h0, w0 = int(plane_full.shape[0]), int(plane_full.shape[1])
    half = float(fov_span) / 2.0
    x0 = int(np.clip(round(center_x - half), 0, max(w0 - 1, 0)))
    y0 = int(np.clip(round(center_y - half), 0, max(h0 - 1, 0)))
    x1 = int(np.clip(round(center_x + half), x0 + 1, w0))
    y1 = int(np.clip(round(center_y + half), y0 + 1, h0))
    crop = plane_full[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("FOV crop empty.")
    ys = np.linspace(0, crop.shape[0] - 1, max(1, int(view_h))).astype(int)
    xs = np.linspace(0, crop.shape[1] - 1, max(1, int(view_w))).astype(int)
    small = crop[np.ix_(ys, xs)]
    u8, lo, hi = _norm_u8(small)
    bounds = {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}
    w, h = int(small.shape[1]), int(small.shape[0])
    return {
        "mode": "fov",
        "timepoint": tp,
        "z_plane": zp,
        "z_max": z_max,
        "width": w,
        "height": h,
        "pixels": u8.flatten().tolist(),
        "intensity_min": lo,
        "intensity_max": hi,
        "focus_x": float(center_x),
        "focus_y": float(center_y),
        "fov_span": float(fov_span),
        "stack_width": w0,
        "stack_height": h0,
        "stack_bounds": bounds,
        "markers": _markers_for_panel(tp, bounds, w, h, focus_pos),
        "oof_overlays": _oof_overlays_for_panel(tp, bounds, w, h),
    }


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #
class LoadResponse(BaseModel):
    animal_id: str
    fov: int
    respan_root: str
    t1_timepoint: str
    timepoint_names: List[str]
    t1_spine_count: int
    lineage_count: int
    active_link_id: str = ""
    global_mode: bool = False
    queue_count: int = 0
    cross_queue_count: int = 0
    registry_path: str = ""
    registry_mtime: str = ""
    decisions_mtime: str = ""
    review_progress: dict = Field(default_factory=dict)
    oof_segment_count: int = 0
    phase_index: int = 0
    phase_count: int = 0
    anchor_timepoint: str = ""
    review_mode: str = REVIEW_MODE_LINEAGE
    orphan_queue_counts: List[int] = Field(default_factory=list)
    phase_kind: str = "lineage"
    coverage: dict = Field(default_factory=dict)
    duplicate_count: int = 0
    unreviewed_count: int = 0
    spine_ids: List[str] = Field(default_factory=list)
    all_fovs_complete: bool = False
    results_final_dir: str = ""
    pending_fovs: List[int] = Field(default_factory=list)
    message: str = ""


class T1SpinesResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[dict]


class SelectSpineRequest(BaseModel):
    t1_spine_id: str


class SelectSpineResponse(BaseModel):
    t1_spine_id: str
    lineage_id: str = ""
    pre_mid_score: Optional[float] = None
    mid_spine_id: str = ""
    cross_dendrite: bool = False
    queue_mode: str = "main"
    local_registration: bool = False
    phase_index: int = 0
    anchor_timepoint: str = ""
    review_mode: str = REVIEW_MODE_LINEAGE
    positions: Dict[str, dict] = Field(default_factory=dict)
    phase_kind: str = "lineage"
    conflict: dict = Field(default_factory=dict)
    lineage_a_key: str = ""
    lineage_b_key: str = ""
    positions_b: Dict[str, dict] = Field(default_factory=dict)
    coverage: dict = Field(default_factory=dict)
    message: str = ""


class QueueModeRequest(BaseModel):
    mode: str = "main"


class PanelRequest(BaseModel):
    timepoint: str
    mode: str = "zoom"
    x: float
    y: float
    z: float = 0.0
    z_plane: Optional[int] = None
    zoom_size: int = 160
    view_width: int = 360
    view_height: int = 260
    fov_span: float = 900.0
    anchor_x: Optional[float] = None
    anchor_y: Optional[float] = None
    lineage_row: str = "a"


class PanelResponse(BaseModel):
    mode: str
    timepoint: str
    z_plane: int
    z_max: int
    width: int
    height: int
    pixels: List[int]
    intensity_min: float
    intensity_max: float
    focus_x: float
    focus_y: float
    fov_span: float = 0.0
    stack_width: int = 0
    stack_height: int = 0
    stack_bounds: dict = Field(default_factory=dict)
    markers: List[dict] = Field(default_factory=list)
    oof_overlays: List[dict] = Field(default_factory=list)


class SetSpineRequest(BaseModel):
    timepoint: str
    spine_id: str
    lineage_row: str = ""


class SetFateRequest(BaseModel):
    timepoint: str
    fate: str = ""
    review_mode: str = ""
    lineage_row: str = ""


class SetFateAllRequest(BaseModel):
    fate: str = "ignore"
    review_mode: str = ""
    lineage_row: str = ""


class ReviewModeRequest(BaseModel):
    mode: str = REVIEW_MODE_LINEAGE


class ResolveConflictRequest(BaseModel):
    keep: str = "a"


class SaveConflictEditsRequest(BaseModel):
    row: str = ""  # "a", "b", or empty = both


class ConflictFocusRequest(BaseModel):
    row: str = "a"


class TagFalsePositiveRequest(BaseModel):
    timepoint: str
    tag: str = "artifact"
    lineage_row: str = ""


class ClearTpRequest(BaseModel):
    timepoint: str
    lineage_row: str = ""


class ClearAllTpsRequest(BaseModel):
    lineage_row: str = ""


class OofSegmentRequest(BaseModel):
    segment_id: Optional[str] = None
    timepoint: str
    shape: str = "rect"
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    points: Optional[List[List[float]]] = None
    action: str = "upsert"
    apply_timepoints: Optional[List[str]] = None


class OofTpBoundsRequest(BaseModel):
    segment_id: str
    timepoint: str
    shape: str = "rect"
    x0: Optional[float] = None
    y0: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    points: Optional[List[List[float]]] = None
    action: str = "update"
    propagate_shift: bool = False   # if True, apply same delta to all other TPs


class FocusRequest(BaseModel):
    timepoint: str
    stack_x: float
    stack_y: float
    z_plane: Optional[int] = None
    lineage_row: str = ""


class AddManualSpineRequest(BaseModel):
    timepoint: str
    stack_x: float
    stack_y: float
    z_plane: Optional[int] = None
    lineage_row: str = ""


class RestorePositionsRequest(BaseModel):
    positions: Dict[str, dict] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post("/load", response_model=LoadResponse)
def load_from_animal(
    fov: int = 1,
    link_id: Optional[str] = None,
    timepoints: str = "",
) -> LoadResponse:
    try:
        from . import timepoint_selection

        cfg = animal_config.load_config()
        respan = animal_config.require_respan_root(cfg)
        requested = timepoint_selection.parse_timepoint_query(timepoints)
        saved = timepoint_selection.load(respan, fov)
        available = animal_layout.discover_available_timepoints(respan, fov)
        selected = animal_config.resolve_active_timepoints(
            cfg,
            available,
            requested=requested or None,
            saved=saved or None,
        )
        if requested:
            timepoint_selection.save(respan, fov, selected, animal_id=cfg.animal_id)
        inv = animal_layout.build_fov_inventory(
            respan, fov, animal_id=cfg.animal_id, selected_timepoints=selected
        )
        _STATE.reset()
        _STATE.animal_id = cfg.animal_id
        _STATE.skip_orphan_phases = bool(cfg.skip_orphan_phases)
        _STATE.fov = fov
        _STATE.respan_root = str(respan)
        _STATE.dendrite_links = dendrite_link_store.load_links(respan, fov)
        _STATE.active_link_id = str(link_id or "").strip()

        tps = [tp.name for tp in inv.timepoints if tp.csv_path]
        _STATE.timepoint_names = list(tps)
        _STATE.t1_timepoint = _STATE.timepoint_names[0] if _STATE.timepoint_names else ""

        for tp in inv.timepoints:
            if not tp.csv_path:
                continue
            df = baseline_adapter.load_spines(Path(tp.csv_path))
            _STATE.spine_dfs[tp.name] = df
            _STATE.spine_lookup[tp.name] = baseline_adapter.to_lookup(df)
            _STATE.files[tp.name] = {"csv": tp.csv_path, "tiff": tp.tiff_path or ""}

        catalog_path_str = _apply_catalog(respan)
        _inject_manual_from_decisions(respan, fov)
        _build_queues()
        _STATE.oof_segments = oof_segment_store.load_segments(respan, fov)
        _STATE.ignored_by_tp = ignored_spine_store.load_ignored(respan, fov)
        _STATE.review_progress = spine_lineage_store.load_progress(respan, fov)
        _STATE.queue_phase_index = int(_STATE.review_progress.get("phase_index", 0) or 0)
        if _STATE.skip_orphan_phases:
            _STATE.queue_phase_index = 0
        _build_phase_queues(respan)
        max_phase = max(0, _review_phase_count() - 1)
        if _STATE.queue_phase_index > max_phase:
            _STATE.queue_phase_index = max_phase
        _STATE.review_progress["phase_index"] = _STATE.queue_phase_index
        pre_tp = _STATE.t1_timepoint
        if _STATE.phase_queues:
            for sid in (_STATE.spine_lookup.get(pre_tp) or {}):
                if str(sid).startswith("manual_") and sid not in _STATE.phase_queues[0]:
                    _STATE.phase_queues[0].append(sid)
        saved_phase = int(_STATE.review_progress.get("phase_index", 0) or 0)
        qc_snap_msg = _snap_to_pending_qc_work()
        _rebuild_spine_id_list()
        _ensure_nonempty_phase(respan)
        if _STATE.queue_phase_index != saved_phase or qc_snap_msg:
            anchor = _current_anchor_timepoint()
            spine_lineage_store.save_progress(
                respan,
                _STATE.fov,
                index=0,
                reviewed_ids=list(_STATE.review_progress.get("reviewed_ids") or []),
                phase_index=_STATE.queue_phase_index,
                anchor_timepoint=anchor,
            )
            _STATE.review_progress = spine_lineage_store.load_progress(respan, _STATE.fov)
        manual_n = len([
            sid for tp in _STATE.spine_lookup for sid in _STATE.spine_lookup[tp]
            if str(sid).startswith("manual_")
        ])

        reg = _find_registry_csv(respan, fov)
        anchor = _current_anchor_timepoint()
        phase_n = _review_phase_count()
        lc = _lineage_phase_count()
        orphan_counts = (
            [len(_STATE.phase_queues[i]) for i in range(1, min(lc, len(_STATE.phase_queues)))]
            if not _STATE.skip_orphan_phases and len(_STATE.phase_queues) > 1
            else []
        )
        dup_n = len(_STATE.phase_queues[lc]) if len(_STATE.phase_queues) > lc else 0
        unrev_n = len(_STATE.phase_queues[lc + 1]) if len(_STATE.phase_queues) > lc + 1 else 0
        kind = _current_phase_kind()
        if kind == "duplicate":
            phase_label = f"duplicate conflicts ({len(_STATE.t1_spine_ids)})"
        elif kind == "unreviewed":
            phase_label = f"unreviewed spines ({len(_STATE.t1_spine_ids)})"
        else:
            phase_label = f"{anchor} ({len(_STATE.t1_spine_ids)} spines)"
        msg_parts = [
            f"Phase {_STATE.queue_phase_index + 1}/{phase_n}: {phase_label}",
        ]
        if qc_snap_msg:
            msg_parts.append(qc_snap_msg)
        cov = _STATE.coverage_summary or {}
        if cov:
            msg_parts.append(
                f"Coverage: {cov.get('tagged', 0)}/{cov.get('catalog_total', 0)} tagged · "
                f"{cov.get('duplicate_unresolved', 0)} duplicates · {cov.get('unreviewed', 0)} unreviewed"
            )
        if dup_n and kind != "duplicate":
            msg_parts.append(f"Duplicates pending: {dup_n}.")
        if unrev_n and kind != "unreviewed":
            msg_parts.append(f"Unreviewed pending: {unrev_n}.")
        if _STATE.skip_orphan_phases:
            msg_parts.append("Orphan phases disabled (demo).")
        if orphan_counts:
            pending = ", ".join(
                f"{_STATE.timepoint_names[i+1]}:{orphan_counts[i]}"
                for i in range(len(orphan_counts))
                if orphan_counts[i] > 0
            )
            if pending:
                msg_parts.append(f"Orphans pending: {pending}.")
        msg_parts.append(
            f"Pre->mid ranked: {len(_STATE.spine_queue)}"
            + (f", cross-dend {len(_STATE.cross_dendrite_queue)}" if _STATE.cross_dendrite_queue else "")
        )
        if manual_n:
            msg_parts.append(f"{manual_n} manual spine(s) loaded.")
        if catalog_path_str:
            msg_parts.append(catalog_path_str)
        if _STATE.active_link_id:
            msg_parts.insert(0, f"Dendrite link {_STATE.active_link_id}.")
        else:
            msg_parts.insert(0, "Global mode - all pre spines by score.")
        if _STATE.dendrite_links:
            msg_parts.append(f"{len(_STATE.dendrite_links)} dendrite link(s) in workspace.")
        reg_mtime = ""
        dec_mtime = ""
        dec_path = respan / "_annotator" / f"fov{fov}" / "lineage_decisions.json"
        if reg is not None:
            _STATE.registry_path = str(reg)
            _STATE.lineages = _parse_registry(reg, _STATE.timepoint_names, fov=fov)
            reg_mtime = _registry_mtime_iso(reg)
            msg_parts.append(f"{len(_STATE.lineages)} lineages from registry.")
        else:
            msg_parts.append("No registry - using nearest-coordinate fallback.")
        if dec_path.is_file():
            dec_mtime = _registry_mtime_iso(dec_path)

        load_extra: dict = {}
        _attach_results_final(respan, load_extra)
        final_msg = ""
        if load_extra.get("results_final_dir"):
            final_msg = f" Results final: {load_extra['results_final_dir']}."
        elif load_extra.get("pending_fovs"):
            final_msg = (
                f" FOV(s) still pending: {', '.join(str(x) for x in load_extra['pending_fovs'])}."
            )

        return LoadResponse(
            animal_id=_STATE.animal_id,
            fov=fov,
            respan_root=_STATE.respan_root,
            t1_timepoint=_STATE.t1_timepoint,
            timepoint_names=_STATE.timepoint_names,
            t1_spine_count=len(_STATE.t1_spine_ids),
            lineage_count=len(_STATE.lineages),
            active_link_id=_STATE.active_link_id,
            global_mode=not bool(_STATE.active_link_id),
            queue_count=len(_STATE.spine_queue),
            cross_queue_count=len(_STATE.cross_dendrite_queue),
            registry_path=_STATE.registry_path,
            registry_mtime=reg_mtime,
            decisions_mtime=dec_mtime,
            review_progress=_STATE.review_progress,
            oof_segment_count=len(_STATE.oof_segments),
            phase_index=_STATE.queue_phase_index,
            phase_count=_review_phase_count(),
            anchor_timepoint=anchor,
            review_mode=_STATE.review_mode,
            orphan_queue_counts=orphan_counts,
            phase_kind=kind,
            coverage=cov,
            duplicate_count=dup_n,
            unreviewed_count=unrev_n,
            spine_ids=list(_STATE.t1_spine_ids),
            all_fovs_complete=bool(load_extra.get("all_fovs_complete")),
            results_final_dir=str(load_extra.get("results_final_dir") or ""),
            pending_fovs=list(load_extra.get("pending_fovs") or []),
            message=" ".join(msg_parts) + final_msg,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/t1-spines", response_model=T1SpinesResponse)
def list_t1_spines(offset: int = 0, limit: int = 100) -> T1SpinesResponse:
    lim = max(1, min(limit, 5000))
    ids = _STATE.t1_spine_ids[offset : offset + lim]
    anchor_tp = _current_anchor_timepoint()
    lookup = _STATE.spine_lookup.get(anchor_tp) or {}
    items = []
    for sid in ids:
        if _is_duplicate_phase():
            conf = _STATE.conflicts_by_id.get(sid) or {}
            items.append(
                {
                    "spine_id": sid,
                    "local_spine_id": str(conf.get("spine_id") or ""),
                    "anchor_timepoint": str(conf.get("timepoint") or anchor_tp),
                    "phase_index": _STATE.queue_phase_index,
                    "dendrite_id": "",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "pre_mid_score": None,
                    "mid_spine_id": "",
                    "cross_dendrite": False,
                    "lineage_a_key": (conf.get("lineage_keys") or ["", ""])[0]
                    if conf.get("lineage_keys")
                    else "",
                    "lineage_b_key": (conf.get("lineage_keys") or ["", ""])[1]
                    if conf.get("lineage_keys") and len(conf.get("lineage_keys")) > 1
                    else "",
                }
            )
            continue
        if _is_unreviewed_phase():
            tp, gid = spine_qc_store.parse_queue_id(sid)
            rec = (_STATE.spine_lookup.get(tp) or {}).get(gid, {})
            items.append(
                {
                    "spine_id": sid,
                    "local_spine_id": str(rec.get("local_spine_id") or gid),
                    "anchor_timepoint": tp or anchor_tp,
                    "phase_index": _STATE.queue_phase_index,
                    "dendrite_id": str(rec.get("dendrite_id") or ""),
                    "x": float(rec.get("x", 0)),
                    "y": float(rec.get("y", 0)),
                    "z": float(rec.get("z", 0)),
                    "pre_mid_score": None,
                    "mid_spine_id": "",
                    "cross_dendrite": False,
                }
            )
            continue
        rec = lookup.get(sid, {})
        if not rec and _STATE.queue_phase_index == 0:
            rec = (_STATE.spine_lookup.get(_STATE.t1_timepoint) or {}).get(sid, {})
        qitem = _queue_item(sid) if _STATE.queue_phase_index == 0 else None
        local_spine_id = str(rec.get("local_spine_id") or (qitem.get("local_pre_spine_id") if qitem else "") or "")
        items.append(
            {
                "spine_id": sid,
                "local_spine_id": local_spine_id,
                "anchor_timepoint": anchor_tp,
                "phase_index": _STATE.queue_phase_index,
                "dendrite_id": str(rec.get("dendrite_id") or ""),
                "x": float(rec.get("x", 0)),
                "y": float(rec.get("y", 0)),
                "z": float(rec.get("z", 0)),
                "pre_mid_score": float(qitem["pre_mid_score"]) if qitem else None,
                "mid_spine_id": str(qitem.get("mid_spine_id", "")) if qitem else "",
                "cross_dendrite": bool(qitem.get("cross_dendrite")) if qitem else False,
            }
        )
    return T1SpinesResponse(total=len(_STATE.t1_spine_ids), offset=offset, limit=lim, items=items)


@router.get("/queue")
def get_queue(mode: str = "main") -> dict:
    if mode == "cross":
        return {"mode": "cross", "total": len(_STATE.cross_dendrite_queue), "items": _STATE.cross_dendrite_queue}
    return {"mode": "main", "total": len(_STATE.spine_queue), "items": _STATE.spine_queue}


@router.post("/queue-mode")
def set_queue_mode(req: QueueModeRequest) -> dict:
    _STATE.queue_mode = "cross" if req.mode == "cross" else "main"
    _rebuild_spine_id_list()
    return {
        "queue_mode": _STATE.queue_mode,
        "total": len(_STATE.t1_spine_ids),
        "spine_ids": list(_STATE.t1_spine_ids),
    }


@router.post("/select-spine", response_model=SelectSpineResponse)
def select_spine(req: SelectSpineRequest) -> SelectSpineResponse:
    try:
        _STATE.review_mode = REVIEW_MODE_LINEAGE
        spine_id = str(req.t1_spine_id)
        anchor_tp = _current_anchor_timepoint()
        phase_idx = _STATE.queue_phase_index
        respan = _respan_path()

        if _is_duplicate_phase():
            conf = _STATE.conflicts_by_id.get(spine_id)
            if not conf:
                raise ValueError(f"Unknown conflict '{spine_id}'.")
            keys = list(conf.get("lineage_keys") or [])
            if len(keys) < 2:
                raise ValueError("Conflict must involve two lineages.")
            lin_a = spine_lineage_store.get_lineage_by_key(respan, _STATE.fov, keys[0])
            lin_b = spine_lineage_store.get_lineage_by_key(respan, _STATE.fov, keys[1])
            _STATE.conflict_meta = conf
            _STATE.lineage_a_key = keys[0]
            _STATE.lineage_b_key = keys[1]
            _STATE.conflict_focus_row = "a"
            _STATE.positions = spine_qc_store.positions_from_lineage(
                lin_a, _STATE.timepoint_names, _STATE.spine_lookup
            )
            _STATE.positions_b = spine_qc_store.positions_from_lineage(
                lin_b, _STATE.timepoint_names, _STATE.spine_lookup
            )
            _STATE.local_shifts = {}
            _STATE.active_t1_spine_id = spine_id
            conf_tp = str(conf.get("timepoint") or "")
            comp_tp = str(conf.get("comparison_timepoint") or conf_tp)
            conf_sid = str(conf.get("spine_id") or "")
            pos_a = _STATE.positions.get(comp_tp) or {}
            pos_b = _STATE.positions_b.get(comp_tp) or {}
            sid_a = str(pos_a.get("spine_id") or "—")
            sid_b = str(pos_b.get("spine_id") or "—")
            loc_a = str(pos_a.get("local_spine_id") or "")
            loc_b = str(pos_b.get("local_spine_id") or "")
            dup_note = f"duplicate {conf_sid} @ {conf_tp}" if conf_tp != comp_tp else f"duplicate {conf_sid}"
            return SelectSpineResponse(
                t1_spine_id=spine_id,
                phase_index=phase_idx,
                anchor_timepoint=anchor_tp,
                review_mode=_STATE.review_mode,
                positions=_positions_response(),
                message=(
                    f"{dup_note} · compare at {comp_tp} (orange column) · "
                    f"A ({keys[0]}): {loc_a + ' ' if loc_a else ''}{sid_a} · "
                    f"B ({keys[1]}): {loc_b + ' ' if loc_b else ''}{sid_b}"
                ),
                **_conflict_response_extra(),
            )

        if _is_unreviewed_phase():
            tp, gid = spine_qc_store.parse_queue_id(spine_id)
            if not tp or not gid:
                raise ValueError(f"Invalid unreviewed queue id '{spine_id}'.")
            _STATE.review_mode = REVIEW_MODE_LINEAGE
            _STATE.lineage_a_key = ""
            _STATE.lineage_b_key = ""
            _STATE.conflict_meta = None
            _STATE.conflict_focus_row = "a"
            positions = mtp_spine_matching.build_lineage_positions_from_anchor(
                anchor_tp=tp,
                anchor_spine_id=gid,
                timepoint_names=_STATE.timepoint_names,
                spine_lookup=_STATE.spine_lookup,
                spine_dfs=_STATE.spine_dfs,
                cross_links=_STATE.dendrite_links,
                allow_cross_dendrite=_STATE.allow_cross_dendrite,
                tiff_paths=_tiff_paths_map(),
            )
            positions, shifts, reg_applied = mtp_spine_matching.apply_local_registration(
                positions,
                pre_tp=tp,
                pre_spine_id=gid,
                timepoint_names=_STATE.timepoint_names,
                spine_lookup=_STATE.spine_lookup,
                cross_links=_STATE.dendrite_links,
                linked_dendrites_fn=_linked_dendrites,
                oof_segments=_STATE.oof_segments,
            )
            _STATE.local_shifts = shifts
            positions = _apply_fate_rules(positions)
            _STATE.positions_b = {}
            _STATE.active_t1_spine_id = spine_id
            _STATE.positions = positions
            local = str((_STATE.spine_lookup.get(tp) or {}).get(gid, {}).get("local_spine_id") or "")
            return SelectSpineResponse(
                t1_spine_id=spine_id,
                phase_index=phase_idx,
                anchor_timepoint=tp,
                review_mode=_STATE.review_mode,
                local_registration=reg_applied,
                positions=_positions_response(),
                phase_kind="unreviewed",
                coverage=dict(_STATE.coverage_summary or {}),
                message=(
                    f"Unreviewed {local + ' ' if local else ''}{gid} at {tp} — "
                    f"click spines to match · Clear/False+ · Save to tag."
                ),
            )

        qitem = _queue_item(spine_id) if phase_idx == 0 else None
        mid_id = str(qitem.get("mid_spine_id", "")) if qitem else None
        cross = bool(qitem.get("cross_dendrite")) if qitem else False
        lin = _lineage_for_t1_spine(spine_id)
        reg_members = (lin.get("members") or {}) if lin else None
        if phase_idx == 0:
            positions = mtp_spine_matching.build_lineage_positions(
                pre_spine_id=spine_id,
                mid_spine_id=mid_id,
                timepoint_names=_STATE.timepoint_names,
                spine_lookup=_STATE.spine_lookup,
                spine_dfs=_STATE.spine_dfs,
                cross_links=_STATE.dendrite_links,
                registry_members=reg_members,
                allow_cross_dendrite=cross or _STATE.allow_cross_dendrite,
                tiff_paths=_tiff_paths_map(),
            )
            reg_tp = anchor_tp
        else:
            existing = spine_lineage_store.get_lineage_by_key(respan, _STATE.fov, spine_id)
            if existing and str(existing.get("pre_timepoint") or "") == anchor_tp:
                positions = spine_qc_store.positions_from_lineage(
                    existing, _STATE.timepoint_names, _STATE.spine_lookup
                )
            else:
                positions = mtp_spine_matching.build_lineage_positions_from_anchor(
                    anchor_tp=anchor_tp,
                    anchor_spine_id=spine_id,
                    timepoint_names=_STATE.timepoint_names,
                    spine_lookup=_STATE.spine_lookup,
                    spine_dfs=_STATE.spine_dfs,
                    cross_links=_STATE.dendrite_links,
                    allow_cross_dendrite=_STATE.allow_cross_dendrite,
                    tiff_paths=_tiff_paths_map(),
                )
            reg_tp = anchor_tp
        positions, shifts, reg_applied = mtp_spine_matching.apply_local_registration(
            positions,
            pre_tp=reg_tp,
            pre_spine_id=spine_id,
            timepoint_names=_STATE.timepoint_names,
            spine_lookup=_STATE.spine_lookup,
            cross_links=_STATE.dendrite_links,
            linked_dendrites_fn=_linked_dendrites,
            oof_segments=_STATE.oof_segments,
        )
        _STATE.local_shifts = shifts
        if qitem and len(_STATE.timepoint_names) > 1:
            mid_tp = _STATE.timepoint_names[1]
            if mid_tp in positions:
                positions[mid_tp]["score"] = qitem.get("pre_mid_score")
                positions[mid_tp]["cross_dendrite"] = cross
        positions = _apply_fate_rules(positions)
        _STATE.active_t1_spine_id = spine_id
        _STATE.positions = positions
        _STATE.positions_b = {}
        _STATE.conflict_meta = None
        return SelectSpineResponse(
            t1_spine_id=spine_id,
            lineage_id=str(lin.get("lineage_id", "")) if lin else "",
            pre_mid_score=float(qitem["pre_mid_score"]) if qitem else None,
            mid_spine_id=mid_id or "",
            cross_dendrite=cross,
            queue_mode=_STATE.queue_mode,
            local_registration=reg_applied,
            phase_index=phase_idx,
            anchor_timepoint=anchor_tp,
            review_mode=_STATE.review_mode,
            positions=_positions_response(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _jump_to_qc_phase_if_pending() -> Optional[str]:
    """Unlike _snap_to_pending_qc_work (which only corrects a phase index already
    at/past the QC phases), this jumps forward from anywhere -- including phase 0
    -- the moment there's nothing left to do in the current phase.
    """
    dup_idx, unrev_idx = _qc_phase_indices()
    dup_n, unrev_n = _pending_qc_counts()
    if dup_n > 0:
        _STATE.queue_phase_index = dup_idx
        _rebuild_spine_id_list()
        return f"Jumped to duplicate conflicts ({dup_n} pending)."
    if unrev_n > 0:
        _STATE.queue_phase_index = unrev_idx
        _rebuild_spine_id_list()
        return f"Jumped to unreviewed sweep ({unrev_n} pending)."
    return None


def _next_unresolved_index_in_current_phase() -> Optional[int]:
    """Index of the next spine in the current phase queue that isn't reviewed yet."""
    ids = _STATE.t1_spine_ids
    if not ids:
        return None
    reviewed = set(_STATE.review_progress.get("reviewed_ids") or [])
    cur = ids.index(_STATE.active_t1_spine_id) if _STATE.active_t1_spine_id in ids else -1
    n = len(ids)
    for offset in range(1, n + 1):
        idx = (cur + offset) % n
        if ids[idx] not in reviewed:
            return idx
    return None


@router.post("/jump-unresolved")
def jump_unresolved() -> dict:
    """Jump to the next thing needing a decision: an un-reviewed spine in the
    current phase, else the earliest QC phase (duplicates, then unreviewed
    detections) that still has pending work.
    """
    try:
        respan = _respan_path()
        idx = _next_unresolved_index_in_current_phase()
        if idx is not None:
            resp = select_spine(SelectSpineRequest(t1_spine_id=_STATE.t1_spine_ids[idx]))
        else:
            snap_msg = _jump_to_qc_phase_if_pending()
            resp = None
            if snap_msg:
                _ensure_nonempty_phase(respan)
                if _STATE.t1_spine_ids:
                    resp = select_spine(SelectSpineRequest(t1_spine_id=_STATE.t1_spine_ids[0]))
                    resp.message = f"{snap_msg} {resp.message}".strip()
            if resp is None:
                return {
                    **_select_spine_response(
                        message="Nothing unresolved — full coverage in this FOV."
                    ).model_dump(),
                    "spine_ids": list(_STATE.t1_spine_ids),
                    "phase_index": _STATE.queue_phase_index,
                }
        out = resp.model_dump()
        out["spine_ids"] = list(_STATE.t1_spine_ids)
        out["phase_index"] = _STATE.queue_phase_index
        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class AcceptSuggestionRequest(BaseModel):
    timepoint: str
    lineage_row: str = ""


@router.post("/accept-suggestion", response_model=SelectSpineResponse)
def accept_suggestion(req: AcceptSuggestionRequest) -> SelectSpineResponse:
    """Confirm the algorithm's current candidate at this TP without a click.

    If the TP already holds a resolved spine_id (the common case -- the
    algorithm's nearest-match is pre-filled), this just re-applies it. If the
    focus was moved to a bare coordinate (no labeled spine under it), this
    looks up the nearest spine there, exactly like clicking that marker would.
    """
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        if tp not in _STATE.timepoint_names:
            raise ValueError(f"Unknown timepoint '{tp}'.")
        pos_map = _active_positions()
        pos = dict(pos_map.get(tp) or {})
        sid = str(pos.get("spine_id") or "").strip()
        lookup = _STATE.spine_lookup.get(tp) or {}
        if not sid or sid not in lookup:
            x, y, z = pos.get("x"), pos.get("y"), pos.get("z")
            if x is None or y is None:
                raise ValueError(f"No candidate to accept at '{tp}'.")
            near = _nearest_spine(lookup, float(x), float(y), float(z or 0), tp=tp)
            if not near:
                raise ValueError(f"No candidate to accept at '{tp}'.")
            pos_map[tp] = {**near, "source": "accepted_suggestion", "fate": None}
        else:
            rec = lookup[sid]
            pos_map[tp] = {
                "spine_id": sid,
                "x": float(rec["x"]),
                "y": float(rec["y"]),
                "z": float(rec["z"]),
                "dendrite_id": str(rec.get("dendrite_id") or ""),
                "source": "accepted_suggestion",
                "fate": None,
            }
        contiguity_error = _check_contiguity_would_violate(pos_map, _STATE.timepoint_names)
        if contiguity_error:
            raise ValueError(contiguity_error)
        _set_active_positions(_apply_fate_rules(pos_map))
        return _select_spine_response(message=f"Accepted suggestion at {tp}.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/focus", response_model=SelectSpineResponse)
def update_focus(req: FocusRequest) -> SelectSpineResponse:
    """Click-to-focus: update one timepoint focus and refresh zoom alignment."""
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        if tp not in _STATE.timepoint_names:
            raise ValueError(f"Timepoint '{tp}' not active.")
        pos_map = _active_positions()
        pos = dict(pos_map.get(tp) or _STATE.positions.get(tp) or {})
        pos["x"] = float(req.stack_x)
        pos["y"] = float(req.stack_y)
        if req.z_plane is not None:
            pos["z"] = float(req.z_plane)
        lookup = _STATE.spine_lookup.get(tp) or {}
        near = _nearest_spine(lookup, pos["x"], pos["y"], pos["z"], tp=tp)
        if near:
            pos.update(near)
            pos["source"] = "click_nearest"
            pos["fate"] = None
        else:
            pos["spine_id"] = None
            pos["source"] = "click_coord"
            pos["fate"] = None
        pos_map[tp] = pos
        updated = _apply_fate_rules(pos_map)
        _set_active_positions(updated)
        return _select_spine_response(
            local_registration=bool(_STATE.local_shifts) and any(
                any(v != 0.0 for v in s) for s in _STATE.local_shifts.values()
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/panel", response_model=PanelResponse)
def render_panel(req: PanelRequest) -> PanelResponse:
    try:
        if req.timepoint not in _STATE.timepoint_names:
            raise ValueError(f"Unknown timepoint '{req.timepoint}'.")
        use_b = str(req.lineage_row or "a").strip().lower() == "b"
        pos_map = _STATE.positions_b if use_b and _is_duplicate_phase() else _STATE.positions
        if req.mode == "fov" and _is_duplicate_phase() and use_b:
            req_mode = "zoom"
        else:
            req_mode = req.mode
        panel_x, panel_y = float(req.x), float(req.y)
        panel_zoom = int(req.zoom_size)
        if req_mode == "fov":
            raw = _render_fov(
                req.timepoint,
                req.x,
                req.y,
                z_plane=int(req.z_plane if req.z_plane is not None else round(req.z)),
                view_w=int(req.view_width),
                view_h=int(req.view_height),
                fov_span=float(req.fov_span),
                focus_pos=pos_map.get(req.timepoint),
            )
        else:
            anchor = None
            pos = pos_map.get(req.timepoint)
            if req.anchor_x is not None and req.anchor_y is not None:
                anchor = {"x": float(req.anchor_x), "y": float(req.anchor_y)}
            elif pos:
                anchor = {"x": float(pos["x"]), "y": float(pos["y"])}
            raw = _render_zoom(
                req.timepoint,
                panel_x,
                panel_y,
                int(round(req.z)),
                size=panel_zoom,
                z_plane=req.z_plane,
                focus_pos=pos,
                anchor_pos=anchor,
            )
            bounds = raw.get("stack_bounds") or {}
            w, h = int(raw.get("width") or 0), int(raw.get("height") or 0)
            viewing = "b" if use_b else "a"
            raw["markers"] = _markers_for_panel_positions(
                pos_map, req.timepoint, bounds, w, h, pos, anchor_pos=anchor
            )
            raw["markers"].extend(
                _peer_conflict_markers(
                    req.timepoint, bounds, w, h, viewing_row=viewing
                )
            )
        return PanelResponse(**raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/add-manual-spine", response_model=SelectSpineResponse)
def add_manual_spine_click(req: AddManualSpineRequest) -> SelectSpineResponse:
    """Place a new manual spine at click coordinates (saved on Next → lineage/registry)."""
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = str(req.timepoint)
        if tp not in _STATE.timepoint_names:
            raise ValueError(f"Unknown timepoint '{tp}'.")
        pos_map = _active_positions()
        z = float(req.z_plane) if req.z_plane is not None else float(
            (pos_map.get(tp) or {}).get("z", 0)
        )
        pre_tp = _STATE.t1_timepoint
        did = _active_pre_dendrite_id() if tp != pre_tp else ""
        respan = _respan_path()
        existing = _all_manual_ids_for_tp(tp, respan, _STATE.fov)
        sid = manual_spine_store.make_manual_spine_id(tp, existing)
        rec = manual_spine_store.to_lookup_row(
            spine_id=sid,
            x=float(req.stack_x),
            y=float(req.stack_y),
            z=z,
            dendrite_id=did,
        )
        _STATE.spine_lookup.setdefault(tp, {})[sid] = rec

        if tp not in pos_map:
            pos_map[tp] = {}
        pos_map[tp] = {
            "spine_id": sid,
            "x": float(rec["x"]),
            "y": float(rec["y"]),
            "z": float(rec["z"]),
            "dendrite_id": str(rec.get("dendrite_id") or ""),
            "source": "manual_added",
            "fate": None,
        }
        if tp == pre_tp and sid not in _STATE.t1_spine_ids and not _is_duplicate_phase():
            _STATE.t1_spine_ids.append(sid)
        contiguity_error = _check_contiguity_would_violate(pos_map, _STATE.timepoint_names)
        if contiguity_error:
            raise ValueError(contiguity_error)
        updated = _apply_fate_rules(pos_map)
        _set_active_positions(updated)
        return _select_spine_response(
            message=f"Manual spine {sid} (saved when you click Next)",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/set-spine", response_model=SelectSpineResponse)
def set_spine(req: SetSpineRequest) -> SelectSpineResponse:
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        sid = str(req.spine_id).strip()
        lookup = _STATE.spine_lookup.get(tp) or {}
        if sid not in lookup:
            raise ValueError(f"Spine '{sid}' not found at '{tp}'.")
        # A spine already claimed by another lineage is allowed here -- the
        # "one spine, one lineage" invariant is enforced at export, not at
        # click time (see find_duplicate_conflicts / apply_conflict_resolution).
        # The claim just becomes a conflict the user resolves later, with both
        # complete lineages in front of her instead of a forced snap decision.
        claimed = _claimed_for_matching(tp)
        warning = (
            f"Spine '{sid}' at '{tp}' is already claimed by another lineage — "
            f"this creates a conflict to resolve later (export is blocked until then)."
            if sid in claimed and sid not in _current_lineage_spine_ids()
            else ""
        )
        rec = lookup[sid]
        pos_map = _active_positions()
        pos_map[tp] = {
            "spine_id": sid,
            "x": float(rec["x"]),
            "y": float(rec["y"]),
            "z": float(rec["z"]),
            "dendrite_id": str(rec.get("dendrite_id") or ""),
            "source": "manual",
            "fate": None,
        }
        contiguity_error = _check_contiguity_would_violate(pos_map, _STATE.timepoint_names)
        if contiguity_error:
            raise ValueError(contiguity_error)
        _set_active_positions(_apply_fate_rules(pos_map))
        return _select_spine_response(message=warning)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/clear-tp", response_model=SelectSpineResponse)
def clear_tp(req: ClearTpRequest) -> SelectSpineResponse:
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        pos_map = _active_positions()
        if tp not in pos_map and tp not in _STATE.positions:
            raise ValueError(f"Timepoint '{tp}' not active.")
        pos = dict(pos_map.get(tp) or {})
        pos["spine_id"] = None
        pos["fate"] = None
        pos["artifact_mode"] = None
        pos["removed_spine_id"] = None
        pos.pop("decision_scope", None)
        pos["source"] = "cleared"
        pos_map[tp] = pos
        if _STATE.review_mode == REVIEW_MODE_LINEAGE:
            pos_map = spine_lineage_store.release_unbound_matches(
                pos_map, _STATE.timepoint_names
            )
        _set_active_positions(_apply_fate_rules(pos_map))
        return _select_spine_response(
            message=f"Cleared {tp} — released spines return to matching pool.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/clear-all-tps", response_model=SelectSpineResponse)
def clear_all_tps(req: ClearAllTpsRequest) -> SelectSpineResponse:
    """Clear spine match and fate on every timepoint for the active lineage (no fate labels)."""
    try:
        _apply_conflict_edit_row(req.lineage_row)
        pos_map = _active_positions()
        for tp in _STATE.timepoint_names:
            if tp not in pos_map:
                continue
            pos = dict(pos_map[tp])
            pos["spine_id"] = None
            pos["fate"] = None
            pos["artifact_mode"] = None
            pos["removed_spine_id"] = None
            pos.pop("decision_scope", None)
            pos["source"] = "cleared"
            pos_map[tp] = pos
        _set_active_positions(_apply_fate_rules(pos_map))
        return _select_spine_response(
            message="All timepoints cleared (no fate).",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _apply_local_fate(pos: dict, fate: str) -> None:
    pos["fate"] = fate
    pos["spine_id"] = None
    pos["removed_spine_id"] = None
    pos["decision_scope"] = spine_lineage_store.DECISION_SCOPE_LOCAL
    pos["artifact_mode"] = None
    if fate == "artifact":
        pos["artifact_mode"] = spine_lineage_store.ARTIFACT_MODE_BLIND_SPOT
        pos["source"] = spine_lineage_store.SOURCE_LOCAL_ARTIFACT
    elif fate == "ignore":
        pos["source"] = spine_lineage_store.SOURCE_SINGLE_TP_FOCUS
    elif fate == "new":
        pos["source"] = spine_lineage_store.SOURCE_LOCAL_NEW
    elif fate == "lost":
        pos["source"] = spine_lineage_store.SOURCE_LOCAL_LOST
    else:
        pos["source"] = f"local_fate_{fate}"


@router.get("/review-mode")
def get_review_mode() -> dict:
    anchor = _current_anchor_timepoint()
    return {
        "review_mode": _STATE.review_mode,
        "anchor_timepoint": anchor,
        "timepoint_mode_supported": True,
    }


@router.post("/review-mode", response_model=SelectSpineResponse)
def set_review_mode(req: ReviewModeRequest) -> SelectSpineResponse:
    mode = str(req.mode or "").strip().lower()
    if mode not in (REVIEW_MODE_LINEAGE, REVIEW_MODE_TIMEPOINT):
        raise HTTPException(status_code=400, detail=f"Invalid review mode '{mode}'.")
    _STATE.review_mode = mode
    anchor = _current_anchor_timepoint()
    label = "Lineage mode" if mode == REVIEW_MODE_LINEAGE else f"Timepoint mode ({anchor}) · A/I/N/L"
    return SelectSpineResponse(
        t1_spine_id=_STATE.active_t1_spine_id,
        positions=_positions_response(),
        anchor_timepoint=anchor,
        review_mode=_STATE.review_mode,
        message=label,
    )


@router.post("/set-fate", response_model=SelectSpineResponse)
def set_fate(req: SetFateRequest) -> SelectSpineResponse:
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        fate = str(req.fate or "").strip().lower()
        review_mode = str(req.review_mode or _STATE.review_mode or REVIEW_MODE_LINEAGE).strip().lower()
        if review_mode not in (REVIEW_MODE_LINEAGE, REVIEW_MODE_TIMEPOINT):
            review_mode = REVIEW_MODE_LINEAGE
        _STATE.review_mode = review_mode
        pos_map = _active_positions()
        if tp not in pos_map and tp not in _STATE.positions:
            raise ValueError(f"Timepoint '{tp}' not active.")
        if fate and review_mode != REVIEW_MODE_TIMEPOINT:
            raise ValueError("Press S (or click the mode button) for timepoint mode, then use A / I / N / L.")
        active = _current_anchor_timepoint()
        if _is_duplicate_phase():
            active = _conflict_timepoint() or active
        if tp != active:
            raise ValueError(
                f"Timepoint mode applies to the current anchor '{active}' only (phase step)."
            )
        pos = dict(pos_map.get(tp) or {})
        if pos.get("fate_locked"):
            raise ValueError(f"Fate blocked at '{tp}' (lineage LOST at previous timepoint).")
        if pos.get("spine_id"):
            if review_mode == REVIEW_MODE_TIMEPOINT and fate:
                pos["spine_id"] = None
            else:
                raise ValueError("Clear spine match before setting fate.")
        if fate and fate not in TIMEPOINT_FATES:
            raise ValueError(f"Invalid fate '{fate}'.")
        allowed = _fate_options_for_tp(tp, pos_map, review_mode=review_mode)
        if fate and fate not in allowed:
            raise ValueError(f"Fate '{fate}' not allowed at '{tp}'.")
        respan = _respan_path()
        if fate and review_mode == REVIEW_MODE_TIMEPOINT and fate in ("artifact", "ignore"):
            spine_to_tag = pos.get("spine_id") or ""
            if pos.get("spine_id"):
                pos["spine_id"] = None
            spine_qc_tag_store.save_tag(
                respan, _STATE.fov,
                timepoint=tp,
                spine_id=spine_to_tag,
                tag=fate,
                animal_id=_STATE.animal_id
            )
            pos["fate"] = None
            pos["removed_spine_id"] = None
            pos["artifact_mode"] = None
            pos.pop("decision_scope", None)
            pos["source"] = "cleared"
        elif fate:
            _apply_local_fate(pos, fate)
        else:
            spine_to_clear = pos.get("spine_id") or ""
            spine_qc_tag_store.clear_tag(respan, _STATE.fov, timepoint=tp, spine_id=spine_to_clear, animal_id=_STATE.animal_id)
            pos["fate"] = None
            pos["spine_id"] = None
            pos["removed_spine_id"] = None
            pos["artifact_mode"] = None
            pos.pop("decision_scope", None)
            pos["source"] = "cleared"
        pos_map[tp] = pos
        _set_active_positions(_apply_fate_rules(pos_map))
        return _select_spine_response(
            anchor_timepoint=active,
            message=f"{fate.upper()} at {tp}" if fate else "Cleared",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/set-fate-all-tps", response_model=SelectSpineResponse)
def set_fate_all_tps(req: SetFateAllRequest) -> SelectSpineResponse:
    """Lineage mode: clear matches and apply local ignore on every timepoint (Tab → I workflow)."""
    try:
        _apply_conflict_edit_row(req.lineage_row)
        fate = str(req.fate or "").strip().lower()
        review_mode = str(req.review_mode or _STATE.review_mode or REVIEW_MODE_LINEAGE).strip().lower()
        if review_mode not in (REVIEW_MODE_LINEAGE, REVIEW_MODE_TIMEPOINT):
            review_mode = REVIEW_MODE_LINEAGE
        _STATE.review_mode = review_mode
        if review_mode != REVIEW_MODE_LINEAGE:
            raise ValueError("Ignore-all is for lineage mode only (Tab → I). In timepoint mode, I applies to the anchor TP.")
        if fate != "ignore":
            raise ValueError("Bulk all-timepoint fate supports IGNORE only.")
        pos_map = _active_positions()
        if not pos_map:
            raise ValueError("No active spine.")
        respan = _respan_path()
        applied: List[str] = []
        for tp in _STATE.timepoint_names:
            pos = dict(pos_map.get(tp) or {})
            if pos.get("fate_locked"):
                continue
            spine_to_tag = pos.get("spine_id") or ""
            if pos.get("spine_id"):
                pos["spine_id"] = None
            spine_qc_tag_store.save_tag(
                respan, _STATE.fov,
                timepoint=tp,
                spine_id=spine_to_tag,
                tag=fate,
                animal_id=_STATE.animal_id
            )
            pos["fate"] = None
            pos["removed_spine_id"] = None
            pos["artifact_mode"] = None
            pos.pop("decision_scope", None)
            pos["source"] = "cleared"
            pos_map[tp] = pos
            applied.append(tp)
        if not applied:
            raise ValueError("No timepoint available for bulk ignore (all blocked by lineage LOST).")
        _set_active_positions(_apply_fate_rules(pos_map))
        active = _current_anchor_timepoint()
        return _select_spine_response(
            review_mode=_STATE.review_mode,
            anchor_timepoint=active,
            message=f"IGNORE on {len(applied)} timepoint(s)",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tag-false-positive", response_model=SelectSpineResponse)
def tag_false_positive(req: TagFalsePositiveRequest) -> SelectSpineResponse:
    """Remove a lone matched spine as false positive (algorithm noise vs OOF location)."""
    try:
        _apply_conflict_edit_row(req.lineage_row)
        tp = req.timepoint
        tag = str(req.tag or "artifact").strip().lower()
        pos_map = _active_positions()
        if tp not in pos_map and tp not in _STATE.positions:
            raise ValueError(f"Timepoint '{tp}' not active.")
        if tag not in ("artifact", "ignore"):
            raise ValueError("False-positive tag must be 'artifact' or 'ignore'.")
        if not _false_positive_eligible(tp, pos_map):
            raise ValueError(
                f"False-positive at '{tp}' requires all other timepoints to be clear "
                f"(no match, no fate)."
            )
        pos = dict(pos_map.get(tp) or {})
        sid = str(pos.get("spine_id") or "").strip()
        if not sid:
            raise ValueError(
                "Select a spine first (click its label). "
                "Use timepoint mode (S) + A for blind-spot artifact."
            )
        pos["removed_spine_id"] = sid
        pos["artifact_mode"] = spine_lineage_store.ARTIFACT_MODE_FALSE_POSITIVE
        pos["fate"] = tag
        pos["spine_id"] = None
        pos["source"] = f"false_positive_{tag}"
        pos_map[tp] = pos
        _set_active_positions(_apply_fate_rules(pos_map))
        if tag == "ignore":
            msg = (
                f"False positive (OOF location) at {tp}: spine exists here but is not "
                f"trackable across other timepoints."
            )
        else:
            msg = (
                f"False positive (algorithm artifact) at {tp}: no real spine at this detection."
            )
        return _select_spine_response(message=msg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/undo-positions", response_model=SelectSpineResponse)
def undo_positions(req: RestorePositionsRequest) -> SelectSpineResponse:
    """Restore the previous in-session positions snapshot for the active spine."""
    try:
        if not _STATE.active_t1_spine_id:
            raise ValueError("No active spine.")
        restored: Dict[str, dict] = {}
        for tp in _STATE.timepoint_names:
            pos = dict((req.positions or {}).get(tp) or {})
            restored[tp] = {
                "spine_id": pos.get("spine_id"),
                "x": pos.get("x"),
                "y": pos.get("y"),
                "z": pos.get("z"),
                "dendrite_id": pos.get("dendrite_id"),
                "source": pos.get("source"),
                "fate": pos.get("fate"),
                "decision_scope": pos.get("decision_scope"),
                "artifact_mode": pos.get("artifact_mode"),
                "removed_spine_id": pos.get("removed_spine_id"),
                "continuity": pos.get("continuity"),
            }
        _STATE.positions = _apply_fate_rules(restored)
        return SelectSpineResponse(
            t1_spine_id=_STATE.active_t1_spine_id,
            positions=_positions_response(),
            message="Undid last change.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/confirm-lineage")
def confirm_lineage() -> dict:
    try:
        if not _STATE.active_t1_spine_id:
            raise ValueError("No active spine.")
        if _is_duplicate_phase():
            raise ValueError(
                "Duplicate conflict phase — use Keep top lineage or Keep bottom lineage."
            )
        _STATE.review_mode = REVIEW_MODE_LINEAGE
        respan = _respan_path()
        anchor_tp = _current_anchor_timepoint()
        per_tp = _save_positions_snapshot()
        if _is_unreviewed_phase():
            tp, anchor_global = spine_qc_store.parse_queue_id(_STATE.active_t1_spine_id)
            anchor_tp = tp or anchor_tp
        else:
            anchor_global = str(_STATE.active_t1_spine_id)
        anchor_local = (
            _STATE.catalog.to_local(anchor_tp, anchor_global)
            if _STATE.catalog
            else anchor_global
        )
        save_info = spine_lineage_store.save_decision(
            respan,
            _STATE.fov,
            animal_id=_STATE.animal_id,
            pre_spine_id=anchor_local,
            pre_timepoint=anchor_tp,
            timepoint_names=_STATE.timepoint_names,
            per_tp=per_tp,
            lineage_key=anchor_global,
            oof_segments=_STATE.oof_segments,
            ignored_by_tp=_STATE.ignored_by_tp,
            shifts=_STATE.local_shifts,
        )
        _STATE.coverage_summary = dict(save_info.get("coverage") or {})
        was_last_in_phase = False
        if _STATE.active_t1_spine_id in _STATE.t1_spine_ids:
            was_last_in_phase = (
                _STATE.t1_spine_ids.index(_STATE.active_t1_spine_id)
                >= len(_STATE.t1_spine_ids) - 1
            )
        idx = (
            _STATE.t1_spine_ids.index(_STATE.active_t1_spine_id)
            if _STATE.active_t1_spine_id in _STATE.t1_spine_ids
            else 0
        )
        prog = spine_lineage_store.mark_reviewed(
            respan,
            _STATE.fov,
            _STATE.active_t1_spine_id,
            idx,
            phase_index=_STATE.queue_phase_index,
            anchor_timepoint=anchor_tp,
        )
        _STATE.review_progress = prog
        _build_phase_queues(respan)
        _rebuild_spine_id_list()
        phase_adv = _advance_phase_if_complete(respan, force_advance=was_last_in_phase)
        out = {
            "saved": True,
            "pre_spine_id": _STATE.active_t1_spine_id,
            "anchor_timepoint": anchor_tp,
            "phase_index": _STATE.queue_phase_index,
            "review_progress": _STATE.review_progress,
            "queue_count": len(_STATE.t1_spine_ids),
            "spine_ids": list(_STATE.t1_spine_ids),
            "phase_advanced": bool(phase_adv),
            "last_seen_tp": save_info.get("last_seen_tp", ""),
            "lost_inferred": bool(save_info.get("lost_inferred")),
            "censored_from_tp": save_info.get("censored_from_tp", ""),
            "right_censored": bool(save_info.get("right_censored")),
        }
        if save_info.get("right_censored"):
            out["inference_message"] = (
                f"Right-censored from {save_info.get('censored_from_tp') or 'final TP'} "
                f"(last seen at {save_info.get('last_seen_tp') or '?'} — not classified as lost)."
            )
        elif save_info.get("censored_from_tp"):
            out["inference_message"] = (
                f"Censored from {save_info['censored_from_tp']} "
                f"(last seen at {save_info.get('last_seen_tp') or '?'} — loss not inferred)."
            )
        elif save_info.get("lost_inferred"):
            out["inference_message"] = (
                f"Lost after {save_info.get('last_seen_tp') or '?'} "
                f"(not seen in later timepoints)."
            )
        if phase_adv:
            out.update(phase_adv)
        out["review_mode"] = REVIEW_MODE_LINEAGE
        tps = _STATE.timepoint_names
        is_last_phase = _STATE.queue_phase_index >= _review_phase_count() - 1
        is_last_spine = (
            _STATE.active_t1_spine_id in _STATE.t1_spine_ids
            and _STATE.t1_spine_ids.index(_STATE.active_t1_spine_id)
            >= len(_STATE.t1_spine_ids) - 1
        )
        out["review_complete"] = bool(
            is_last_phase
            and is_last_spine
            and not phase_adv
            and int((_STATE.coverage_summary or {}).get("unreviewed", 1)) == 0
            and int((_STATE.coverage_summary or {}).get("duplicate_unresolved", 1)) == 0
        )
        out["review_mode"] = REVIEW_MODE_LINEAGE
        out["phase_kind"] = _current_phase_kind()
        out["coverage"] = dict(_STATE.coverage_summary or {})
        reg_path = str(save_info.get("registry_path") or "")
        if not reg_path:
            reg_path = str(respan / "_annotator" / f"fov{_STATE.fov}" / "spine_registry_wide.csv")
        out["registry_path"] = reg_path
        out["registry_mtime"] = _registry_mtime_iso(Path(reg_path))
        out["decisions_path"] = str(save_info.get("path") or "")
        out["disposition_path"] = str(save_info.get("disposition_path") or "")
        out["coverage"] = dict(_STATE.coverage_summary or {})
        out["message"] = (
            f"Saved {anchor_global} — registry updated ({len(_STATE.review_progress.get('reviewed_ids', []) or [])} spine(s))."
        )
        if out["review_complete"]:
            cov = _STATE.coverage_summary or {}
            _attach_results_final(respan, out)
            if out.get("all_fovs_complete") and out.get("results_final_dir"):
                out["message"] = (
                    f"Review complete for FOV {_STATE.fov}. "
                    f"All FOVs done — Results final: {out['results_final_dir']}"
                )
            elif out.get("pending_fovs"):
                pending = ", ".join(str(x) for x in out["pending_fovs"])
                out["message"] = (
                    f"Review complete for FOV {_STATE.fov} "
                    f"({cov.get('tagged', '?')}/{cov.get('catalog_total', '?')} tagged). "
                    f"Still pending: FOV {pending}."
                )
            else:
                out["message"] = (
                    f"Review complete — {cov.get('tagged', '?')}/{cov.get('catalog_total', '?')} spines tagged · "
                    f"0 duplicates · 0 unreviewed."
                )
        return out
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/save-conflict-edits")
def save_conflict_edits(req: SaveConflictEditsRequest) -> dict:
    """Persist edits to lineage A and/or B without resolving the duplicate."""
    try:
        if not _is_duplicate_phase():
            raise ValueError("Not in duplicate conflict phase.")
        if not _STATE.conflict_meta:
            raise ValueError("No active conflict loaded.")
        respan = _respan_path()
        row = str(req.row or "").strip().lower()
        saved_keys = _save_conflict_edits_row(row)
        if not saved_keys:
            raise ValueError("No lineage to save.")
        reg_path = respan / "_annotator" / f"fov{_STATE.fov}" / "spine_registry_wide.csv"
        _reload_conflict_lineages()
        _, summary = spine_qc_store.coverage_and_unreviewed(
            respan,
            _STATE.fov,
            _STATE.timepoint_names,
            ignored_by_tp=_STATE.ignored_by_tp,
            animal_id=_STATE.animal_id,
            rebuild=True,
        )
        _STATE.coverage_summary = dict(summary)
        label = (
            f"row {'B' if row.startswith('b') else 'A'}"
            if row in ("a", "b")
            else "both lineages"
        )
        return {
            "saved": True,
            "saved_lineage_keys": saved_keys,
            "registry_path": str(reg_path),
            "registry_mtime": _registry_mtime_iso(reg_path),
            "phase_kind": _current_phase_kind(),
            "coverage": dict(_STATE.coverage_summary or {}),
            "positions": _positions_response(),
            "positions_b": _positions_response_b(),
            "message": f"Saved {label} ({', '.join(saved_keys)}) — conflict still open. Edit the other row, then Keep A or Keep B.",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/resolve-conflict")
def resolve_conflict(req: ResolveConflictRequest) -> dict:
    try:
        if not _is_duplicate_phase():
            raise ValueError("Not in duplicate conflict phase.")
        conf = _STATE.conflict_meta or {}
        if not conf:
            raise ValueError("No active conflict loaded.")
        keep = str(req.keep or "a").strip().lower()
        keep_key = _STATE.lineage_a_key if keep.startswith("a") else _STATE.lineage_b_key
        if not keep_key:
            raise ValueError("Choose Keep top (A) or Keep bottom (B) lineage.")
        respan = _respan_path()
        _save_conflict_edits()
        tp = str(conf.get("timepoint") or "")
        sid = str(conf.get("spine_id") or "")
        result = spine_qc_store.apply_conflict_resolution(
            respan,
            _STATE.fov,
            animal_id=_STATE.animal_id,
            timepoint=tp,
            spine_id=sid,
            keep_lineage_key=keep_key,
            timepoint_names=_STATE.timepoint_names,
            ignored_by_tp=_STATE.ignored_by_tp,
        )
        _STATE.coverage_summary = dict(result.get("coverage") or {})
        conflict_id = str(_STATE.active_t1_spine_id or "")
        idx = (
            _STATE.t1_spine_ids.index(conflict_id)
            if conflict_id in _STATE.t1_spine_ids
            else 0
        )
        prog = spine_lineage_store.mark_reviewed(
            respan,
            _STATE.fov,
            conflict_id,
            idx,
            phase_index=_STATE.queue_phase_index,
            anchor_timepoint=tp,
        )
        _STATE.review_progress = prog
        was_last_in_phase = idx >= max(0, len(_STATE.t1_spine_ids) - 1)
        _build_phase_queues(respan)
        _rebuild_spine_id_list()
        phase_adv = _advance_phase_if_complete(respan, force_advance=was_last_in_phase)
        is_last_phase = _STATE.queue_phase_index >= _review_phase_count() - 1
        is_last_spine = not _STATE.t1_spine_ids or idx >= len(_STATE.t1_spine_ids) - 1
        review_complete = bool(
            is_last_phase
            and is_last_spine
            and not phase_adv
            and int((_STATE.coverage_summary or {}).get("unreviewed", 1)) == 0
            and int((_STATE.coverage_summary or {}).get("duplicate_unresolved", 1)) == 0
        )
        reg_path = str(
            respan / "_annotator" / f"fov{_STATE.fov}" / "spine_registry_wide.csv"
        )
        out = {
            "saved": True,
            "resolved": True,
            "keep_lineage_key": keep_key,
            "phase_index": _STATE.queue_phase_index,
            "phase_advanced": bool(phase_adv),
            "queue_count": len(_STATE.t1_spine_ids),
            "spine_ids": list(_STATE.t1_spine_ids),
            "review_complete": review_complete,
            "phase_kind": _current_phase_kind(),
            "coverage": dict(_STATE.coverage_summary or {}),
            "registry_path": reg_path,
            "registry_mtime": _registry_mtime_iso(Path(reg_path)),
            "disposition_path": result.get("disposition_path", ""),
            "message": f"Kept {keep_key} for {sid} @ {tp}.",
        }
        if phase_adv:
            out.update(phase_adv)
        if review_complete:
            cov = _STATE.coverage_summary or {}
            out["message"] = (
                f"Review complete — {cov.get('tagged', '?')}/{cov.get('catalog_total', '?')} spines tagged."
            )
            _attach_results_final(respan, out)
            if out.get("results_final_dir") and out.get("all_fovs_complete"):
                out["message"] = (
                    f"Review complete for FOV {_STATE.fov}. "
                    f"All FOVs done — Results final: {out['results_final_dir']}"
                )
            elif out.get("pending_fovs"):
                pending = ", ".join(str(x) for x in out["pending_fovs"])
                out["message"] = (
                    f"Review complete for FOV {_STATE.fov}. "
                    f"Still pending: FOV {pending}."
                )
        return out
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/conflict-focus")
def conflict_focus(req: ConflictFocusRequest) -> dict:
    r = str(req.row or "a").strip().lower()
    _STATE.conflict_focus_row = "b" if r.startswith("b") else "a"
    return _conflict_response_extra()


@router.get("/oof-segments")
def list_oof_segments() -> dict:
    return {"segments": _STATE.oof_segments}


@router.post("/oof-segment")
def upsert_oof_segment(req: OofSegmentRequest) -> dict:
    try:
        respan = _respan_path()
        tp = req.timepoint
        if req.action == "delete" and req.segment_id:
            deleted = oof_segment_store.delete_segment(
                respan, _STATE.fov, req.segment_id, animal_id=_STATE.animal_id
            )
            if not deleted:
                raise HTTPException(status_code=404, detail="OOF segment not found.")
            _STATE.oof_segments = oof_segment_store.load_segments(respan, _STATE.fov)
            _sync_ignored_from_oof(respan)
            _build_phase_queues(respan)
            _rebuild_spine_id_list()
            _ensure_nonempty_phase(respan)
            return {
                "deleted": True,
                "segment_id": req.segment_id,
                "segments": _STATE.oof_segments,
                "queue_count": len(_STATE.t1_spine_ids),
                "spine_ids": list(_STATE.t1_spine_ids),
            }

        if str(req.shape or "").lower() == "polygon":
            src_shape = oof_segment_store.polygon_shape(list(req.points or []))
        else:
            src_shape = oof_segment_store.rect_shape(req.x0, req.y0, req.x1, req.y1)

        did = _active_pre_dendrite_id()
        pre_tp = _STATE.t1_timepoint
        apply_tps = list(req.apply_timepoints or [tp])
        valid = set(_STATE.timepoint_names)
        apply_tps = [t for t in apply_tps if t in valid]
        if not apply_tps:
            apply_tps = [tp]
        shifts = _STATE.local_shifts or {}

        pre_shape = oof_segment_store.shape_between_timepoints(
            src_shape, tp, pre_tp, pre_tp=pre_tp, shifts=shifts
        )
        per_tp: Dict[str, dict] = {}
        ignored_add: Dict[str, List[str]] = {}
        for dst in apply_tps:
            dst_shape = oof_segment_store.shape_between_timepoints(
                src_shape, tp, dst, pre_tp=pre_tp, shifts=shifts
            )
            if dst != pre_tp:
                per_tp[dst] = dst_shape
            lookup = _STATE.spine_lookup.get(dst) or {}
            ignored_add[dst] = oof_segment_store.spines_inside_shape(
                dst_shape, dst, lookup
            )

        entry = oof_segment_store.upsert_segment(
            respan,
            _STATE.fov,
            animal_id=_STATE.animal_id,
            segment_id=req.segment_id,
            dendrite_id_pre=did,
            pre_tp=pre_tp,
            pre_shape=pre_shape,
            per_tp=per_tp,
        )
        _STATE.oof_segments = oof_segment_store.load_segments(respan, _STATE.fov)
        _STATE.ignored_by_tp = ignored_spine_store.add_ignored(
            respan,
            _STATE.fov,
            animal_id=_STATE.animal_id,
            by_tp_add=ignored_add,
        )
        _build_phase_queues(respan)
        _rebuild_spine_id_list()
        _ensure_nonempty_phase(respan)
        n_ignored = sum(len(v) for v in ignored_add.values())
        return {
            "segment": entry,
            "segments": _STATE.oof_segments,
            "ignored_spines": {k: v for k, v in ignored_add.items() if v},
            "ignored_count": n_ignored,
            "queue_count": len(_STATE.t1_spine_ids),
            "spine_ids": list(_STATE.t1_spine_ids),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/oof-segment-tp")
def update_oof_tp(req: OofTpBoundsRequest) -> dict:
    try:
        respan = _respan_path()
        if req.action == "delete":
            seg = oof_segment_store.update_tp_shape(
                respan,
                _STATE.fov,
                animal_id=_STATE.animal_id,
                segment_id=req.segment_id,
                tp=req.timepoint,
                shape=None,
            )
        else:
            if str(req.shape or "").lower() == "polygon":
                shape = oof_segment_store.polygon_shape(list(req.points or []))
            else:
                shape = oof_segment_store.rect_shape(
                    float(req.x0 or 0),
                    float(req.y0 or 0),
                    float(req.x1 or 0),
                    float(req.y1 or 0),
                )
            # Find old shape to compute delta for propagation
            segs_before = oof_segment_store.load_segments(respan, _STATE.fov)
            old_seg = next(
                (s for s in segs_before if str(s.get("segment_id","")) == req.segment_id),
                None,
            )
            seg = oof_segment_store.update_tp_shape(
                respan,
                _STATE.fov,
                animal_id=_STATE.animal_id,
                segment_id=req.segment_id,
                tp=req.timepoint,
                shape=shape,
            )
            # Propagate move delta to other TPs
            if req.propagate_shift and seg and old_seg and shape["shape"] == "rect":
                old_sh = oof_segment_store.segment_shape_at_timepoint(old_seg, req.timepoint)
                if old_sh and old_sh["shape"] == "rect":
                    dx = shape["x0"] - old_sh["x0"]
                    dy = shape["y0"] - old_sh["y0"]
                    per_tp = seg.get("per_tp") or {}
                    pre_tp_name = str(seg.get("pre_tp", ""))
                    all_tps = list(_STATE.timepoint_names)
                    for other_tp in all_tps:
                        if other_tp == req.timepoint:
                            continue
                        other_sh = oof_segment_store.segment_shape_at_timepoint(seg, other_tp)
                        if other_sh:
                            moved = oof_segment_store.translate_shape_by(other_sh, dx, dy)
                            oof_segment_store.update_tp_shape(
                                respan, _STATE.fov,
                                animal_id=_STATE.animal_id,
                                segment_id=req.segment_id,
                                tp=other_tp,
                                shape=moved,
                            )
                    seg = oof_segment_store.load_segments(respan, _STATE.fov)
                    seg = next(
                        (s for s in seg if str(s.get("segment_id","")) == req.segment_id), None
                    )
        if not seg:
            raise HTTPException(status_code=404, detail="OOF segment not found.")
        _STATE.oof_segments = oof_segment_store.load_segments(respan, _STATE.fov)
        _sync_ignored_from_oof(respan)
        _build_phase_queues(respan)
        _rebuild_spine_id_list()
        _ensure_nonempty_phase(respan)
        return {
            "segment": seg,
            "segments": _STATE.oof_segments,
            "queue_count": len(_STATE.t1_spine_ids),
            "spine_ids": list(_STATE.t1_spine_ids),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/state")
def viewer_state() -> dict:
    return {
        "animal_id": _STATE.animal_id,
        "fov": _STATE.fov,
        "t1_timepoint": _STATE.t1_timepoint,
        "timepoint_names": _STATE.timepoint_names,
        "active_t1_spine_id": _STATE.active_t1_spine_id,
        "positions": _positions_response(),
        "active_link_id": _STATE.active_link_id,
        "global_mode": not bool(_STATE.active_link_id),
        "queue_mode": _STATE.queue_mode,
        "queue_count": len(_STATE.spine_queue),
        "cross_queue_count": len(_STATE.cross_dendrite_queue),
        "local_registration": bool(_STATE.local_shifts),
        "oof_segments": _STATE.oof_segments,
        "review_progress": _STATE.review_progress,
    }


@router.get("/", response_class=HTMLResponse)
def viewer_page() -> Response:
    page_path = Path(__file__).with_name("mtp_viewer_page.html")
    return Response(
        content=page_path.read_text(encoding="utf-8"),
        media_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
