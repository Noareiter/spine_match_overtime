"""Persist longitudinal spine lineage decisions under respan/_annotator/."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import dendrite_link_store

FATE_TO_STATUS = {
    "lost": "lost",
    "artifact": "artifact",
    "new": "new",
    "ignore": "ignored",
}

ARTIFACT_MODE_BLIND_SPOT = "blind_spot"
ARTIFACT_MODE_FALSE_POSITIVE = "false_positive"

DECISION_SCOPE_LINEAGE = "lineage"
DECISION_SCOPE_LOCAL = "local"

SOURCE_SINGLE_TP_FOCUS = "single_tp_focus_tracking"
SOURCE_LOCAL_ARTIFACT = "local_artifact"
SOURCE_LOCAL_NEW = "local_new"
SOURCE_LOCAL_LOST = "local_lost"
SOURCE_RETURNED_TO_POOL = "returned_to_pool"
SOURCE_LOST_INFERRED_PREFIX = "lost_inferred_after_"

# Fates the inference pass writes itself. They are cleared and re-derived on
# every finalize so a re-saved lineage always follows the current rules.
_INFERRED_SOURCE_PREFIXES = (
    SOURCE_LOST_INFERRED_PREFIX,
    "lost_propagated_after_",
    "ignore_inferred_after_",
    "ignore_propagated_after_",
)


def decision_scope(td: dict) -> str:
    scope = str(td.get("decision_scope") or DECISION_SCOPE_LINEAGE).strip().lower()
    return scope if scope in (DECISION_SCOPE_LINEAGE, DECISION_SCOPE_LOCAL) else DECISION_SCOPE_LINEAGE


def is_local_decision(td: dict) -> bool:
    return decision_scope(td) == DECISION_SCOPE_LOCAL


def is_single_tp_focus_ignore(td: dict) -> bool:
    if str(td.get("fate") or "").strip().lower() != "ignore":
        return False
    src = str(td.get("source") or "").strip().lower()
    return is_local_decision(td) or src == SOURCE_SINGLE_TP_FOCUS


def _artifact_mode(td: dict) -> str:
    mode = str(td.get("artifact_mode") or "").strip().lower()
    if mode in (ARTIFACT_MODE_BLIND_SPOT, ARTIFACT_MODE_FALSE_POSITIVE):
        return mode
    src = str(td.get("source") or "").strip().lower()
    if src.startswith("false_positive_"):
        return ARTIFACT_MODE_FALSE_POSITIVE
    fate = str(td.get("fate") or "").strip().lower()
    if fate == "artifact" and not str(td.get("spine_id") or "").strip():
        if str(td.get("removed_spine_id") or "").strip():
            return ARTIFACT_MODE_FALSE_POSITIVE
        return ARTIFACT_MODE_BLIND_SPOT
    return ""


def _is_blind_spot(td: dict) -> bool:
    if _artifact_mode(td) == ARTIFACT_MODE_BLIND_SPOT:
        return True
    if is_local_decision(td) and str(td.get("fate") or "").strip().lower() == "artifact":
        return True
    return False


def _is_false_positive_mark(td: dict) -> bool:
    return _artifact_mode(td) == ARTIFACT_MODE_FALSE_POSITIVE


def _is_false_positive_artifact(td: dict) -> bool:
    if not _is_false_positive_mark(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    src = str(td.get("source") or "").strip().lower()
    return fate == "artifact" or src in ("false_positive_artifact", "false_positive_removed")


def _is_false_positive_ignore(td: dict) -> bool:
    if not _is_false_positive_mark(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    src = str(td.get("source") or "").strip().lower()
    return fate == "ignore" or src == "false_positive_ignore"


def all_other_timepoints_cleared(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    tp: str,
) -> bool:
    """True when every timepoint except tp has no match and no fate."""
    for t in timepoint_names:
        if t == tp:
            continue
        td = per_tp.get(t) or {}
        if str(td.get("spine_id") or "").strip():
            return False
        if str(td.get("fate") or "").strip():
            return False
        if _is_false_positive_mark(td):
            return False
    return True



def _is_observed_match(td: dict) -> bool:
    """Positively matched spine (not inferred / censored)."""
    if _is_blind_spot(td) or _is_false_positive_mark(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    if fate in ("lost", "artifact", "ignore"):
        return False
    sid = str(td.get("spine_id") or "").strip()
    if not sid or sid.startswith("artifact_"):
        return False
    src = str(td.get("source") or "").strip().lower()
    if src in ("false_positive_removed", "false_positive_bridged"):
        return False
    return True


def _lineage_continuous_present(td: dict) -> bool:
    """Present for survival / lifespan / event continuity."""
    return _is_observed_match(td)


def _tp_is_present(td: dict) -> bool:
    """Spine was matched at this timepoint."""
    if _is_blind_spot(td) or _is_false_positive_mark(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    if fate in ("lost", "artifact", "ignore"):
        return False
    src = str(td.get("source") or "").strip().lower()
    if src == "false_positive_removed":
        return False
    sid = str(td.get("spine_id") or "").strip()
    return bool(sid) and not sid.startswith("artifact_")


def _tp_is_unmarked_gap(td: dict) -> bool:
    """No spine_id and no explicit fate (UI: — no match —)."""
    if _is_blind_spot(td):
        return False
    if str(td.get("fate") or "").strip():
        return False
    return not str(td.get("spine_id") or "").strip()


def _tp_is_censored(td: dict) -> bool:
    """Blind-spot artifact. Legacy only: a lineage can no longer be censored."""
    return _is_blind_spot(td)


_FATE_KEYS = ("fate", "artifact_mode", "decision_scope", "continuity", "fate_locked")


def normalize_lineage_input(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Dict[str, dict]:
    """Reduce a lineage to what the user decides: which spine at which timepoint.

    Rules review 2026-10 (docs/RULES_REVIEW_2026-10.md, decision 29): a lineage
    ends only MATCHED or LOST -> ABSENT. ARTIFACT / IGNORE are Timepoint-mode
    tags on one detection, never lineage fates. So every stored fate (NEW, LOST,
    IGNORE, ARTIFACT, blind-spot censoring), Timepoint-mode local decision and
    false-positive mark is dropped here; NEW and LOST are re-inferred after.
    """
    out: Dict[str, dict] = {}
    for tp in timepoint_names:
        td = dict(per_tp.get(tp) or {})
        sid = str(td.get("spine_id") or "").strip()
        if _is_false_positive_mark(td) or sid.startswith("artifact_"):
            sid = ""
        for key in _FATE_KEYS:
            td.pop(key, None)
        td["fate"] = None
        if not sid:
            td["spine_id"] = None
            src = str(td.get("source") or "").strip().lower()
            if src.startswith(_INFERRED_SOURCE_PREFIXES) or src.startswith("false_positive_"):
                td["source"] = "cleared"
        out[tp] = td
    return out


def _present_flags(per_tp: Dict[str, dict], timepoint_names: List[str]) -> List[bool]:
    return [_tp_is_present(per_tp.get(tp) or {}) for tp in timepoint_names]


def validate_lineage_shape(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Optional[str]:
    """Return an error string when the matched timepoints are not one block.

    Run on normalize_lineage_input() output, so the only thing left to check is
    that a lineage is never split: no gap between two matched timepoints.
    (Fates are inferred, so "only ABSENT before NEW / after LOST" holds by
    construction.)
    """
    present = _present_flags(per_tp, timepoint_names)
    if not any(present):
        return None
    first = present.index(True)
    last = len(present) - 1 - present[::-1].index(True)
    for i in range(first, last + 1):
        if not present[i]:
            return (
                f"Lineage cannot be split: matched at {timepoint_names[first]} and "
                f"{timepoint_names[last]} but not at {timepoint_names[i]}. "
                "Matched timepoints must form one continuous block."
            )
    return None


def _release_tp_to_pool(td: dict) -> dict:
    """Drop a matched spine from this lineage row so it re-enters the matching pool."""
    out = dict(td)
    sid = str(out.get("spine_id") or "").strip()
    if sid:
        out["removed_spine_id"] = sid
    out["spine_id"] = None
    out["fate"] = None
    out.pop("continuity", None)
    out["source"] = SOURCE_RETURNED_TO_POOL
    return out


def _keep_one_block(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    anchor_timepoint: str = "",
) -> Dict[str, dict]:
    """Keep one continuous block of matches; release the rest to the pool.

    Saves are validated, so this only acts on lineages that were not: a
    duplicate-conflict resolution that removed a middle spine, or legacy data.
    The block holding the anchor timepoint wins, else the first block.
    """
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    present = _present_flags(out, timepoint_names)
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(present):
        if present[i]:
            j = i
            while j + 1 < len(present) and present[j + 1]:
                j += 1
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(blocks) <= 1:
        return out
    keep = blocks[0]
    if anchor_timepoint in timepoint_names:
        a = timepoint_names.index(anchor_timepoint)
        keep = next((b for b in blocks if b[0] <= a <= b[1]), keep)
    for k, tp in enumerate(timepoint_names):
        if present[k] and not keep[0] <= k <= keep[1]:
            out[tp] = _release_tp_to_pool(out[tp])
    return out


def _infer_new_and_lost(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Tuple[Dict[str, dict], str, str]:
    """NEW at the first match (unless it is the first session timepoint); LOST at
    the first timepoint after the last match (unless that was the last one).
    Everything else outside the block stays empty = ABSENT.

    Returns (per_tp, last_seen_tp, disappeared_at_tp).
    """
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    present = _present_flags(out, timepoint_names)
    if not any(present):
        return out, "", ""
    first = present.index(True)
    last = len(present) - 1 - present[::-1].index(True)
    if first > 0:
        td = out[timepoint_names[first]]
        td["fate"] = "new"
        td["source"] = str(td.get("source") or "inferred_new")
    disappeared_at = ""
    if last + 1 < len(timepoint_names):
        tp = timepoint_names[last + 1]
        td = out[tp]
        td["fate"] = "lost"
        td["spine_id"] = None
        td["source"] = f"{SOURCE_LOST_INFERRED_PREFIX}{timepoint_names[last]}"
        disappeared_at = tp
    return out, timepoint_names[last], disappeared_at


def _finalize_lineage_per_tp(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    pre_spine_id: str,
    anchor_timepoint: str = "",
    pre_tp: str = "",
    oof_segments: Optional[List[dict]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
    shifts: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> tuple[Dict[str, dict], str, str, str, bool]:
    """normalize -> keep one block -> infer NEW / LOST.

    Returns (per_tp, last_seen_tp, disappeared_at_tp, censored_from_tp,
    right_censored); the last two are always empty / False now (no lineage
    censoring). oof_segments / ignored_by_tp / shifts / pre_spine_id are
    accepted for call-site compatibility and not used.
    """
    normalized = normalize_lineage_input(per_tp, timepoint_names)
    one_block = _keep_one_block(
        normalized, timepoint_names, anchor_timepoint=anchor_timepoint or pre_tp
    )
    finalized, last_seen_tp, disappeared_at_tp = _infer_new_and_lost(one_block, timepoint_names)
    return finalized, last_seen_tp, disappeared_at_tp, "", False


def _meta_dir(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov)


def paths(respan: Path, fov: int) -> Dict[str, Path]:
    base = _meta_dir(respan, fov)
    return {
        "dir": base,
        "decisions": base / "lineage_decisions.json",
        "progress": base / "spine_review_progress.json",
        "registry": base / "spine_registry_wide.csv",
    }


def load_decisions(respan: Path, fov: int) -> dict:
    p = paths(respan, fov)["decisions"]
    if not p.is_file():
        return {"animal_id": "", "fov": str(fov), "lineages": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    data["lineages"] = [_apply_current_rules(lin) for lin in data.get("lineages") or []]
    return data


def _apply_current_rules(lin: dict) -> dict:
    """Re-derive a stored lineage under the current rules (in memory).

    Legacy lineages may carry fates the rules no longer allow (IGNORE / ARTIFACT /
    censored at lineage level, LOST on every trailing timepoint, a split block).
    Every reader sees the same normalized lineage; the file is rewritten in this
    shape on the next save.
    """
    per_tp = dict(lin.get("per_tp") or {})
    tps = list(lin.get("timepoint_names") or per_tp.keys())
    if not tps:
        return lin
    finalized, last_seen, _, _, _ = _finalize_lineage_per_tp(
        per_tp,
        tps,
        pre_spine_id=str(lin.get("pre_spine_id") or ""),
        anchor_timepoint=str(lin.get("pre_timepoint") or ""),
    )
    first_seen, _, _, _ = _derive_lineage_summary(finalized, tps)
    out = dict(lin)
    out["per_tp"] = {tp: dict(finalized.get(tp) or {}) for tp in tps}
    out["first_seen_tp"] = first_seen
    out["last_seen_tp"] = last_seen
    out["censored_from_tp"] = ""
    out["right_censored"] = False
    out.pop("ignore_after_tp", None)
    return out


def get_lineage_by_key(respan: Path, fov: int, lineage_key: str) -> Optional[dict]:
    key = str(lineage_key or "").strip()
    if not key:
        return None
    for lin in load_decisions(respan, fov).get("lineages") or []:
        row_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if row_key == key:
            return lin
    return None


def status_for_tp_data(tp_data: dict) -> str:
    return _status_for_tp(tp_data)


def finalize_lineage_per_tp(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    pre_spine_id: str,
    anchor_timepoint: str = "",
    pre_tp: str = "",
    oof_segments: Optional[List[dict]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
    shifts: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> tuple[Dict[str, dict], str, str, str, bool]:
    return _finalize_lineage_per_tp(
        per_tp,
        timepoint_names,
        pre_spine_id=pre_spine_id,
        anchor_timepoint=anchor_timepoint,
        pre_tp=pre_tp,
        oof_segments=oof_segments,
        ignored_by_tp=ignored_by_tp,
        shifts=shifts,
    )


def save_decision(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    pre_spine_id: str,
    pre_timepoint: str,
    timepoint_names: List[str],
    per_tp: Dict[str, dict],
    lineage_key: str = "",
    oof_segments: Optional[List[dict]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
    shifts: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> dict:
    meta = paths(respan, fov)
    meta["dir"].mkdir(parents=True, exist_ok=True)
    data = load_decisions(respan, fov)
    data["animal_id"] = animal_id
    data["fov"] = str(fov)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    lineages: List[dict] = list(data.get("lineages") or [])
    key = str(lineage_key or pre_spine_id).strip()
    local_anchor = str(pre_spine_id).strip()
    # Validate what the user submitted, before _keep_one_block() can silently
    # drop a match after a gap.
    shape_error = validate_lineage_shape(
        normalize_lineage_input(per_tp, timepoint_names), timepoint_names
    )
    if shape_error:
        raise ValueError(shape_error)
    finalized, last_seen_tp, disappeared_at_tp, censored_from_tp, right_censored = (
        _finalize_lineage_per_tp(
        per_tp,
        timepoint_names,
        pre_spine_id=local_anchor,
        anchor_timepoint=pre_timepoint,
        pre_tp=pre_timepoint,
        oof_segments=oof_segments,
        ignored_by_tp=ignored_by_tp,
        shifts=shifts,
    )
    )

    first_seen_tp, _, _, _ = _derive_lineage_summary(finalized, timepoint_names)

    # Check if lineage is empty (all TPs clear + no faith)
    has_any_match = any(
        str(finalized.get(tp, {}).get("spine_id") or "").strip()
        for tp in timepoint_names
    )

    deleted = False
    if not has_any_match:
        # Empty lineage: delete it and return spines to pool
        for i, row in enumerate(lineages):
            row_key = str(row.get("lineage_key") or row.get("pre_spine_id") or "").strip()
            if row_key == key:
                lineages.pop(i)
                deleted = True
                break
        data["lineages"] = lineages
        meta["decisions"].write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {
            "path": str(meta["decisions"]),
            "registry_path": "",
            "disposition_path": "",
            "coverage": {},
            "first_seen_tp": "",
            "last_seen_tp": "",
            "lost_inferred": False,
            "censored_from_tp": "",
            "right_censored": False,
            "deleted": True,
        }

    entry = {
        "lineage_key": key,
        "pre_spine_id": local_anchor,
        "pre_timepoint": pre_timepoint,
        "timepoint_names": list(timepoint_names),
        "first_seen_tp": first_seen_tp,
        "last_seen_tp": last_seen_tp,
        "censored_from_tp": censored_from_tp,
        "right_censored": right_censored,
        "per_tp": {tp: dict(finalized.get(tp) or {}) for tp in timepoint_names},
    }
    replaced = False
    for i, row in enumerate(lineages):
        row_key = str(row.get("lineage_key") or row.get("pre_spine_id") or "").strip()
        if row_key == key:
            lineages[i] = entry
            replaced = True
            break
    if not replaced:
        lineages.append(entry)
    data["lineages"] = lineages
    meta["decisions"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    registry_path = rebuild_registry_wide(respan, fov, animal_id=animal_id)
    disposition_path = ""
    coverage: dict = {}
    try:
        from . import spine_qc_store

        disp, coverage = spine_qc_store.rebuild_spine_disposition(
            respan,
            fov,
            animal_id=animal_id,
            timepoint_names=timepoint_names,
            ignored_by_tp=ignored_by_tp,
        )
        disposition_path = str(disp)
    except Exception:
        pass
    return {
        "path": str(meta["decisions"]),
        "registry_path": str(registry_path),
        "disposition_path": disposition_path,
        "coverage": coverage,
        "first_seen_tp": first_seen_tp,
        "last_seen_tp": last_seen_tp,
        "lost_inferred": bool(disappeared_at_tp),
        "censored_from_tp": censored_from_tp,
        "right_censored": right_censored,
        "deleted": False,
    }


def apply_undo_snapshot(
    respan: Path,
    fov: int,
    stash: dict,
    *,
    timepoint_names: Optional[List[str]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
) -> dict:
    """Reverse the most recent save_decision() call using a pre-save stash
    captured by the caller right before that call (see
    mtp_spine_viewer.confirm_lineage). Restores the lineage's exact prior
    state (or removes it if it was newly created), and rolls reviewed_ids /
    phase_index back to their pre-save values. Single-use: the caller is
    responsible for clearing its stash after calling this.
    """
    key = str(stash.get("lineage_key") or "").strip()
    if not key:
        return {"ok": False, "message": "Nothing to undo."}

    meta = paths(respan, fov)
    data = load_decisions(respan, fov)
    lineages: List[dict] = [
        row
        for row in (data.get("lineages") or [])
        if str(row.get("lineage_key") or row.get("pre_spine_id") or "").strip() != key
    ]
    prior_entry = stash.get("prior_entry")
    if prior_entry:
        lineages.append(prior_entry)
    data["lineages"] = lineages
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta["dir"].mkdir(parents=True, exist_ok=True)
    meta["decisions"].write_text(json.dumps(data, indent=2), encoding="utf-8")

    animal_id = str(data.get("animal_id") or "")
    registry_path = rebuild_registry_wide(respan, fov, animal_id=animal_id)
    disposition_path = ""
    coverage: dict = {}
    try:
        from . import spine_qc_store

        disp, coverage = spine_qc_store.rebuild_spine_disposition(
            respan,
            fov,
            animal_id=animal_id,
            timepoint_names=timepoint_names,
            ignored_by_tp=ignored_by_tp,
        )
        disposition_path = str(disp)
    except Exception:
        pass

    reviewed_id = str(stash.get("reviewed_id") or "")
    prog = load_progress(respan, fov)
    ids = list(prog.get("reviewed_ids") or [])
    if not stash.get("was_already_reviewed") and reviewed_id in ids:
        ids.remove(reviewed_id)
    phase_index = int(stash.get("phase_index", prog.get("phase_index", 0)) or 0)
    anchor_timepoint = str(stash.get("anchor_timepoint") or prog.get("anchor_timepoint") or "")
    save_progress(
        respan,
        fov,
        index=int(prog.get("last_pre_spine_index", 0) or 0),
        reviewed_ids=ids,
        phase_index=phase_index,
        anchor_timepoint=anchor_timepoint,
    )

    return {
        "ok": True,
        "lineage_key": key,
        "spine_id": reviewed_id,
        "phase_index": phase_index,
        "anchor_timepoint": anchor_timepoint,
        "registry_path": str(registry_path),
        "disposition_path": disposition_path,
        "coverage": coverage,
        "restored": bool(prior_entry),
        "message": (
            f"Restored prior state of {key}."
            if prior_entry
            else f"Removed {key} (was a new lineage) — back to unreviewed."
        ),
    }


def delete_lineages_claiming(
    respan: Path,
    fov: int,
    timepoint: str,
    spine_ids: set,
) -> List[dict]:
    """Delete every saved lineage that actively claims one of spine_ids at
    timepoint (a matched spine_id there, or its anchor spine when timepoint is
    the lineage's own pre_timepoint). The lineage's other timepoints are NOT
    kept: they return to the unmatched pool. Reviewed marks for the deleted
    lineages are dropped too, so their anchors re-enter the queue.

    Returns the deleted lineage entries (for undo); empty if none matched.
    """
    wanted = {str(x).strip() for x in spine_ids if str(x).strip()}
    if not wanted:
        return []
    data = load_decisions(respan, fov)
    kept: List[dict] = []
    removed: List[dict] = []
    for lin in data.get("lineages") or []:
        if _lineage_claim_ids_at_tp(lin, timepoint) & wanted:
            removed.append(lin)
        else:
            kept.append(lin)
    if not removed:
        return []
    _write_lineages_and_rebuild(respan, fov, data, kept)
    gone_keys = {
        str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        for lin in removed
    }
    prog = load_progress(respan, fov)
    ids = [x for x in (prog.get("reviewed_ids") or []) if str(x) not in gone_keys]
    save_progress(
        respan,
        fov,
        index=int(prog.get("last_pre_spine_index", 0) or 0),
        reviewed_ids=ids,
        phase_index=int(prog.get("phase_index", 0) or 0),
        anchor_timepoint=str(prog.get("anchor_timepoint") or ""),
    )
    return removed


def restore_lineages(respan: Path, fov: int, entries: List[dict]) -> None:
    """Put back lineage entries previously returned by delete_lineages_claiming."""
    if not entries:
        return
    data = load_decisions(respan, fov)
    restore_keys = {
        str(e.get("lineage_key") or e.get("pre_spine_id") or "").strip() for e in entries
    }
    kept = [
        lin
        for lin in (data.get("lineages") or [])
        if str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        not in restore_keys
    ]
    kept.extend(entries)
    _write_lineages_and_rebuild(respan, fov, data, kept)


def _write_lineages_and_rebuild(respan: Path, fov: int, data: dict, lineages: List[dict]) -> None:
    meta = paths(respan, fov)
    data["lineages"] = lineages
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta["dir"].mkdir(parents=True, exist_ok=True)
    meta["decisions"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    rebuild_registry_wide(respan, fov, animal_id=str(data.get("animal_id") or ""))


def _status_for_tp(tp_data: dict) -> str:
    if is_single_tp_focus_ignore(tp_data):
        return "single_tp_only"
    if _tp_is_censored(tp_data):
        return "censored"
    src = str(tp_data.get("source") or "").strip().lower()
    if src in ("false_positive_artifact", "false_positive_removed"):
        return "artifact"
    if src == "false_positive_ignore":
        return "ignored"
    fate = str(tp_data.get("fate") or "").strip().lower()
    if fate:
        return FATE_TO_STATUS.get(fate, fate)
    sid = str(tp_data.get("spine_id") or "").strip()
    if sid:
        if sid.startswith("manual_") or str(tp_data.get("source") or "") == "manual_added":
            return "manual"
        return "matched"
    return "absent"


def _event_present_state(td: dict) -> bool:
    """Observed match for strict event anchoring."""
    return _is_observed_match(td)


def _event_continuous_state(td: dict) -> bool:
    """Continuous lineage presence for survival / lifespan."""
    return _lineage_continuous_present(td)


def _event_clear_state(td: dict) -> bool:
    """Clear for event inference: unmarked gap only (censored blind spots are not clear)."""
    return _tp_is_unmarked_gap(td)


def _tp_is_absent_tail(td: dict) -> bool:
    """Absent after last continuous observation (clear or inferred lost)."""
    if _tp_is_censored(td) or _lineage_continuous_present(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    if fate == "lost" and is_local_decision(td):
        return False
    return _tp_is_unmarked_gap(td) or fate == "lost"


def _derive_events_and_lifecycle(
    timepoint_names: List[str],
    per_tp: Dict[str, dict],
) -> tuple[Dict[str, str], str, str, str]:
    """Derive formation/loss events and lineage lifecycle from per-timepoint states."""
    events: Dict[str, str] = {tp: "" for tp in timepoint_names}
    if not timepoint_names:
        return events, "", "", "unclassified"

    continuous = [_event_continuous_state(per_tp.get(tp) or {}) for tp in timepoint_names]
    observed = [_event_present_state(per_tp.get(tp) or {}) for tp in timepoint_names]
    clear = [_event_clear_state(per_tp.get(tp) or {}) for tp in timepoint_names]
    censored = [_tp_is_censored(per_tp.get(tp) or {}) for tp in timepoint_names]

    formation_idx: Optional[int] = None
    for i in range(1, len(timepoint_names)):
        if observed[i] and not continuous[i - 1] and clear[i - 1]:
            formation_idx = i
            break
    formation_tp = timepoint_names[formation_idx] if formation_idx is not None else ""
    if formation_tp:
        prior = events.get(formation_tp, "")
        events[formation_tp] = (
            f"{prior};appeared_at_{formation_tp}" if prior else f"appeared_at_{formation_tp}"
        )

    termination_idx: Optional[int] = None
    for y in range(len(timepoint_names) - 1):
        if not continuous[y]:
            continue
        if all(
            _tp_is_absent_tail(per_tp.get(timepoint_names[k]) or {})
            for k in range(y + 1, len(timepoint_names))
        ):
            if any(censored[k] for k in range(y + 1, len(timepoint_names))):
                continue
            termination_idx = y
    # The loss event sits on the LOST timepoint: the first one after the last match.
    termination_tp = timepoint_names[termination_idx + 1] if termination_idx is not None else ""
    if termination_tp:
        current = events.get(termination_tp, "")
        events[termination_tp] = (
            f"{current};lost_at_{termination_tp}" if current else f"lost_at_{termination_tp}"
        )

    if all(continuous):
        lifecycle = "stable"
    elif any(censored) and termination_idx is None:
        lifecycle = "right_censored" if censored[-1] else "censored_interval"
    else:
        is_transient = (
            formation_idx is not None
            and termination_idx is not None
            and formation_idx <= termination_idx
            and (termination_idx - formation_idx + 1) < len(timepoint_names)
        )
        if is_transient:
            lifecycle = "transient"
        elif formation_idx is not None or termination_idx is not None:
            lifecycle = "persistent_engram"
        else:
            lifecycle = "unclassified"
    return events, formation_tp, termination_tp, lifecycle


def _fate_for_tp(tp_data: dict) -> str:
    src = str(tp_data.get("source") or "").strip().lower()
    if src == "false_positive_artifact":
        return "false_positive_artifact"
    if src == "false_positive_ignore":
        return "false_positive_ignore"
    if src in ("false_positive_removed",):
        return "false_positive_artifact"
    fate = str(tp_data.get("fate") or "").strip().lower()
    if fate:
        return fate
    if src == "blind_spot_artifact":
        return "artifact"
    return ""


def _ordered_union_timepoints(lineages: List[dict]) -> List[str]:
    from . import animal_config

    cfg = animal_config.load_config()
    order = list(cfg.timepoint_order)
    seen: set[str] = set()
    union: List[str] = []
    for lin in lineages:
        for tp in lin.get("timepoint_names") or []:
            tp = str(tp).strip()
            if tp and tp not in seen:
                seen.add(tp)
                union.append(tp)
        for tp in (lin.get("per_tp") or {}):
            tp = str(tp).strip()
            if tp and tp not in seen:
                seen.add(tp)
                union.append(tp)
    if order:
        return [t for t in order if t in seen] + [t for t in union if t not in order]
    return union


def _derive_lineage_summary(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> tuple[str, str, str, bool]:
    """Derive first_seen_tp, last_seen_tp from per_tp.

    censored_from_tp / right_censored are always empty / False: a lineage can no
    longer be censored (rules review decision 29).
    """
    present = _present_flags(per_tp, timepoint_names)
    if not any(present):
        return "", "", "", False
    first = present.index(True)
    last = len(present) - 1 - present[::-1].index(True)
    return timepoint_names[first], timepoint_names[last], "", False


def _lineage_summary_fields(lineage: dict, per_tp: Dict[str, dict], timepoint_names: List[str]) -> tuple[str, str, str, bool]:
    """Prefer recomputed summary; fall back to stored fields for legacy JSON."""
    first_seen, last_seen, censored_from, right_censored = _derive_lineage_summary(
        per_tp, timepoint_names
    )
    if not first_seen:
        first_seen = str(lineage.get("first_seen_tp", "") or "")
    if not last_seen:
        last_seen = str(lineage.get("last_seen_tp", "") or "")
    if not censored_from:
        censored_from = str(lineage.get("censored_from_tp", "") or "")
        if not censored_from and lineage.get("ignore_after_tp"):
            anchor = str(lineage.get("ignore_after_tp") or "")
            if anchor in timepoint_names:
                anchor_i = timepoint_names.index(anchor)
                if anchor_i + 1 < len(timepoint_names):
                    censored_from = timepoint_names[anchor_i + 1]
    if not right_censored:
        right_censored = bool(lineage.get("right_censored"))
    return first_seen, last_seen, censored_from, right_censored


def _registry_header(timepoint_names: List[str]) -> List[str]:
    # Survival-analysis / interpretation fields (event_<tp>, formation_tp,
    # lifecycle, right_censored, censored_from_tp, n_timepoints_seen,
    # n_timepoints_continuous, active_timepoints) are derived classifications,
    # not observations -- reconstruct them with postprocess_registry.py instead
    # of reading them from this file. first_seen_tp/last_seen_tp stay: they're
    # cheap summaries used elsewhere (not raw per-TP observations, but not an
    # interpretation of what happened either).
    header = [
        "animal_id",
        "fov",
        "lineage_id",
        "lineage_key",
        "pre_spine_id",
        "anchor_timepoint",
        "first_seen_tp",
        "last_seen_tp",
    ]
    header += [f"id_{tp}" for tp in timepoint_names]
    header += [f"local_id_{tp}" for tp in timepoint_names]
    header += [f"status_{tp}" for tp in timepoint_names]
    header += [f"fate_{tp}" for tp in timepoint_names]
    header += [f"artifact_mode_{tp}" for tp in timepoint_names]
    header += [f"{tp}_x" for tp in timepoint_names]
    header += [f"{tp}_y" for tp in timepoint_names]
    header += [f"{tp}_z" for tp in timepoint_names]
    return header


def _build_registry_row(
    animal_id: str,
    fov: int,
    lineage: dict,
    timepoint_names: List[str],
) -> Dict[str, str]:
    pre_id = str(lineage.get("pre_spine_id", "") or "")
    lineage_key = str(lineage.get("lineage_key") or pre_id or "")
    per_tp = dict(lineage.get("per_tp") or {})
    row_tps = list(lineage.get("timepoint_names") or timepoint_names)
    first_seen_tp, last_seen_tp, _censored_from_tp, _right_censored = _lineage_summary_fields(
        lineage, per_tp, row_tps
    )
    row: Dict[str, str] = {
        "animal_id": animal_id,
        "fov": str(fov),
        "lineage_id": f"L_{lineage_key.replace(' ', '_')}",
        "lineage_key": lineage_key,
        "pre_spine_id": pre_id,
        "anchor_timepoint": str(lineage.get("pre_timepoint", "") or ""),
        "first_seen_tp": first_seen_tp,
        "last_seen_tp": last_seen_tp,
    }
    for tp in timepoint_names:
        td = per_tp.get(tp) or {}
        sid = str(td.get("spine_id") or "").strip()
        local_sid = str(td.get("local_spine_id") or "").strip()
        status = _status_for_tp(td) if tp in per_tp else ""
        row[f"id_{tp}"] = sid if tp in per_tp else ""
        row[f"local_id_{tp}"] = local_sid if tp in per_tp else ""
        row[f"status_{tp}"] = status
        row[f"fate_{tp}"] = _fate_for_tp(td) if tp in per_tp else ""
        row[f"artifact_mode_{tp}"] = _artifact_mode(td) if tp in per_tp else ""
        row[f"{tp}_x"] = str(td.get("x", "")) if tp in per_tp and td.get("x") is not None else ""
        row[f"{tp}_y"] = str(td.get("y", "")) if tp in per_tp and td.get("y") is not None else ""
        row[f"{tp}_z"] = str(td.get("z", "")) if tp in per_tp and td.get("z") is not None else ""
    return row


def rebuild_registry_wide(
    respan: Path,
    fov: int,
    *,
    animal_id: str = "",
) -> Path:
    """Rewrite spine_registry_wide.csv from all saved lineage decisions."""
    meta = paths(respan, fov)
    data = load_decisions(respan, fov)
    aid = str(animal_id or data.get("animal_id") or "")
    lineages: List[dict] = list(data.get("lineages") or [])
    timepoint_names = _ordered_union_timepoints(lineages)
    header = _registry_header(timepoint_names)
    rows: List[dict] = []
    for lin in lineages:
        rows.append(_build_registry_row(aid, fov, lin, timepoint_names))
    meta["dir"].mkdir(parents=True, exist_ok=True)
    with meta["registry"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return meta["registry"]


def load_progress(respan: Path, fov: int) -> dict:
    p = paths(respan, fov)["progress"]
    if not p.is_file():
        return {
            "phase_index": 0,
            "anchor_timepoint": "",
            "last_pre_spine_index": 0,
            "reviewed_ids": [],
        }
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "phase_index": int(data.get("phase_index", 0) or 0),
        "anchor_timepoint": str(data.get("anchor_timepoint", "") or ""),
        "last_pre_spine_index": int(data.get("last_pre_spine_index", 0) or 0),
        "reviewed_ids": [str(x) for x in data.get("reviewed_ids", []) if str(x).strip()],
        "updated_at": str(data.get("updated_at", "") or ""),
    }


def save_progress(
    respan: Path,
    fov: int,
    *,
    index: int,
    reviewed_ids: List[str],
    phase_index: int = 0,
    anchor_timepoint: str = "",
) -> Path:
    meta = paths(respan, fov)
    meta["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "phase_index": int(phase_index),
        "anchor_timepoint": str(anchor_timepoint or ""),
        "last_pre_spine_index": int(index),
        "reviewed_ids": list(reviewed_ids),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta["progress"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta["progress"]


def mark_reviewed(
    respan: Path,
    fov: int,
    spine_id: str,
    index: int,
    *,
    phase_index: int = 0,
    anchor_timepoint: str = "",
) -> dict:
    prog = load_progress(respan, fov)
    ids = list(prog.get("reviewed_ids") or [])
    pid = str(spine_id).strip()
    if pid and pid not in ids:
        ids.append(pid)
    prog["reviewed_ids"] = ids
    prog["last_pre_spine_index"] = int(index)
    prog["phase_index"] = int(phase_index)
    prog["anchor_timepoint"] = str(anchor_timepoint or "")
    save_progress(
        respan,
        fov,
        index=int(index),
        reviewed_ids=ids,
        phase_index=int(phase_index),
        anchor_timepoint=str(anchor_timepoint or ""),
    )
    return prog


def _lineage_claim_ids_at_tp(lin: dict, timepoint: str) -> set[str]:
    """Global/local ids at a timepoint that should leave the orphan pool.

    Only actively matched spines (spine_id) are claimed. Released spines (removed_spine_id)
    return to the orphan pool for re-matching elsewhere.
    """
    out: set[str] = set()
    tp = str(timepoint)
    td = dict((lin.get("per_tp") or {}).get(tp) or {})
    # Only count spine_id (active match), not removed_spine_id (released)
    sid = str(td.get("spine_id") or "").strip()
    if sid:
        out.add(sid)
    if str(lin.get("pre_timepoint") or "") == tp:
        for sid in (
            str(lin.get("lineage_key") or "").strip(),
            str(lin.get("pre_spine_id") or "").strip(),
        ):
            if sid:
                out.add(sid)
    return out


def claimed_spine_ids_at_tp(
    respan: Path,
    fov: int,
    timepoint: str,
    *,
    exclude_lineage_key: str = "",
    pending_lineage: Optional[Dict[str, dict]] = None,
) -> set[str]:
    """Spine IDs already assigned at this timepoint in saved lineages.

    When editing one lineage, pass exclude_lineage_key + pending_lineage so cleared
    matches return to the pool before save.
    """
    claimed: set[str] = set()
    exclude = str(exclude_lineage_key or "").strip()
    for lin in load_decisions(respan, fov).get("lineages") or []:
        row_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        if exclude and row_key == exclude:
            if pending_lineage is not None:
                td = dict(pending_lineage.get(timepoint) or {})
                for sid in (
                    str(td.get("spine_id") or "").strip(),
                    str(td.get("removed_spine_id") or "").strip(),
                ):
                    if sid:
                        claimed.add(sid)
            continue
        claimed.update(_lineage_claim_ids_at_tp(lin, timepoint))
    return claimed
