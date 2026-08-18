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


def decision_scope(td: dict) -> str:
    scope = str(td.get("decision_scope") or DECISION_SCOPE_LINEAGE).strip().lower()
    return scope if scope in (DECISION_SCOPE_LINEAGE, DECISION_SCOPE_LOCAL) else DECISION_SCOPE_LINEAGE


def is_local_decision(td: dict) -> bool:
    return decision_scope(td) == DECISION_SCOPE_LOCAL


def validate_contiguity(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Optional[str]:
    """Validate no gaps in matched timepoints. Return error string if invalid, None if OK.

    A valid lineage has matched TPs forming one continuous block: M...M with no C gaps.
    Patterns like MCMMM (matched, gap, matched) are forbidden.
    """
    if not timepoint_names:
        return None

    # Find first and last matched TP
    first_match_idx = None
    last_match_idx = None
    matched_indices = []

    for i, tp in enumerate(timepoint_names):
        td = per_tp.get(tp) or {}
        sid = str(td.get("spine_id") or "").strip()
        if sid:
            if first_match_idx is None:
                first_match_idx = i
            last_match_idx = i
            matched_indices.append(i)

    # No matches or single match is always OK
    if not matched_indices or len(matched_indices) == 1:
        return None

    # Check for gap: if we have matches but some indices between first and last are missing
    for i in range(first_match_idx, last_match_idx + 1):
        if i not in matched_indices:
            # Found a gap
            gap_tp = timepoint_names[i]
            return f"Gap in lineage: spine matched at {timepoint_names[first_match_idx]} and {timepoint_names[last_match_idx]}, but missing at {gap_tp}. Matched timepoints must form one continuous block."

    return None


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
    if fate in ("lost", "artifact", "ignore", "bridge"):
        return False
    sid = str(td.get("spine_id") or "").strip()
    if not sid or sid.startswith("bridge_") or sid.startswith("artifact_"):
        return False
    src = str(td.get("source") or "").strip().lower()
    if src in ("false_positive_removed", "bridge_assumed", "false_positive_bridged"):
        return False
    return True


def strip_bridge_fates(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Dict[str, dict]:
    """Remove legacy bridge labels and synthetic bridge_* IDs."""
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    for tp in timepoint_names:
        td = out[tp]
        fate = str(td.get("fate") or "").strip().lower()
        sid = str(td.get("spine_id") or "").strip()
        src = str(td.get("source") or "").strip().lower()
        if fate != "bridge" and not sid.startswith("bridge_") and not src.startswith("bridge_"):
            continue
        td["fate"] = None
        td["spine_id"] = None
        td.pop("continuity", None)
        for key in ("x", "y", "z"):
            td.pop(key, None)
        td["source"] = "cleared"
    return out


def _lineage_continuous_present(td: dict) -> bool:
    """Present for survival / lifespan / event continuity."""
    return _is_observed_match(td)


def remove_false_positive_observations(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    pre_spine_id: str = "",
) -> Dict[str, dict]:
    """Drop a lone false-positive match."""
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    for i, tp in enumerate(timepoint_names):
        td = out[tp]
        if not _is_false_positive_mark(td):
            continue
        if not all_other_timepoints_cleared(out, timepoint_names, tp):
            raise ValueError(
                f"False-positive tag at '{tp}' requires all other timepoints to be clear "
                f"(no match, no fate)."
            )
        removed = str(td.get("removed_spine_id") or td.get("spine_id") or "").strip()
        td["removed_spine_id"] = removed
        td["spine_id"] = None
        td["artifact_mode"] = ARTIFACT_MODE_FALSE_POSITIVE
        if _is_false_positive_ignore(td):
            td["fate"] = "ignore"
            td["source"] = "false_positive_ignore"
        else:
            td["fate"] = "artifact"
            td["source"] = "false_positive_artifact"
        td["continuity"] = "broken"
    return out


def _prior_blocks_new_inference(prev: dict) -> bool:
    """Earlier TP blocks inferred NEW unless the label was local-only."""
    if _lineage_continuous_present(prev):
        return True
    fate = str(prev.get("fate") or "").strip()
    if not fate:
        return False
    return not is_local_decision(prev)


def _all_prior_no_match_no_fate(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    idx: int,
) -> bool:
    """True when every earlier TP has no continuity (NEW eligibility)."""
    for j in range(idx):
        prev = per_tp.get(timepoint_names[j]) or {}
        if _prior_blocks_new_inference(prev):
            return False
    return True


def infer_new_at_first_match(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    anchor_timepoint: str = "",
) -> Dict[str, dict]:
    """NEW = first match when all earlier TPs are no match, no fate (never at anchor TP)."""
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    anchor = str(anchor_timepoint or "").strip()
    for i, tp in enumerate(timepoint_names):
        if anchor and tp == anchor:
            continue
        td = out[tp]
        if str(td.get("fate") or "").strip():
            continue
        if not str(td.get("spine_id") or "").strip():
            continue
        if _all_prior_no_match_no_fate(out, timepoint_names, i):
            td["fate"] = "new"
            td["source"] = str(td.get("source") or "inferred_new")
    return out


def _tp_is_present(td: dict) -> bool:
    """Spine was matched at this timepoint."""
    if _is_blind_spot(td) or _is_false_positive_mark(td):
        return False
    fate = str(td.get("fate") or "").strip().lower()
    if fate in ("lost", "artifact", "ignore", "bridge"):
        return False
    src = str(td.get("source") or "").strip().lower()
    if src in ("false_positive_removed", "bridge_unfulfilled", "bridge_assumed"):
        return False
    sid = str(td.get("spine_id") or "").strip()
    if sid.startswith("bridge_"):
        return False
    return bool(sid)


def _tp_is_unmarked_gap(td: dict) -> bool:
    """No spine_id and no explicit fate (UI: — no match —)."""
    if _is_blind_spot(td):
        return False
    if str(td.get("fate") or "").strip():
        return False
    return not str(td.get("spine_id") or "").strip()


def _tp_is_censored(td: dict) -> bool:
    """Blind-spot artifact: image quality too poor to observe presence at this TP."""
    return _is_blind_spot(td)


def _has_present_match_after(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    idx: int,
) -> bool:
    """Later TP with real match (lineage recovery after artifact)."""
    for j in range(idx + 1, len(timepoint_names)):
        if _tp_is_present(per_tp.get(timepoint_names[j]) or {}):
            return True
    return False


def _mark_lost_from_index(
    out: Dict[str, dict],
    timepoint_names: List[str],
    start_i: int,
    disappeared_after: str,
) -> None:
    for j in range(start_i, len(timepoint_names)):
        tj = timepoint_names[j]
        tdj = out[tj]
        if _tp_is_censored(tdj):
            return
        if _tp_is_present(tdj):
            return
        if _tp_is_unmarked_gap(tdj):
            tdj["fate"] = "lost"
            tdj["spine_id"] = None
            tag = "lost_inferred_after_" if j == start_i else "lost_propagated_after_"
            tdj["source"] = f"{tag}{disappeared_after}"


def _mark_ignore_from_index(
    out: Dict[str, dict],
    timepoint_names: List[str],
    start_i: int,
    source_after: str,
) -> None:
    for j in range(start_i, len(timepoint_names)):
        tj = timepoint_names[j]
        tdj = out[tj]
        if _tp_is_censored(tdj):
            return
        if _tp_is_present(tdj):
            return
        if _tp_is_unmarked_gap(tdj):
            tdj["fate"] = "ignore"
            tdj["spine_id"] = None
            tag = "ignore_inferred_after_" if j == start_i else "ignore_propagated_after_"
            tdj["source"] = f"{tag}{source_after}"


def _tail_has_censored_or_present(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    start_i: int,
) -> tuple[bool, bool]:
    """From start_i onward: any censored TP, and any present match."""
    censored = False
    present = False
    for j in range(start_i, len(timepoint_names)):
        td = per_tp.get(timepoint_names[j]) or {}
        if _tp_is_censored(td):
            censored = True
        if _tp_is_present(td):
            present = True
    return censored, present


def _shift_stack_coords(
    x: float,
    y: float,
    src_tp: str,
    dst_tp: str,
    *,
    pre_tp: str,
    shifts: Dict[str, Tuple[float, float, float]],
) -> Tuple[float, float]:
    sx_s, sy_s, _ = shifts.get(src_tp, (0.0, 0.0, 0.0))
    sx_d, sy_d, _ = shifts.get(dst_tp, (0.0, 0.0, 0.0))
    if src_tp != pre_tp:
        x, y = x - sx_s, y - sy_s
    if dst_tp != pre_tp:
        x, y = x + sx_d, y + sy_d
    return x, y


def _gap_should_infer_ignore(
    tp: str,
    last_seen_td: dict,
    *,
    last_seen_tp: str,
    pre_tp: str,
    oof_segments: Optional[List[dict]],
    ignored_by_tp: Optional[Dict[str, set]],
    shifts: Optional[Dict[str, Tuple[float, float, float]]],
) -> bool:
    """Empty gap after a match → IGNORE when track falls in OOF / ignored region."""
    if str(last_seen_td.get("fate") or "").strip().lower() == "ignore":
        return True
    lx = last_seen_td.get("x")
    ly = last_seen_td.get("y")
    if lx is not None and ly is not None and oof_segments:
        from . import oof_segment_store

        x, y = _shift_stack_coords(
            float(lx),
            float(ly),
            last_seen_tp,
            tp,
            pre_tp=pre_tp,
            shifts=shifts or {},
        )
        if oof_segment_store.coords_in_oof(x, y, tp, oof_segments):
            return True
    if ignored_by_tp and ignored_by_tp.get(tp):
        if lx is not None and ly is not None and oof_segments:
            return False
        return True
    return False


def infer_lost_after_last_match(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
    *,
    pre_tp: str = "",
    oof_segments: Optional[List[dict]] = None,
    ignored_by_tp: Optional[Dict[str, set]] = None,
    shifts: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> tuple[Dict[str, dict], str, str, str, bool]:
    """Infer LOST or IGNORE when later TPs have no match and no fate.

    IGNORE (not LOST) when the gap falls in an OOF / ignored region (FOV edge).
    Blind-spot ARTIFACT (clear + artifact) is censored and never inferred as LOST.

    Summary fields returned:
    - last_seen_tp: last timepoint with observed presence
    - disappeared_at_tp: last timepoint with observed presence before the loss
      (empty if not lost) — per the LOST convention, a spine is lost at the last
      TP it was seen, not the first TP it went missing.
    - censored_from_tp: first censored / OOF-ignore timepoint (empty if fully tracked)
    """
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}
    last_seen_tp = ""
    disappeared_at_tp = ""
    censored_from_tp = ""
    right_censored = False
    anchor_pre = str(pre_tp or (timepoint_names[0] if timepoint_names else ""))

    for i, tp in enumerate(timepoint_names):
        td = out[tp]
        fate = str(td.get("fate") or "").strip().lower()

        if is_local_decision(td):
            if fate == "artifact" or _is_blind_spot(td):
                if not censored_from_tp:
                    censored_from_tp = tp
                td.setdefault("artifact_mode", ARTIFACT_MODE_BLIND_SPOT)
            elif _tp_is_present(td):
                last_seen_tp = tp
            else:
                # Timepoint-mode local fate (L/I/A/N): this TP only — never infer lineage LOST/IGNORE after it.
                last_seen_tp = ""
            continue

        if fate == "lost" and not _tp_is_present(td):
            if not disappeared_at_tp:
                disappeared_at_tp = last_seen_tp
            _mark_lost_from_index(
                out,
                timepoint_names,
                i + 1,
                last_seen_tp,
            )
            break

        if fate == "ignore" and not _tp_is_present(td) and not _is_false_positive_mark(td):
            if not censored_from_tp:
                censored_from_tp = tp
            _mark_ignore_from_index(
                out,
                timepoint_names,
                i + 1,
                last_seen_tp,
            )
            break

        if _tp_is_censored(td):
            if not censored_from_tp:
                censored_from_tp = tp
                td.setdefault("artifact_mode", ARTIFACT_MODE_BLIND_SPOT)
                td["source"] = str(td.get("source") or "blind_spot_censored")
            tail_censored, tail_present = _tail_has_censored_or_present(out, timepoint_names, i)
            if last_seen_tp and tail_censored and not tail_present:
                right_censored = True
            continue

        if _tp_is_present(td):
            last_seen_tp = tp
            continue

        if not last_seen_tp:
            continue

        if _tp_is_unmarked_gap(td):
            last_i = timepoint_names.index(last_seen_tp)
            has_censored_after = any(
                _tp_is_censored(out.get(timepoint_names[k]) or {})
                for k in range(last_i + 1, len(timepoint_names))
            )
            if has_censored_after:
                present_after_last = any(
                    _tp_is_present(out.get(timepoint_names[k]) or {})
                    for k in range(last_i + 1, len(timepoint_names))
                )
                if not present_after_last:
                    if not censored_from_tp:
                        for k in range(last_i + 1, len(timepoint_names)):
                            if _tp_is_censored(out.get(timepoint_names[k]) or {}):
                                censored_from_tp = timepoint_names[k]
                                break
                    right_censored = all(
                        _tp_is_censored(out.get(timepoint_names[k]) or {})
                        or _tp_is_unmarked_gap(out.get(timepoint_names[k]) or {})
                        for k in range(last_i + 1, len(timepoint_names))
                    ) and _tp_is_censored(out.get(timepoint_names[-1]) or {})
                    break
            last_seen_td = out.get(last_seen_tp) or {}
            if _gap_should_infer_ignore(
                tp,
                last_seen_td,
                last_seen_tp=last_seen_tp,
                pre_tp=anchor_pre,
                oof_segments=oof_segments,
                ignored_by_tp=ignored_by_tp,
                shifts=shifts,
            ):
                if not censored_from_tp:
                    censored_from_tp = tp
                td["fate"] = "ignore"
                td["spine_id"] = None
                td["source"] = f"ignore_inferred_after_{last_seen_tp}"
                _mark_ignore_from_index(out, timepoint_names, i + 1, last_seen_tp)
                break
            # Lineage ends at last_seen_tp (last visible). A LOST timepoint carries no
            # spine_id (see _mark_lost_from_index), so writing the lost fate here does
            # not conflict with gap TPs returning their matches to the pool.
            _mark_lost_from_index(
                out,
                timepoint_names,
                i,
                last_seen_tp,
            )
            disappeared_at_tp = last_seen_tp
            break

    if right_censored:
        disappeared_at_tp = ""

    return out, last_seen_tp, disappeared_at_tp, censored_from_tp, right_censored


def _release_tp_to_pool(td: dict) -> dict:
    """Drop a matched spine from this lineage row so it re-enters the matching pool."""
    out = dict(td)
    sid = str(out.get("spine_id") or "").strip()
    if sid:
        out["removed_spine_id"] = sid
    out["spine_id"] = None
    out["fate"] = None
    out["artifact_mode"] = None
    out.pop("decision_scope", None)
    out.pop("continuity", None)
    out["source"] = SOURCE_RETURNED_TO_POOL
    return out


def _first_new_index(out: Dict[str, dict], timepoint_names: List[str]) -> Optional[int]:
    for i, tp in enumerate(timepoint_names):
        if str((out.get(tp) or {}).get("fate") or "").strip().lower() == "new":
            return i
    return None


def _last_present_index(out: Dict[str, dict], timepoint_names: List[str]) -> int:
    last_i = -1
    for i, tp in enumerate(timepoint_names):
        td = out.get(tp) or {}
        if is_local_decision(td) and str(td.get("fate") or "").strip():
            return i
        if _tp_is_unmarked_gap(td):
            if last_i >= 0:
                break
            continue
        if str(td.get("fate") or "").strip().lower() == "lost":
            break
        if _tp_is_present(td):
            last_i = i
    return last_i


def release_unbound_matches(
    per_tp: Dict[str, dict],
    timepoint_names: List[str],
) -> Dict[str, dict]:
    """Return stale auto-matches to the pool outside this lineage's span.

    - S-mode local fate (I/A/N/L): lineage exists at that TP only.
    - NEW at TPk: earlier TP matches are released.
    - LOST after last seen: later TP matches are released (even if algorithm-filled).
    """
    out = {tp: dict(per_tp.get(tp) or {}) for tp in timepoint_names}

    for i, tp in enumerate(timepoint_names):
        td = out[tp]
        if is_local_decision(td) and str(td.get("fate") or "").strip():
            for j, tp2 in enumerate(timepoint_names):
                if j != i:
                    out[tp2] = _release_tp_to_pool(out[tp2])
            return out

    new_i = _first_new_index(out, timepoint_names)
    if new_i is not None:
        for j in range(new_i):
            out[timepoint_names[j]] = _release_tp_to_pool(out[timepoint_names[j]])
        end_i = new_i
        for j in range(new_i, len(timepoint_names)):
            td = out[timepoint_names[j]]
            if _tp_is_present(td):
                end_i = j
            elif str(td.get("fate") or "").strip().lower() in ("lost", "ignore", "artifact"):
                break
            elif _tp_is_unmarked_gap(td):
                break
        for j in range(end_i + 1, len(timepoint_names)):
            out[timepoint_names[j]] = _release_tp_to_pool(out[timepoint_names[j]])
        return out

    last_present = _last_present_index(out, timepoint_names)
    if last_present >= 0:
        for j in range(last_present + 1, len(timepoint_names)):
            td = out[timepoint_names[j]]
            if str(td.get("spine_id") or "").strip():
                out[timepoint_names[j]] = _release_tp_to_pool(td)

    return out


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
    without_fp = remove_false_positive_observations(
        per_tp, timepoint_names, pre_spine_id=pre_spine_id
    )
    without_bridge = strip_bridge_fates(without_fp, timepoint_names)
    with_new = infer_new_at_first_match(
        without_bridge, timepoint_names, anchor_timepoint=anchor_timepoint
    )
    trimmed = release_unbound_matches(with_new, timepoint_names)
    finalized, _infer_last, disappeared_at_tp, censored_from_tp, right_censored = (
        infer_lost_after_last_match(
            trimmed,
            timepoint_names,
            pre_tp=pre_tp or anchor_timepoint,
            oof_segments=oof_segments,
            ignored_by_tp=ignored_by_tp,
            shifts=shifts,
        )
    )
    released = release_unbound_matches(finalized, timepoint_names)
    first_seen_tp, last_seen_tp, censored_from, rc_summary = _derive_lineage_summary(
        released, timepoint_names
    )
    if rc_summary:
        right_censored = rc_summary
    if censored_from and not censored_from_tp:
        censored_from_tp = censored_from
    return released, last_seen_tp, disappeared_at_tp, censored_from_tp, right_censored


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
    return json.loads(p.read_text(encoding="utf-8"))


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

    # Validate contiguity: no gaps allowed
    contiguity_error = validate_contiguity(finalized, timepoint_names)
    if contiguity_error:
        raise ValueError(contiguity_error)

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
    termination_tp = timepoint_names[termination_idx] if termination_idx is not None else ""
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
    """Derive first_seen_tp, last_seen_tp, censored_from_tp, right_censored from per_tp."""
    first_seen = ""
    last_i = _last_present_index(per_tp, timepoint_names)
    last_seen = timepoint_names[last_i] if last_i >= 0 else ""
    for i in range(last_i + 1):
        tp = timepoint_names[i]
        if _tp_is_present(per_tp.get(tp) or {}):
            if not first_seen:
                first_seen = tp

    disappeared_at = ""
    censored_from = ""
    for tp in timepoint_names:
        td = per_tp.get(tp) or {}
        fate = str(td.get("fate") or "").strip().lower()
        if fate == "lost" and not disappeared_at:
            disappeared_at = tp
        if _tp_is_censored(td) and not censored_from:
            censored_from = tp
        elif fate == "ignore" and not _is_false_positive_mark(td) and not censored_from:
            censored_from = tp

    right_censored = False
    if last_seen and censored_from and not disappeared_at:
        try:
            last_i = timepoint_names.index(last_seen)
            cf_i = timepoint_names.index(censored_from)
        except ValueError:
            last_i = cf_i = -1
        if cf_i > last_i:
            tail = timepoint_names[cf_i:]
            if tail and all(
                _tp_is_censored(per_tp.get(t) or {})
                or _tp_is_unmarked_gap(per_tp.get(t) or {})
                or str((per_tp.get(t) or {}).get("fate") or "").lower() == "ignore"
                for t in tail
            ):
                if _tp_is_censored(per_tp.get(tail[-1]) or {}):
                    right_censored = True

    return first_seen, last_seen, censored_from, right_censored


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


def _append_registry_row(
    registry_path: Path,
    animal_id: str,
    fov: int,
    pre_spine_id: str,
    timepoint_names: List[str],
    per_tp: Dict[str, dict],
    *,
    last_seen_tp: str = "",
    censored_from_tp: str = "",
    right_censored: bool = False,
) -> None:
    """Legacy single-row append — prefer rebuild_registry_wide after each save."""
    lineage = {
        "pre_spine_id": pre_spine_id,
        "pre_timepoint": "",
        "timepoint_names": list(timepoint_names),
        "first_seen_tp": "",
        "last_seen_tp": last_seen_tp,
        "censored_from_tp": censored_from_tp,
        "right_censored": right_censored,
        "per_tp": per_tp,
    }
    row = _build_registry_row(animal_id, fov, lineage, timepoint_names)
    header = _registry_header(timepoint_names)
    existing: List[dict] = []
    lineage_id = f"L_{pre_spine_id}"
    if registry_path.is_file():
        with registry_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if str(r.get("lineage_id", "")) == lineage_id:
                    continue
                existing.append(r)
    existing.append({k: row.get(k, "") for k in header})
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)


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
