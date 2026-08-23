"""Pre→mid algo queue and multi-timepoint chain for the longitudinal spine viewer."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from . import baseline_adapter, dendrite_link_store, oof_segment_store

MAX_Z = baseline_adapter.MAX_MATCH_Z_GAP


def _lookup_rec(lookup: Dict[str, dict], spine_id: str) -> dict:
    return dict(lookup.get(str(spine_id)) or {})


def _local_id_for_lookup(lookup: Dict[str, dict], spine_id: str) -> str:
    """Resolve global S_* (or manual) to local CSV id for dataframe scoring."""
    sid = str(spine_id or "").strip()
    if not sid:
        return sid
    rec = _lookup_rec(lookup, sid)
    if rec:
        return str(rec.get("local_spine_id") or rec.get("spine_id") or sid)
    return sid


def _global_id_for_lookup(tp_lookup: Dict[str, dict], local_or_global: str) -> str:
    sid = str(local_or_global or "").strip()
    if not sid:
        return sid
    if sid in tp_lookup:
        return sid
    for gid, rec in tp_lookup.items():
        if str(rec.get("local_spine_id") or "") == sid:
            return str(gid)
    return sid


def _globalize_algo_hit(tp_lookup: Dict[str, dict], hit: dict) -> dict:
    if not hit:
        return hit
    out = dict(hit)
    sid = str(out.get("spine_id") or "").strip()
    if sid:
        out["spine_id"] = _global_id_for_lookup(tp_lookup, sid)
    return out


def _link_pre_dendrite_ids(cross_links: List[dict], link_id: str, pre_tp: str) -> Set[str]:
    for link in cross_links:
        if str(link.get("link_id", "")) != str(link_id):
            continue
        return {str(x) for x in link.get("members", {}).get(pre_tp, []) if str(x).strip()}
    return set()


def _candidate_rows(
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    *,
    allowed_t1_by_t2: Dict[str, set[str]],
    require_link: bool,
) -> Tuple[List[dict], List[dict]]:
    linked: List[dict] = []
    cross: List[dict] = []
    for _, t2 in t2_df.iterrows():
        t2_id = str(t2["id"])
        t2_d = str(t2["dendrite_id"])
        allowed_t1_d = allowed_t1_by_t2.get(t2_d, set())
        for _, t1 in t1_df.iterrows():
            t1_id = str(t1["id"])
            t1_d = str(t1["dendrite_id"])
            dz = float(abs(float(t1["z"]) - float(t2["z"])))
            if dz > MAX_Z:
                continue
            row = {
                "t1_spine_id": t1_id,
                "t2_spine_id": t2_id,
                "distance_xy": float(
                    np.hypot(float(t1["x"]) - float(t2["x"]), float(t1["y"]) - float(t2["y"]))
                ),
                "distance_z": dz,
            }
            if require_link and not allowed_t1_d:
                continue
            if t1_d in allowed_t1_d:
                linked.append(row)
            else:
                cross.append(row)
    return linked, cross


def _rank_pre_mid(
    rows: List[dict],
    pre_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    *,
    cross_dendrite: bool,
    pre_tiff_path: Optional[str] = None,
    mid_tiff_path: Optional[str] = None,
    exclude_t2_ids: Optional[Set[str]] = None,
) -> List[dict]:
    if not rows:
        return []
    scored = baseline_adapter.score_candidates_hybrid(
        pd.DataFrame(rows), pre_df, mid_df, gating_z=MAX_Z,
        t1_tiff_path=pre_tiff_path,
        t2_tiff_path=mid_tiff_path,
    ).sort_values("final_score", ascending=False)
    best: Dict[str, dict] = {}
    pre_by = pre_df.set_index(pre_df["id"].astype(str))
    mid_by = mid_df.set_index(mid_df["id"].astype(str))
    exclude = exclude_t2_ids or set()
    for _, r in scored.iterrows():
        pre_id = str(r["t1_spine_id"])
        if pre_id in best:
            continue
        mid_id = str(r["t2_spine_id"])
        if mid_id in exclude:
            continue
        if pre_id not in pre_by.index or mid_id not in mid_by.index:
            continue
        pr = pre_by.loc[pre_id]
        mr = mid_by.loc[mid_id]
        best[pre_id] = {
            "pre_spine_id": pre_id,
            "mid_spine_id": mid_id,
            "pre_mid_score": float(r.get("final_score", 0.0)),
            "cross_dendrite": cross_dendrite,
            "pre_dendrite_id": str(pr["dendrite_id"]),
            "mid_dendrite_id": str(mr["dendrite_id"]),
        }
    return sorted(best.values(), key=lambda x: -float(x["pre_mid_score"]))


def build_pre_mid_queues(
    pre_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    cross_links: List[dict],
    *,
    pre_tp: str,
    mid_tp: str,
    link_id: Optional[str] = None,
    tiff_paths: Optional[Dict[str, str]] = None,
    exclude_mid_ids: Optional[Set[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Return (linked_queue, cross_dendrite_queue) sorted by pre→mid score (high first)."""
    pre = pre_df.copy()
    mid = mid_df.copy()
    pre["id"] = pre["id"].astype(str)
    mid["id"] = mid["id"].astype(str)
    pre["dendrite_id"] = pre["dendrite_id"].astype(str)
    mid["dendrite_id"] = mid["dendrite_id"].astype(str)

    if link_id:
        allowed_pre = _link_pre_dendrite_ids(cross_links, link_id, pre_tp)
        if allowed_pre:
            pre = pre[pre["dendrite_id"].isin(allowed_pre)]

    pairwise = dendrite_link_store.to_pairwise_links(cross_links, pre_tp, mid_tp)
    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(pairwise)

    linked_rows, cross_rows = _candidate_rows(
        pre, mid, allowed_t1_by_t2=allowed_t1_by_t2, require_link=bool(pairwise)
    )
    if not linked_rows and pairwise:
        linked_rows, _ = _candidate_rows(
            pre, mid, allowed_t1_by_t2={}, require_link=False
        )
        cross_rows = []

    pre_tiff_path = (tiff_paths or {}).get(pre_tp)
    mid_tiff_path = (tiff_paths or {}).get(mid_tp)
    main_q = _rank_pre_mid(
        linked_rows, pre, mid, cross_dendrite=False,
        pre_tiff_path=pre_tiff_path, mid_tiff_path=mid_tiff_path,
        exclude_t2_ids=exclude_mid_ids,
    )
    cross_q = _rank_pre_mid(
        cross_rows, pre, mid, cross_dendrite=True,
        pre_tiff_path=pre_tiff_path, mid_tiff_path=mid_tiff_path,
        exclude_t2_ids=exclude_mid_ids,
    )
    return main_q, cross_q


def _score_pairwise_best(
    t1_id: str,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    *,
    cross_links: List[dict],
    t1_tp: str,
    t2_tp: str,
    allow_cross: bool,
    t1_tiff_path: Optional[str] = None,
    t2_tiff_path: Optional[str] = None,
    exclude_t2_ids: Optional[Set[str]] = None,
) -> Optional[dict]:
    if t1_id not in set(t1_df["id"].astype(str)):
        return None
    t1_row = t1_df[t1_df["id"].astype(str) == t1_id].iloc[0]
    pairwise = dendrite_link_store.to_pairwise_links(cross_links, t1_tp, t2_tp)
    allowed_t1_by_t2 = baseline_adapter.linked_dendrite_map(pairwise) if pairwise else {}

    exclude = exclude_t2_ids or set()
    rows: List[dict] = []
    t1_d = str(t1_row["dendrite_id"])
    for _, t2 in t2_df.iterrows():
        if str(t2["id"]) in exclude:
            continue
        t2_d = str(t2["dendrite_id"])
        if pairwise and not allow_cross and t1_d not in allowed_t1_by_t2.get(t2_d, set()):
            continue
        dz = float(abs(float(t1_row["z"]) - float(t2["z"])))
        if dz > MAX_Z:
            continue
        rows.append(
            {
                "t1_spine_id": t1_id,
                "t2_spine_id": str(t2["id"]),
                "distance_xy": float(
                    np.hypot(float(t1_row["x"]) - float(t2["x"]), float(t1_row["y"]) - float(t2["y"]))
                ),
                "distance_z": dz,
            }
        )
    if not rows:
        return None
    scored = baseline_adapter.score_candidates_hybrid(
        pd.DataFrame(rows), t1_df, t2_df, gating_z=MAX_Z,
        t1_tiff_path=t1_tiff_path,
        t2_tiff_path=t2_tiff_path,
    ).sort_values("final_score", ascending=False)
    top = scored.iloc[0]
    t2_id = str(top["t2_spine_id"])
    t2_row = t2_df[t2_df["id"].astype(str) == t2_id].iloc[0]
    cross = bool(pairwise) and t1_d not in allowed_t1_by_t2.get(str(t2_row["dendrite_id"]), set())
    return {
        "spine_id": t2_id,
        "x": float(t2_row["x"]),
        "y": float(t2_row["y"]),
        "z": float(t2_row["z"]),
        "dendrite_id": str(t2_row["dendrite_id"]),
        "score": float(top.get("final_score", 0.0)),
        "source": "algo_cross" if cross else "algo_chain",
        "cross_dendrite": cross,
    }


def _pos_from_lookup(lookup: dict, spine_id: str, source: str, **extra) -> dict:
    rec = lookup[str(spine_id)]
    out = {
        "spine_id": str(spine_id),
        "x": float(rec["x"]),
        "y": float(rec["y"]),
        "z": float(rec["z"]),
        "dendrite_id": str(rec.get("dendrite_id") or ""),
        "source": source,
    }
    out.update(extra)
    return out


def _coord_fallback(x: float, y: float, z: float, source: str) -> dict:
    return {
        "spine_id": None,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "dendrite_id": "",
        "source": source,
        "score": None,
        "cross_dendrite": False,
    }


def build_lineage_positions(
    *,
    pre_spine_id: str,
    mid_spine_id: Optional[str],
    timepoint_names: List[str],
    spine_lookup: Dict[str, Dict[str, dict]],
    spine_dfs: Dict[str, pd.DataFrame],
    cross_links: List[dict],
    registry_members: Optional[Dict[str, dict]] = None,
    allow_cross_dendrite: bool = False,
    tiff_paths: Optional[Dict[str, str]] = None,
    claimed_spine_ids: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, dict]:
    """Build per-TP positions: pre→mid match, then chain algo; registry overrides.

    claimed_spine_ids (global ids per timepoint, claimed by OTHER lineages) are
    excluded from ranking/suggestion entirely - an already-claimed candidate is
    never offered as this lineage's match, falling through to the next-best
    free candidate or a coordinate-estimate fallback.
    """
    if not timepoint_names:
        return {}
    pre_tp = timepoint_names[0]
    positions: Dict[str, dict] = {}
    registry_members = registry_members or {}
    claimed_spine_ids = claimed_spine_ids or {}

    def _excluded_local_ids(tp: str) -> Set[str]:
        lookup = spine_lookup.get(tp) or {}
        return {
            _local_id_for_lookup(lookup, gid) for gid in claimed_spine_ids.get(tp, set())
        }

    pre_lookup = spine_lookup.get(pre_tp) or {}
    if str(pre_spine_id) not in pre_lookup:
        raise ValueError(f"Pre spine '{pre_spine_id}' not found.")
    positions[pre_tp] = _pos_from_lookup(pre_lookup, pre_spine_id, "pre_base")

    if len(timepoint_names) > 1:
        mid_tp = timepoint_names[1]
        mid_lookup = spine_lookup.get(mid_tp) or {}
        reg_mid = registry_members.get(mid_tp)
        if reg_mid:
            reg_sid = _global_id_for_lookup(
                mid_lookup, str(reg_mid.get("spine_id", ""))
            )
            if reg_sid in mid_lookup:
                positions[mid_tp] = _pos_from_lookup(
                    mid_lookup, reg_sid, "registry", score=None
                )
        elif (
            mid_spine_id
            and str(mid_spine_id) in mid_lookup
            and str(mid_spine_id) not in claimed_spine_ids.get(mid_tp, set())
        ):
            positions[mid_tp] = _pos_from_lookup(
                mid_lookup, str(mid_spine_id), "pre_mid_match", score=None
            )
        else:
            best = _score_pairwise_best(
                _local_id_for_lookup(pre_lookup, pre_spine_id),
                spine_dfs[pre_tp],
                spine_dfs[mid_tp],
                cross_links=cross_links,
                t1_tp=pre_tp,
                t2_tp=mid_tp,
                allow_cross=allow_cross_dendrite,
                t1_tiff_path=(tiff_paths or {}).get(pre_tp),
                t2_tiff_path=(tiff_paths or {}).get(mid_tp),
                exclude_t2_ids=_excluded_local_ids(mid_tp),
            )
            positions[mid_tp] = (
                _globalize_algo_hit(mid_lookup, best)
                if best
                else _coord_fallback(
                    float(positions[pre_tp]["x"]),
                    float(positions[pre_tp]["y"]),
                    float(positions[pre_tp]["z"]),
                    "coord_fallback",
                )
            )

    prev_tp = timepoint_names[1] if len(timepoint_names) > 1 else pre_tp
    prev_id = str(positions.get(prev_tp, {}).get("spine_id") or "")
    for tp in timepoint_names[2:]:
        reg = registry_members.get(tp)
        tp_lookup = spine_lookup.get(tp) or {}
        if reg:
            reg_sid = _global_id_for_lookup(tp_lookup, str(reg.get("spine_id", "")))
            if reg_sid in tp_lookup:
                positions[tp] = _pos_from_lookup(
                    tp_lookup, reg_sid, "registry", score=None
                )
                prev_tp, prev_id = tp, reg_sid
                continue
        if prev_id:
            prev_local = _local_id_for_lookup(
                spine_lookup.get(prev_tp) or {}, prev_id
            )
            best = _score_pairwise_best(
                prev_local,
                spine_dfs[prev_tp],
                spine_dfs[tp],
                cross_links=cross_links,
                t1_tp=prev_tp,
                t2_tp=tp,
                allow_cross=allow_cross_dendrite,
                t1_tiff_path=(tiff_paths or {}).get(prev_tp),
                t2_tiff_path=(tiff_paths or {}).get(tp),
                exclude_t2_ids=_excluded_local_ids(tp),
            )
            if best:
                positions[tp] = _globalize_algo_hit(tp_lookup, best)
                prev_tp, prev_id = tp, str(positions[tp]["spine_id"])
                continue
        prev_pos = positions.get(prev_tp) or positions[pre_tp]
        positions[tp] = _coord_fallback(
            float(prev_pos["x"]),
            float(prev_pos["y"]),
            float(prev_pos["z"]),
            "coord_region",
        )
        prev_tp = tp
        prev_id = ""

    return positions


def build_lineage_positions_from_anchor(
    *,
    anchor_tp: str,
    anchor_spine_id: str,
    timepoint_names: List[str],
    spine_lookup: Dict[str, Dict[str, dict]],
    spine_dfs: Dict[str, pd.DataFrame],
    cross_links: List[dict],
    allow_cross_dendrite: bool = False,
    tiff_paths: Optional[Dict[str, str]] = None,
    claimed_spine_ids: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, dict]:
    """Build lineage with anchor at any timepoint; chain forward and backward.

    claimed_spine_ids (global ids per timepoint, claimed by OTHER lineages) are
    excluded from ranking/suggestion entirely, same as build_lineage_positions.
    """
    if not timepoint_names:
        return {}
    if anchor_tp not in timepoint_names:
        raise ValueError(f"Anchor timepoint '{anchor_tp}' not in session.")
    anchor_lookup = spine_lookup.get(anchor_tp) or {}
    aid = str(anchor_spine_id)
    if aid not in anchor_lookup:
        raise ValueError(f"Spine '{aid}' not found at '{anchor_tp}'.")
    positions: Dict[str, dict] = {
        anchor_tp: _pos_from_lookup(anchor_lookup, aid, "anchor_base"),
    }
    anchor_idx = timepoint_names.index(anchor_tp)
    claimed_spine_ids = claimed_spine_ids or {}

    def _excluded_local_ids(tp: str) -> Set[str]:
        lookup = spine_lookup.get(tp) or {}
        return {
            _local_id_for_lookup(lookup, gid) for gid in claimed_spine_ids.get(tp, set())
        }

    prev_tp = anchor_tp
    prev_id = aid
    for tp in timepoint_names[anchor_idx + 1 :]:
        tp_lookup = spine_lookup.get(tp) or {}
        if prev_id and prev_tp in spine_dfs and tp in spine_dfs:
            best = _score_pairwise_best(
                _local_id_for_lookup(spine_lookup.get(prev_tp) or {}, prev_id),
                spine_dfs[prev_tp],
                spine_dfs[tp],
                cross_links=cross_links,
                t1_tp=prev_tp,
                t2_tp=tp,
                allow_cross=allow_cross_dendrite,
                t1_tiff_path=(tiff_paths or {}).get(prev_tp),
                t2_tiff_path=(tiff_paths or {}).get(tp),
                exclude_t2_ids=_excluded_local_ids(tp),
            )
            if best:
                positions[tp] = _globalize_algo_hit(tp_lookup, best)
                prev_tp, prev_id = tp, str(positions[tp]["spine_id"])
                continue
        ref = positions.get(prev_tp) or positions[anchor_tp]
        positions[tp] = _coord_fallback(
            float(ref["x"]),
            float(ref["y"]),
            float(ref["z"]),
            "coord_region",
        )
        prev_tp = tp
        prev_id = ""

    next_tp = anchor_tp
    next_id = aid
    for tp in reversed(timepoint_names[:anchor_idx]):
        tp_lookup = spine_lookup.get(tp) or {}
        if next_id and next_tp in spine_dfs and tp in spine_dfs:
            best = _score_pairwise_best(
                _local_id_for_lookup(spine_lookup.get(next_tp) or {}, next_id),
                spine_dfs[next_tp],
                spine_dfs[tp],
                cross_links=cross_links,
                t1_tp=next_tp,
                t2_tp=tp,
                allow_cross=allow_cross_dendrite,
                t1_tiff_path=(tiff_paths or {}).get(next_tp),
                t2_tiff_path=(tiff_paths or {}).get(tp),
                exclude_t2_ids=_excluded_local_ids(tp),
            )
            if best:
                positions[tp] = _globalize_algo_hit(tp_lookup, best)
                next_tp, next_id = tp, str(positions[tp]["spine_id"])
                continue
        ref = positions.get(next_tp) or positions[anchor_tp]
        positions[tp] = _coord_fallback(
            float(ref["x"]),
            float(ref["y"]),
            float(ref["z"]),
            "coord_region",
        )
        next_tp = tp
        next_id = ""

    return positions


def _anchor_pre_to_tp(
    pre_tp: str,
    pre_id: str,
    tp: str,
    tp_id: str,
    spine_lookup: Dict[str, Dict[str, dict]],
) -> dict:
    r1 = spine_lookup[pre_tp][str(pre_id)]
    r2 = spine_lookup[tp][str(tp_id)]
    return {
        "x2": float(r2["x"]),
        "y2": float(r2["y"]),
        "z2": float(r2["z"]),
        "dx": float(r1["x"]) - float(r2["x"]),
        "dy": float(r1["y"]) - float(r2["y"]),
        "dz": float(r1["z"]) - float(r2["z"]),
    }


def _estimate_shift_at(
    anchor_rows: List[dict],
    x2: float,
    y2: float,
    z2: float,
    *,
    window_px: float = 40.0,
) -> Tuple[float, float, float]:
    sum_w = 0.0
    sum_dx = sum_dy = sum_dz = 0.0
    for a in anchor_rows:
        d3 = float(
            np.sqrt((a["x2"] - x2) ** 2 + (a["y2"] - y2) ** 2 + (a["z2"] - z2) ** 2)
        )
        if d3 > window_px:
            continue
        w = 1.0 / (d3 + 1.0)
        sum_w += w
        sum_dx += w * float(a["dx"])
        sum_dy += w * float(a["dy"])
        sum_dz += w * float(a["dz"])
    if sum_w <= 0.0:
        return 0.0, 0.0, 0.0
    return float(sum_dx / sum_w), float(sum_dy / sum_w), float(sum_dz / sum_w)


def _predict_pre_location_in_tp(
    pre_tp: str,
    pre_id: str,
    tp: str,
    spine_lookup: Dict[str, Dict[str, dict]],
    anchor_tp: str,
    anchor_tp_id: str,
) -> Tuple[float, float, float]:
    """Map pre (reference) XYZ into tp imaging coordinates via one anchor pair."""
    row = _anchor_pre_to_tp(pre_tp, pre_id, anchor_tp, anchor_tp_id, spine_lookup)
    r1 = spine_lookup[pre_tp][str(pre_id)]
    bx, by, bz = float(r1["x"]), float(r1["y"]), float(r1["z"])
    sx, sy, sz = _estimate_shift_at([row], bx, by, bz)
    return bx - sx, by - sy, bz - sz


def apply_local_registration(
    positions: Dict[str, dict],
    *,
    pre_tp: str,
    pre_spine_id: str,
    timepoint_names: List[str],
    spine_lookup: Dict[str, Dict[str, dict]],
    cross_links: List[dict],
    linked_dendrites_fn,
    oof_segments: Optional[List[dict]] = None,
) -> Tuple[Dict[str, dict], Dict[str, Tuple[float, float, float]], bool]:
    """Refine coord-only panels using pre→mid (and other matched) anchors."""
    pre_id = str(pre_spine_id)
    if pre_tp not in spine_lookup or pre_id not in spine_lookup[pre_tp]:
        return positions, {}, False

    out = {k: dict(v) for k, v in positions.items()}
    shifts: Dict[str, Tuple[float, float, float]] = {pre_tp: (0.0, 0.0, 0.0)}
    r_pre = spine_lookup[pre_tp][pre_id]
    bx, by, bz = float(r_pre["x"]), float(r_pre["y"]), float(r_pre["z"])
    bd = str(r_pre.get("dendrite_id") or "")

    anchor_rows: List[dict] = []
    for tp in timepoint_names:
        if tp == pre_tp:
            continue
        pos = out.get(tp) or {}
        sid = str(pos.get("spine_id") or "")
        if sid and sid in (spine_lookup.get(tp) or {}):
            anchor_rows.append(_anchor_pre_to_tp(pre_tp, pre_id, tp, sid, spine_lookup))

    if not anchor_rows:
        return out, shifts, False

    sx, sy, sz = _estimate_shift_at(anchor_rows, bx, by, bz)
    for tp in timepoint_names:
        if tp == pre_tp:
            continue
        shifts[tp] = (sx, sy, sz)

    for tp in timepoint_names:
        if tp == pre_tp:
            continue
        pos = out.get(tp) or {}
        sid = str(pos.get("spine_id") or "")
        src = str(pos.get("source") or "")
        if sid and src not in {"coord_fallback", "coord_region", "local_reg_region"}:
            continue

        px, py, pz = bx - sx, by - sy, bz - sz
        if oof_segments and oof_segment_store.coords_in_oof(px, py, tp, oof_segments, dendrite_id=bd):
            out[tp] = {
                "spine_id": None,
                "x": px,
                "y": py,
                "z": pz,
                "dendrite_id": "",
                "source": "oof_region",
                "score": None,
                "cross_dendrite": False,
            }
            continue
        lookup = spine_lookup.get(tp) or {}
        linked = linked_dendrites_fn(pre_tp, bd, tp) if bd else set()
        near = None
        best_d = float("inf")
        for rec_id, rec in lookup.items():
            did = str(rec.get("dendrite_id") or "")
            if linked and did not in linked:
                continue
            if oof_segments and oof_segment_store.coords_in_oof(
                float(rec["x"]), float(rec["y"]), tp, oof_segments, dendrite_id=did
            ):
                continue
            d = float(
                np.hypot(float(rec["x"]) - px, float(rec["y"]) - py)
                + 0.25 * abs(float(rec["z"]) - pz)
            )
            if d < best_d:
                best_d = d
                near = {
                    "spine_id": str(rec_id),
                    "x": float(rec["x"]),
                    "y": float(rec["y"]),
                    "z": float(rec["z"]),
                    "dendrite_id": did,
                    "source": "local_reg_nearest",
                    "score": pos.get("score"),
                    "cross_dendrite": pos.get("cross_dendrite", False),
                }
        if near:
            out[tp] = near
        else:
            out[tp] = {
                "spine_id": None,
                "x": px,
                "y": py,
                "z": pz,
                "dendrite_id": "",
                "source": "local_reg_region",
                "score": None,
                "cross_dendrite": False,
            }

    return out, shifts, True
