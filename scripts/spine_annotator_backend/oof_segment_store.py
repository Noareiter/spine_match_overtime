"""Out-of-focus region persistence — rectangle or freehand polygon."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import dendrite_link_store


def _meta_dir(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov)


def segments_path(respan: Path, fov: int) -> Path:
    return _meta_dir(respan, fov) / "oof_segments.json"


def load_segments(respan: Path, fov: int) -> List[dict]:
    path = segments_path(respan, fov)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("segments") or [])


def save_segments(respan: Path, fov: int, segments: List[dict], *, animal_id: str = "") -> Path:
    path = segments_path(respan, fov)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "segments": segments,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _norm_bounds(b: dict) -> dict:
    x0 = float(b.get("x0", 0))
    y0 = float(b.get("y0", 0))
    x1 = float(b.get("x1", x0))
    y1 = float(b.get("y1", y0))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def rect_shape(x0: float, y0: float, x1: float, y1: float) -> dict:
    b = _norm_bounds({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return {"shape": "rect", **b}


def polygon_shape(points: List[List[float]]) -> dict:
    pts = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
    if len(pts) < 3:
        raise ValueError("Freehand polygon needs at least 3 points.")
    return {"shape": "polygon", "points": pts}


def normalize_shape(data: Optional[dict]) -> Optional[dict]:
    if not data:
        return None
    if str(data.get("shape") or "") == "polygon":
        return polygon_shape(list(data.get("points") or []))
    if "x0" in data or str(data.get("shape") or "") == "rect":
        b = _norm_bounds(data)
        return {"shape": "rect", **b}
    return None


def shape_bounds(shape: Optional[dict]) -> Optional[dict]:
    """Axis-aligned bbox for a shape (overlay culling)."""
    sh = normalize_shape(shape)
    if not sh:
        return None
    if sh["shape"] == "rect":
        return _norm_bounds(sh)
    xs = [p[0] for p in sh["points"]]
    ys = [p[1] for p in sh["points"]]
    return _norm_bounds({"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})


def point_in_bounds(x: float, y: float, bounds: Optional[dict]) -> bool:
    if not bounds:
        return False
    b = _norm_bounds(bounds)
    return b["x0"] <= float(x) <= b["x1"] and b["y0"] <= float(y) <= b["y1"]


def point_in_polygon(x: float, y: float, points: List[List[float]]) -> bool:
    n = len(points)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(points[i][0]), float(points[i][1])
        xj, yj = float(points[j][0]), float(points[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_shape(x: float, y: float, shape: Optional[dict]) -> bool:
    sh = normalize_shape(shape)
    if not sh:
        return False
    if sh["shape"] == "polygon":
        return point_in_polygon(x, y, sh["points"])
    return point_in_bounds(x, y, sh)


def _segment_shape(seg: dict, tp: str, pre_tp: str) -> Optional[dict]:
    per_tp = seg.get("per_tp") or {}
    raw = per_tp.get(tp)
    if raw is None and tp == str(seg.get("pre_tp", "")):
        raw = seg.get("pre_shape") or seg.get("pre_bounds")
    elif raw is None and tp == pre_tp:
        raw = seg.get("pre_shape") or seg.get("pre_bounds")
    return normalize_shape(raw)


def coords_in_oof(
    x: float,
    y: float,
    tp: str,
    segments: List[dict],
    *,
    dendrite_id: str = "",
) -> bool:
    did = str(dendrite_id or "")
    pre_tp = ""
    for seg in segments:
        if not pre_tp:
            pre_tp = str(seg.get("pre_tp", ""))
        if did and str(seg.get("dendrite_id_pre", "")) not in ("", did):
            if str(seg.get("dendrite_id_pre", "")) != did:
                continue
        sh = _segment_shape(seg, tp, str(seg.get("pre_tp", "")))
        if point_in_shape(x, y, sh):
            return True
    return False


def spines_inside_shape(
    shape: dict,
    tp: str,
    spine_lookup: Dict[str, dict],
) -> List[str]:
    sh = normalize_shape(shape)
    if not sh:
        return []
    out: List[str] = []
    for sid, rec in spine_lookup.items():
        if str(sid).startswith("bridge_") or str(sid).startswith("artifact_"):
            continue
        x = rec.get("x")
        y = rec.get("y")
        if x is None or y is None:
            continue
        if point_in_shape(float(x), float(y), sh):
            out.append(str(sid))
    return sorted(out)


def shape_between_timepoints(
    shape: dict,
    src_tp: str,
    dst_tp: str,
    *,
    pre_tp: str,
    shifts: Dict[str, Tuple[float, float, float]],
) -> dict:
    sh = normalize_shape(shape)
    if not sh:
        raise ValueError("Invalid shape.")
    if src_tp == dst_tp:
        return sh
    if sh["shape"] == "rect":
        return normalize_shape(
            bounds_between_timepoints(sh, src_tp, dst_tp, pre_tp=pre_tp, shifts=shifts)
        ) or sh
    sx_s, sy_s, _ = shifts.get(src_tp, (0.0, 0.0, 0.0))
    sx_d, sy_d, _ = shifts.get(dst_tp, (0.0, 0.0, 0.0))

    def to_pre(x: float, y: float) -> Tuple[float, float]:
        if src_tp != pre_tp:
            return x - sx_s, y - sy_s
        return x, y

    def from_pre(x: float, y: float) -> Tuple[float, float]:
        if dst_tp != pre_tp:
            return x + sx_d, y + sy_d
        return x, y

    pts = [list(from_pre(*to_pre(p[0], p[1]))) for p in sh["points"]]
    return polygon_shape(pts)


def bounds_between_timepoints(
    bounds: dict,
    src_tp: str,
    dst_tp: str,
    *,
    pre_tp: str,
    shifts: Dict[str, Tuple[float, float, float]],
) -> dict:
    if src_tp == dst_tp:
        return _norm_bounds(bounds)
    sx_s, sy_s, _ = shifts.get(src_tp, (0.0, 0.0, 0.0))
    sx_d, sy_d, _ = shifts.get(dst_tp, (0.0, 0.0, 0.0))
    b = _norm_bounds(bounds)
    if src_tp != pre_tp:
        x0, y0, x1, y1 = b["x0"] - sx_s, b["y0"] - sy_s, b["x1"] - sx_s, b["y1"] - sy_s
    else:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
    if dst_tp != pre_tp:
        return _norm_bounds(
            {"x0": x0 + sx_d, "y0": y0 + sy_d, "x1": x1 + sx_d, "y1": y1 + sy_d}
        )
    return _norm_bounds({"x0": x0, "y0": y0, "x1": x1, "y1": y1})


def translate_bounds(bounds: dict, dx: float, dy: float) -> dict:
    b = _norm_bounds(bounds)
    return {
        "x0": b["x0"] + dx,
        "y0": b["y0"] + dy,
        "x1": b["x1"] + dx,
        "y1": b["y1"] + dy,
    }


def upsert_segment(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    segment_id: Optional[str],
    dendrite_id_pre: str,
    pre_tp: str,
    pre_shape: dict,
    per_tp: Optional[Dict[str, dict]] = None,
    notes: str = "",
) -> dict:
    segments = load_segments(respan, fov)
    sid = str(segment_id or "").strip() or f"seg_{uuid.uuid4().hex[:8]}"
    pre_n = normalize_shape(pre_shape)
    if not pre_n:
        raise ValueError("Invalid pre_shape.")
    per_norm = {
        k: normalize_shape(v) for k, v in (per_tp or {}).items() if normalize_shape(v)
    }
    entry = {
        "segment_id": sid,
        "dendrite_id_pre": str(dendrite_id_pre),
        "pre_tp": str(pre_tp),
        "pre_shape": pre_n,
        "pre_bounds": shape_bounds(pre_n),
        "per_tp": per_norm,
        "notes": str(notes or ""),
    }
    replaced = False
    for i, seg in enumerate(segments):
        if str(seg.get("segment_id", "")) == sid:
            segments[i] = entry
            replaced = True
            break
    if not replaced:
        segments.append(entry)
    save_segments(respan, fov, segments, animal_id=animal_id)
    return entry


def update_tp_bounds(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    segment_id: str,
    tp: str,
    bounds: Optional[dict],
) -> Optional[dict]:
    return update_tp_shape(
        respan, fov, animal_id=animal_id, segment_id=segment_id, tp=tp, shape=bounds
    )


def update_tp_shape(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    segment_id: str,
    tp: str,
    shape: Optional[dict],
) -> Optional[dict]:
    segments = load_segments(respan, fov)
    for i, seg in enumerate(segments):
        if str(seg.get("segment_id", "")) != str(segment_id):
            continue
        pre_tp = str(seg.get("pre_tp", ""))
        if shape is None:
            if tp != pre_tp:
                per_tp = dict(seg.get("per_tp") or {})
                per_tp.pop(tp, None)
                seg["per_tp"] = per_tp
        else:
            sh = normalize_shape(shape)
            if not sh:
                raise ValueError("Invalid shape.")
            if tp == pre_tp:
                seg["pre_shape"] = sh
                seg["pre_bounds"] = shape_bounds(sh)
            else:
                per_tp = dict(seg.get("per_tp") or {})
                per_tp[tp] = sh
                seg["per_tp"] = per_tp
        segments[i] = seg
        save_segments(respan, fov, segments, animal_id=animal_id)
        return seg
    return None


def segment_shape_at_timepoint(seg: dict, tp: str) -> Optional[dict]:
    pre_tp = str(seg.get("pre_tp", ""))
    return _segment_shape(seg, tp, pre_tp)


def translate_shape_by(shape: dict, dx: float, dy: float) -> dict:
    sh = normalize_shape(shape)
    if not sh:
        raise ValueError("Invalid shape.")
    if sh["shape"] == "polygon":
        return {"shape": "polygon", "points": [[p[0] + dx, p[1] + dy] for p in sh["points"]]}
    return {"shape": "rect", "x0": sh["x0"] + dx, "y0": sh["y0"] + dy,
            "x1": sh["x1"] + dx, "y1": sh["y1"] + dy}


def ignored_spines_by_timepoint(
    segments: List[dict],
    timepoints: List[str],
    spine_lookup: Dict[str, Dict[str, dict]],
) -> Dict[str, Set[str]]:
    """Union of spine IDs inside any OOF segment, per timepoint."""
    out: Dict[str, Set[str]] = {str(tp): set() for tp in timepoints}
    for seg in segments:
        pre_tp = str(seg.get("pre_tp", ""))
        for tp in timepoints:
            sh = _segment_shape(seg, tp, pre_tp)
            if not sh:
                continue
            lookup = spine_lookup.get(tp) or {}
            out[str(tp)].update(spines_inside_shape(sh, tp, lookup))
    return out


def delete_segment(respan: Path, fov: int, segment_id: str, *, animal_id: str = "") -> bool:
    segments = load_segments(respan, fov)
    before = len(segments)
    segments = [s for s in segments if str(s.get("segment_id", "")) != str(segment_id)]
    if len(segments) == before:
        return False
    save_segments(respan, fov, segments, animal_id=animal_id)
    return True
