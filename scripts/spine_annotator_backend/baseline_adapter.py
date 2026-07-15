from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from .project_paths import tracking_scripts_root

_PROJECT_SCRIPTS_DIR = tracking_scripts_root()
HYBRID_PATH = _PROJECT_SCRIPTS_DIR / "hybrid_tracking" / "track_hybrid.py"
SPINE_UTILS_PATH = _PROJECT_SCRIPTS_DIR / "step2-spine tracking" / "spine_matching_tool" / "utils.py"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_hybrid = _load_module("track_hybrid_backend", HYBRID_PATH)
_spine_utils = _load_module("spine_matching_utils_backend", SPINE_UTILS_PATH)
LOCAL_REG_WINDOW_PX = 40.0
# Hard cap: never score or suggest T1–T2 pairs with |Δz| above this (CSV z units).
MAX_MATCH_Z_GAP = 7.0


def load_stack(path: Path) -> np.ndarray:
    return _spine_utils.load_stack(str(path))


def load_spines(path: Path) -> pd.DataFrame:
    return _hybrid.load_spines(path)


def to_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        spine_id = str(row["id"])
        feature_map = {}
        for col in df.columns:
            if col in {"id", "x", "y", "z"}:
                continue
            val = row[col]
            feature_map[col] = None if pd.isna(val) else val
        rows[spine_id] = {
            "spine_id": spine_id,
            "dendrite_id": None if "dendrite_id" not in row or pd.isna(row.get("dendrite_id")) else str(row.get("dendrite_id")),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "features": feature_map,
        }
    return rows


def dendrite_groups(df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    if "dendrite_id" not in df.columns:
        return []
    work = df.copy()
    work["dendrite_id"] = work["dendrite_id"].astype(str)
    grouped = work.groupby("dendrite_id")["id"].apply(lambda s: [str(v) for v in s.tolist()])
    return [(d_id, ids) for d_id, ids in grouped.items()]


def exclude_matched_spines(
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    matched_t1_ids: set[str],
    matched_t2_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_t1 = t1_df.copy()
    out_t2 = t2_df.copy()
    if matched_t1_ids:
        out_t1 = out_t1[~out_t1["id"].astype(str).isin(matched_t1_ids)]
    if matched_t2_ids:
        out_t2 = out_t2[~out_t2["id"].astype(str).isin(matched_t2_ids)]
    return out_t1, out_t2


def linked_dendrite_map(dendrite_links: List[Dict[str, object]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for link in dendrite_links:
        t1_ids = {str(v) for v in link.get("t1_dendrite_ids", [])}
        for t2_id in link.get("t2_dendrite_ids", []):
            out.setdefault(str(t2_id), set()).update(t1_ids)
    return out


def score_candidates_hybrid(
    candidates: pd.DataFrame,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    *,
    gating_xy: float = 80.0,
    gating_z: float = MAX_MATCH_Z_GAP,
    w_xy: float = 0.45,
    w_z: float = 0.20,
    w_feat: float = 0.35,
    w_app: float = 0.0,
    anchors: Optional[List[Dict[str, str]]] = None,
    nearby_xy: float = 140.0,
    t1_tiff_path: Optional[str] = None,
    t2_tiff_path: Optional[str] = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    working = candidates.copy()
    if anchors:
        t1_by = t1_df.set_index(t1_df["id"].astype(str))
        t2_by = t2_df.set_index(t2_df["id"].astype(str))
        shift_by_t2: Dict[str, Tuple[float, float, float]] = {}
        t2_points = {
            str(r["id"]): (float(r["x"]), float(r["y"]), float(r["z"]))
            for _, r in t2_df.iterrows()
        }
        anchor_rows = []
        for a in anchors:
            t1_id = str(a.get("t1_spine_id", ""))
            t2_id = str(a.get("t2_spine_id", ""))
            if t1_id not in t1_by.index or t2_id not in t2_by.index:
                continue
            r1 = t1_by.loc[t1_id]
            r2 = t2_by.loc[t2_id]
            anchor_rows.append(
                {
                    "t2_id": t2_id,
                    "x2": float(r2["x"]),
                    "y2": float(r2["y"]),
                    "z2": float(r2["z"]),
                    "dx": float(r1["x"]) - float(r2["x"]),
                    "dy": float(r1["y"]) - float(r2["y"]),
                    "dz": float(r1["z"]) - float(r2["z"]),
                }
            )
        if anchor_rows:
            for t2_id, (x2, y2, _z2) in t2_points.items():
                sum_w = 0.0
                sum_dx = 0.0
                sum_dy = 0.0
                sum_dz = 0.0
                for a in anchor_rows:
                    d3 = float(
                        np.sqrt(
                            (a["x2"] - x2) ** 2
                            + (a["y2"] - y2) ** 2
                            + (a["z2"] - _z2) ** 2
                        )
                    )
                    if d3 > LOCAL_REG_WINDOW_PX:
                        continue
                    w = 1.0 / (d3 + 1.0)
                    sum_w += w
                    sum_dx += w * float(a["dx"])
                    sum_dy += w * float(a["dy"])
                    sum_dz += w * float(a["dz"])
                if sum_w > 0.0:
                    sx = float(sum_dx / sum_w)
                    sy = float(sum_dy / sum_w)
                    sz = float(sum_dz / sum_w)
                    shift_by_t2[t2_id] = (sx, sy, sz)
            if shift_by_t2:
                t1_coords = {
                    str(r["id"]): (float(r["x"]), float(r["y"]), float(r["z"]))
                    for _, r in t1_df.iterrows()
                }
                t2_coords = {
                    str(r["id"]): (float(r["x"]), float(r["y"]), float(r["z"]))
                    for _, r in t2_df.iterrows()
                }
                dxy_new = []
                dz_new = []
                for _, r in working.iterrows():
                    t1_id = str(r["t1_spine_id"])
                    t2_id = str(r["t2_spine_id"])
                    p1 = t1_coords.get(t1_id)
                    p2 = t2_coords.get(t2_id)
                    if p1 is None or p2 is None:
                        dxy_new.append(float(r.get("distance_xy", np.inf)))
                        dz_new.append(float(r.get("distance_z", np.inf)))
                        continue
                    sx, sy, sz = shift_by_t2.get(t2_id, (0.0, 0.0, 0.0))
                    x2s = p2[0] + sx
                    y2s = p2[1] + sy
                    z2s = p2[2] + sz
                    dxy_new.append(float(np.hypot(p1[0] - x2s, p1[1] - y2s)))
                    dz_new.append(float(abs(p1[2] - z2s)))
                working["distance_xy"] = dxy_new
                working["distance_z"] = dz_new

    if "distance_z" in working.columns:
        working = working[working["distance_z"] <= MAX_MATCH_Z_GAP].copy()
    if working.empty:
        return working.copy()

    scalers = _hybrid.robust_scalers(t1_df, t2_df, _hybrid.FEATURE_KEYS)
    t1_scaled = _hybrid.add_scaled_features(t1_df, scalers, _hybrid.FEATURE_KEYS)
    t2_scaled = _hybrid.add_scaled_features(t2_df, scalers, _hybrid.FEATURE_KEYS)
    scored = _hybrid.build_feature_weighted_scores(
        candidates=working,
        t1_df=t1_scaled,
        t2_df=t2_scaled,
        gating_xy=gating_xy,
        gating_z=gating_z,
        w_xy=w_xy,
        w_z=w_z,
        w_feat=w_feat,
    )
    scored["score_toolb_model"] = np.nan
    scored["stability_score"] = _hybrid.compute_stability_proxy(scored, t1_scaled, t2_scaled)

    # Add appearance scores if TIFF paths provided and w_app > 0
    if w_app > 0.0 and t1_tiff_path and t2_tiff_path:
        scored = _add_appearance_scores(
            scored,
            t1_df=t1_df,
            t2_df=t2_df,
            t1_tiff_path=t1_tiff_path,
            t2_tiff_path=t2_tiff_path,
        )

    out = _hybrid.finalize_scores(scored, mode="hybrid_confidence")

    # Re-blend with appearance scores if available
    if "score_app" in out.columns and np.isfinite(out["score_app"]).any() and w_app > 0.0:
        s_feat = out["score_feature_weighted"].fillna(0.0)
        s_app = out["score_app"].fillna(0.0)
        stability = out["stability_score"].fillna(0.5)
        w_feat_eff = max(w_feat, 0.0)
        w_app_eff = max(w_app, 0.0)
        w_stab = 0.15
        denom = w_feat_eff + w_app_eff + w_stab
        if denom > 0:
            out["final_score"] = (
                w_feat_eff * s_feat + w_app_eff * s_app + w_stab * stability
            ) / denom
            out["final_score"] = out["final_score"].clip(0.0, 1.0)

    return out


def _add_appearance_scores(
    scored: pd.DataFrame,
    *,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    t1_tiff_path: Optional[str],
    t2_tiff_path: Optional[str],
) -> pd.DataFrame:
    """Optional Siamese appearance score via SPINE_MATCHER_CHECKPOINT.

    Encodes spine crops via pre-trained model and scores cosine similarity,
    then calibrates to probability space via sigmoid(logit_scale * (cosine - tau)).
    """
    out = scored.copy()
    out["score_app"] = np.nan
    if not t1_tiff_path or not t2_tiff_path or scored.empty:
        return out
    try:
        import sys
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from spine_matcher.infer import get_matcher

        matcher = get_matcher()
        if matcher is None:
            return out
    except Exception:
        return out

    t1_by = t1_df.set_index(t1_df["id"].astype(str))
    t2_by = t2_df.set_index(t2_df["id"].astype(str))
    tau = getattr(matcher, "tau", 0.175)
    logit_scale = 10.0
    apps: List[float] = []
    for _, r in out.iterrows():
        t1_id = str(r["t1_spine_id"])
        t2_id = str(r["t2_spine_id"])
        if t1_id not in t1_by.index or t2_id not in t2_by.index:
            apps.append(np.nan)
            continue
        p1 = t1_by.loc[t1_id]
        p2 = t2_by.loc[t2_id]
        try:
            sim = matcher.score_coords(
                t1_tiff_path,
                float(p1["x"]),
                float(p1["y"]),
                float(p1["z"]),
                t2_tiff_path,
                float(p2["x"]),
                float(p2["y"]),
                float(p2["z"]),
            )
            # Calibrate raw cosine with sigmoid(logit_scale * (cosine - tau))
            calibrated = 1.0 / (1.0 + np.exp(-logit_scale * (sim - tau)))
            apps.append(float(np.clip(calibrated, 0.0, 1.0)))
        except Exception:
            apps.append(np.nan)
    out["score_app"] = apps
    return out

