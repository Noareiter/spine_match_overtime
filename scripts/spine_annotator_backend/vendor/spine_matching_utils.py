from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tifffile


REQUIRED_SPINE_COLS = ["id", "x", "y", "z"]


def canonical_spine_id(value: Any) -> str:
    s = str(value).strip()
    if s == "":
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_stack(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {arr.shape}")
    return arr.astype(np.float32)


def load_spines_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept legacy names if present.
    rename_map = {}
    if "spine_id" in df.columns and "id" not in df.columns:
        rename_map["spine_id"] = "id"
    if rename_map:
        df = df.rename(columns=rename_map)
    for c in REQUIRED_SPINE_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")
    df["id"] = df["id"].apply(canonical_spine_id)
    for c in ("x", "y", "z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    return df


def normalize_for_display(stack: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(stack, [1, 99])
    if p99 <= p1:
        return np.zeros_like(stack, dtype=np.float32)
    out = np.clip((stack - p1) / (p99 - p1), 0, 1)
    return out.astype(np.float32)


def spine_points_xyz(df: pd.DataFrame) -> np.ndarray:
    # napari points use (z, y, x)
    return df[["z", "y", "x"]].to_numpy(dtype=np.float32)


def spine_id_sort_key(s: str) -> Tuple[int, Union[int, str], str]:
    """Sort key: numeric ids first (ascending), then lexicographic, then hil_manual_t2_* sequence."""
    s = str(s).strip()
    pfx = "hil_manual_t2_"
    if s.startswith(pfx):
        tail = s[len(pfx) :]
        try:
            return (2, int(tail, 10), "")
        except ValueError:
            return (3, 0, s.lower())
    try:
        return (0, int(s, 10), "")
    except ValueError:
        return (1, 0, s.lower())


def sorted_spine_ids(ids) -> List[str]:
    return sorted((str(x) for x in ids), key=spine_id_sort_key)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_corrections_csv(path: str, rows: list[dict]):
    ensure_parent(path)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=["t2_spine_id", "original_t1_id", "corrected_t1_id", "action", "notes"]
        )
    df.to_csv(path, index=False)


def load_optional_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def serialize_session(path: str, data: Dict):
    ensure_parent(path)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def deserialize_session(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def maybe_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)


def build_candidate_table(
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    max_xy_radius: float = 30.0,
    max_z_delta: float = 4.0,
) -> pd.DataFrame:
    """
    Build candidate pairs from spatial neighborhood only.
    Returned columns include core feature-ready fields.
    """
    rows = []
    t1_arr = t1_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    t2_arr = t2_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    for j, t2 in enumerate(t2_arr):
        dx = t1_arr[:, 0] - t2[0]
        dy = t1_arr[:, 1] - t2[1]
        dz = t1_arr[:, 2] - t2[2]
        dxy = np.hypot(dx, dy)
        keep = (dxy <= max_xy_radius) & (np.abs(dz) <= max_z_delta)
        idxs = np.where(keep)[0]
        if len(idxs) == 0:
            continue
        order = idxs[np.argsort(dxy[idxs])]
        for rank, i in enumerate(order, start=1):
            rows.append(
                {
                    "t2_spine_id": str(t2_df.iloc[j]["id"]),
                    "t1_spine_id": str(t1_df.iloc[i]["id"]),
                    "distance_xy": float(dxy[i]),
                    "distance_z": float(abs(dz[i])),
                    "candidate_rank": float(rank),
                }
            )
    return pd.DataFrame(rows)


def local_density(df: pd.DataFrame, radius_xy: float = 12.0) -> pd.Series:
    pts = df[["x", "y"]].to_numpy(dtype=np.float32)
    out = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        out[i] = float(np.sum(d <= radius_xy) - 1)
    return pd.Series(out, index=df.index)


def compute_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dendrite_id exists, estimate local branch angle per spine from PCA direction.
    """
    res = pd.DataFrame(index=df.index)
    if "dendrite_id" not in df.columns:
        res["branch_angle"] = np.nan
        return res
    angles = np.full(len(df), np.nan, dtype=np.float32)
    for did, sub in df.groupby("dendrite_id"):
        pts = sub[["x", "y"]].to_numpy(dtype=np.float32)
        if len(sub) < 2:
            continue
        c = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
        axis = vh[0]
        ang = float(np.arctan2(axis[1], axis[0]))
        angles[sub.index.values] = ang
    res["branch_angle"] = angles
    return res


def safe_angle_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    d = abs(a - b)
    return float(min(d, 2 * np.pi - d))


def prepare_spine_feature_maps(t1_df: pd.DataFrame, t2_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t1 = t1_df.copy()
    t2 = t2_df.copy()
    t1["local_density"] = local_density(t1)
    t2["local_density"] = local_density(t2)
    t1 = pd.concat([t1, compute_direction(t1)], axis=1)
    t2 = pd.concat([t2, compute_direction(t2)], axis=1)
    return t1, t2
