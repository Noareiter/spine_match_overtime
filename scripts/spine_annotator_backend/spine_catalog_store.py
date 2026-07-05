"""Global spine catalog: unique S_* IDs for every detection across all timepoints."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import baseline_adapter, dendrite_link_store

CATALOG_FILENAME = "spine_catalog.csv"
GLOBAL_PREFIX = "S_"

CATALOG_COLUMNS = [
    "global_spine_id",
    "animal_id",
    "fov",
    "timepoint",
    "local_spine_id",
    "x",
    "y",
    "z",
    "dendrite_id",
    "source_csv",
]


def catalog_path(respan: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan, fov) / CATALOG_FILENAME


def is_global_id(spine_id: str) -> bool:
    return str(spine_id or "").strip().startswith(GLOBAL_PREFIX)


def _format_global_id(n: int) -> str:
    return f"{GLOBAL_PREFIX}{n:05d}"


def _sort_local_ids(ids: Iterable[str]) -> List[str]:
    def key(s: str):
        try:
            return (0, float(s), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(s))

    return sorted({str(x).strip() for x in ids if str(x).strip()}, key=key)


@dataclass
class SpineCatalog:
    animal_id: str
    fov: int
    rows: List[dict] = field(default_factory=list)
    by_global: Dict[str, dict] = field(default_factory=dict)
    local_to_global: Dict[str, Dict[str, str]] = field(default_factory=dict)
    global_by_tp: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: List[dict], *, animal_id: str = "", fov: int = 0) -> "SpineCatalog":
        cat = cls(animal_id=str(animal_id or ""), fov=int(fov or 0), rows=list(rows))
        cat._index()
        return cat

    def _index(self) -> None:
        self.by_global = {}
        self.local_to_global = {}
        self.global_by_tp = {}
        for row in self.rows:
            gid = str(row["global_spine_id"])
            tp = str(row["timepoint"])
            local = str(row["local_spine_id"])
            self.by_global[gid] = row
            self.local_to_global.setdefault(tp, {})[local] = gid
            self.global_by_tp.setdefault(tp, {})[gid] = {
                "spine_id": gid,
                "global_spine_id": gid,
                "local_spine_id": local,
                "label": local,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "dendrite_id": str(row.get("dendrite_id") or ""),
                "features": {},
            }

    def global_id(self, timepoint: str, local_id: str) -> Optional[str]:
        return self.local_to_global.get(str(timepoint), {}).get(str(local_id))

    def local_id(self, timepoint: str, global_id: str) -> Optional[str]:
        row = self.by_global.get(str(global_id))
        if not row:
            return None
        if str(row.get("timepoint")) != str(timepoint):
            return None
        return str(row.get("local_spine_id") or "")

    def lookup_for_timepoint(self, timepoint: str) -> Dict[str, dict]:
        return dict(self.global_by_tp.get(str(timepoint), {}))

    def to_local(self, timepoint: str, spine_id: str) -> str:
        """Return local CSV id; pass through manual_/bridge_/already-global unknown."""
        sid = str(spine_id or "").strip()
        if not sid or manual_or_special(sid):
            return sid
        if is_global_id(sid):
            return self.local_id(timepoint, sid) or sid
        return sid

    def to_global(self, timepoint: str, spine_id: str) -> str:
        sid = str(spine_id or "").strip()
        if not sid or manual_or_special(sid):
            return sid
        if is_global_id(sid):
            return sid
        return self.global_id(timepoint, sid) or sid

    def display_label(self, timepoint: str, spine_id: str) -> str:
        sid = str(spine_id or "").strip()
        if not sid:
            return ""
        if manual_or_special(sid):
            return sid
        if is_global_id(sid):
            local = self.local_id(timepoint, sid)
            return f"{local} ({sid})" if local else sid
        gid = self.global_id(timepoint, sid)
        return f"{sid} ({gid})" if gid else sid


def manual_or_special(spine_id: str) -> bool:
    s = str(spine_id or "").strip()
    return s.startswith("manual_") or s.startswith("bridge_") or s.startswith("artifact_")


def build_catalog(
    *,
    animal_id: str,
    fov: int,
    timepoint_names: List[str],
    csv_by_tp: Dict[str, Path],
) -> SpineCatalog:
    """Assign S_* IDs to every row in each timepoint detection CSV."""
    rows: List[dict] = []
    n = 1
    for tp in timepoint_names:
        csv_path = csv_by_tp.get(tp)
        if not csv_path or not Path(csv_path).is_file():
            continue
        df = baseline_adapter.load_spines(Path(csv_path))
        df = df.copy()
        df["id"] = df["id"].astype(str)
        for local_id in _sort_local_ids(df["id"].tolist()):
            rec = df[df["id"] == local_id].iloc[0]
            rows.append(
                {
                    "global_spine_id": _format_global_id(n),
                    "animal_id": str(animal_id or ""),
                    "fov": str(fov),
                    "timepoint": str(tp),
                    "local_spine_id": str(local_id),
                    "x": float(rec["x"]),
                    "y": float(rec["y"]),
                    "z": float(rec["z"]),
                    "dendrite_id": ""
                    if "dendrite_id" not in rec or pd.isna(rec.get("dendrite_id"))
                    else str(rec.get("dendrite_id")),
                    "source_csv": str(csv_path),
                }
            )
            n += 1
    return SpineCatalog.from_rows(rows, animal_id=animal_id, fov=fov)


def save_catalog(respan: Path, fov: int, catalog: SpineCatalog) -> Path:
    path = catalog_path(respan, fov)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in catalog.rows:
            writer.writerow({k: row.get(k, "") for k in CATALOG_COLUMNS})
    return path


def load_catalog(respan: Path, fov: int) -> Optional[SpineCatalog]:
    path = catalog_path(respan, fov)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    animal_id = str(rows[0].get("animal_id") or "")
    fov_val = int(rows[0].get("fov") or fov)
    for row in rows:
        row["x"] = float(row.get("x") or 0)
        row["y"] = float(row.get("y") or 0)
        row["z"] = float(row.get("z") or 0)
    return SpineCatalog.from_rows(rows, animal_id=animal_id, fov=fov_val)


def _catalog_stale(
    catalog_path: Path,
    catalog: SpineCatalog,
    timepoint_names: List[str],
    csv_by_tp: Dict[str, Path],
) -> bool:
    if not catalog_path.is_file():
        return True
    cat_mtime = catalog_path.stat().st_mtime
    for tp in timepoint_names:
        csv_p = Path(csv_by_tp.get(tp) or "")
        if not csv_p.is_file():
            continue
        if csv_p.stat().st_mtime > cat_mtime:
            return True
        expected = sum(1 for row in catalog.rows if str(row.get("timepoint")) == str(tp))
        df = baseline_adapter.load_spines(csv_p)
        if len(df) != expected:
            return True
    return False


def ensure_catalog(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    timepoint_names: List[str],
    csv_by_tp: Dict[str, Path],
    rebuild: bool = False,
) -> Tuple[SpineCatalog, Path, bool]:
    """Load or build catalog. Returns (catalog, path, created)."""
    path = catalog_path(respan, fov)
    existing = None if rebuild else load_catalog(respan, fov)
    if existing and not _catalog_stale(path, existing, timepoint_names, csv_by_tp):
        return existing, path, False
    catalog = build_catalog(
        animal_id=animal_id,
        fov=fov,
        timepoint_names=timepoint_names,
        csv_by_tp=csv_by_tp,
    )
    path = save_catalog(respan, fov, catalog)
    return catalog, path, True


def sync_catalog_for_fov(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    csv_by_tp: Dict[str, Path],
    timepoint_names: Optional[List[str]] = None,
    rebuild: bool = False,
) -> dict:
    """Build or load spine_catalog.csv for a FOV. Safe to call on every app load."""
    tps = [str(t) for t in (timepoint_names or list(csv_by_tp.keys())) if str(t) in csv_by_tp]
    if not tps:
        return {"catalog_path": "", "catalog_count": 0, "catalog_created": False, "message": ""}
    catalog, path, created = ensure_catalog(
        respan,
        int(fov),
        animal_id=str(animal_id or ""),
        timepoint_names=tps,
        csv_by_tp=csv_by_tp,
        rebuild=rebuild,
    )
    verb = "built" if created else "loaded"
    return {
        "catalog_path": str(path),
        "catalog_count": len(catalog.rows),
        "catalog_created": created,
        "message": f"Spine catalog {verb}: {len(catalog.rows)} detections (S_* IDs)",
    }


def merge_manual_into_lookup(
    lookup: Dict[str, dict],
    manual_rows: Iterable[dict],
) -> None:
    """Add manual spines (already unique manual_* ids) into a per-TP lookup."""
    for rec in manual_rows:
        sid = str(rec.get("spine_id") or "").strip()
        if not sid:
            continue
        lookup[sid] = {
            "spine_id": sid,
            "global_spine_id": sid,
            "local_spine_id": sid,
            "label": sid,
            "x": float(rec["x"]),
            "y": float(rec["y"]),
            "z": float(rec["z"]),
            "dendrite_id": str(rec.get("dendrite_id") or ""),
            "features": {},
        }
