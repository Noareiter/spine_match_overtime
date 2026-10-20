"""spine_observations_long.csv: one row per lineage x timepoint, with morphology.

The wide registry holds ids, status and coordinates. This long file carries
every RESPAN measure (volumes, lengths, head / neck, intensities, distance to
soma) for the detection a lineage holds at each timepoint, ready for analysis
in Python / R. Measures are empty where the lineage holds no detection
(absent / lost) and for manually added spines (not in any RESPAN CSV).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from . import baseline_adapter, spine_catalog_store

OBSERVATIONS_FILENAME = "spine_observations_long.csv"

FIXED_COLUMNS = [
    "animal_id",
    "fov",
    "lineage_id",
    "lineage_key",
    "timepoint",
    "timepoint_index",
    "status",
    "global_spine_id",
    "local_spine_id",
    "dendrite_id",
    "x",
    "y",
    "z",
]

# Columns already in FIXED_COLUMNS (or the id column) — not repeated as measures.
_RESPAN_SKIP = {"spine_id", "id", "x", "y", "z", "dendrite_id"}

_OBSERVED = ("matched", "new", "manual")

# path -> (mtime, measure column order, {local id: row})
_CSV_CACHE: Dict[str, Tuple[float, List[str], Dict[str, dict]]] = {}


def _respan_rows(csv_path: str) -> Tuple[List[str], Dict[str, dict]]:
    p = Path(csv_path)
    if not p.is_file():
        return [], {}
    mtime = p.stat().st_mtime
    cached = _CSV_CACHE.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    df = pd.read_csv(p)
    id_col = "spine_id" if "spine_id" in df.columns else "id"
    measures = [c for c in df.columns if c not in _RESPAN_SKIP]
    by_id: Dict[str, dict] = {}
    for rec in df.to_dict("records"):
        local = baseline_adapter._hybrid.canonical_id(rec.get(id_col))
        if local:
            by_id[local] = rec
    _CSV_CACHE[str(p)] = (mtime, measures, by_id)
    return measures, by_id


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def rebuild_observations_long(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    lineages: List[dict],
    timepoint_names: List[str],
    out_dir: Path,
    status_for_tp,
) -> Path:
    """Write out_dir/spine_observations_long.csv for the given (finalized) lineages."""
    catalog = spine_catalog_store.load_catalog(respan, fov)
    measure_cols: List[str] = []
    rows: List[dict] = []
    for lin in lineages:
        lineage_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "")
        per_tp = dict(lin.get("per_tp") or {})
        for idx, tp in enumerate(timepoint_names):
            if tp not in per_tp:
                continue
            td = per_tp.get(tp) or {}
            status = status_for_tp(td)
            row = {
                "animal_id": animal_id,
                "fov": str(fov),
                "lineage_id": f"L_{lineage_key.replace(' ', '_')}",
                "lineage_key": lineage_key,
                "timepoint": tp,
                "timepoint_index": str(idx),
                "status": status,
            }
            gid = str(td.get("spine_id") or "").strip()
            if status in _OBSERVED and gid:
                cat_row = catalog.by_global.get(gid) if catalog else None
                local = str(cat_row.get("local_spine_id") or "") if cat_row else ""
                row.update({
                    "global_spine_id": gid,
                    "local_spine_id": local,
                    "dendrite_id": str((cat_row or {}).get("dendrite_id") or td.get("dendrite_id") or ""),
                    "x": _fmt(td.get("x")),
                    "y": _fmt(td.get("y")),
                    "z": _fmt(td.get("z")),
                })
                if cat_row and local:
                    measures, by_id = _respan_rows(str(cat_row.get("source_csv") or ""))
                    for col in measures:
                        if col not in measure_cols:
                            measure_cols.append(col)
                    rec = by_id.get(local) or {}
                    for col in measures:
                        row[col] = _fmt(rec.get(col))
            rows.append(row)

    header = FIXED_COLUMNS + measure_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / OBSERVATIONS_FILENAME
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return out
