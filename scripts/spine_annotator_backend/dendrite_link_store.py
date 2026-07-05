"""Persist and load cross-timepoint dendrite links under respan/_annotator/."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def annotator_meta_dir(respan: Path, fov: int) -> Path:
    return respan / "_annotator" / f"fov{fov}"


def link_paths(respan: Path, fov: int) -> Dict[str, Path]:
    base = annotator_meta_dir(respan, fov)
    return {
        "dir": base,
        "json": base / "dendrite_links.json",
        "wide": base / "dendrite_links_wide.csv",
        "progress": base / "dendrite_link_progress.json",
    }


def _parse_id_cell(cell: str) -> List[str]:
    if not cell or not str(cell).strip():
        return []
    raw = str(cell).strip().strip('"')
    return [x.strip() for x in raw.split(",") if x.strip()]


def wide_csv_to_links(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return rows
        tp_cols = [c for c in reader.fieldnames if c.startswith("dendrite_id_")]
        for row in reader:
            members: Dict[str, List[str]] = {}
            for col in tp_cols:
                tp = col.replace("dendrite_id_", "", 1)
                ids = _parse_id_cell(row.get(col, ""))
                if ids:
                    members[tp] = ids
            if not members:
                continue
            rows.append(
                {
                    "link_id": str(row.get("link_id", "") or f"link_{len(rows)+1}"),
                    "members": members,
                    "notes": "",
                }
            )
    return rows


def load_links(respan: Path, fov: int) -> List[dict]:
    paths = link_paths(respan, fov)
    if paths["json"].is_file():
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        return list(data.get("links", []))
    if paths["wide"].is_file():
        return wide_csv_to_links(paths["wide"])
    return []


def save_links(
    respan: Path,
    fov: int,
    *,
    animal_id: str,
    timepoint_names: List[str],
    links: List[dict],
    files: Optional[dict] = None,
) -> Dict[str, str]:
    paths = link_paths(respan, fov)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "timepoint_names": timepoint_names,
        "files": files or {},
        "links": links,
    }
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = ["animal_id", "fov", "link_id"] + [f"dendrite_id_{n}" for n in timepoint_names]
    lines = [",".join(header)]
    for link in links:
        row = [animal_id, str(fov), str(link.get("link_id", ""))]
        for name in timepoint_names:
            ids = link.get("members", {}).get(name, [])
            cell = ",".join(str(i) for i in ids)
            row.append(f'"{cell}"' if "," in cell else cell)
        lines.append(",".join(row))
    paths["wide"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(paths["json"]), "wide": str(paths["wide"])}


def to_pairwise_links(links: List[dict], t1_timepoint: str, t2_timepoint: str) -> List[dict]:
    """Convert cross-TP links to pairwise session.dendrite_links rows."""
    out: List[dict] = []
    for link in links:
        members = link.get("members") or {}
        t1_ids = [str(x) for x in members.get(t1_timepoint, []) if str(x).strip()]
        t2_ids = [str(x) for x in members.get(t2_timepoint, []) if str(x).strip()]
        if not t1_ids or not t2_ids:
            continue
        out.append(
            {
                "link_id": str(link.get("link_id", f"link_{len(out)+1}")),
                "t1_dendrite_ids": t1_ids,
                "t2_dendrite_ids": t2_ids,
                "notes": str(link.get("notes", "")),
            }
        )
    return out


def load_progress(respan: Path, fov: int) -> dict:
    path = link_paths(respan, fov)["progress"]
    if not path.is_file():
        return {"visited_link_ids": [], "last_link_id": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(x) for x in data.get("visited_link_ids", []) if str(x).strip()]
    return {
        "visited_link_ids": ids,
        "last_link_id": str(data.get("last_link_id", "") or ""),
        "updated_at": str(data.get("updated_at", "") or ""),
    }


def save_progress(respan: Path, fov: int, progress: dict) -> Path:
    paths = link_paths(respan, fov)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = {
        "visited_link_ids": list(progress.get("visited_link_ids") or []),
        "last_link_id": str(progress.get("last_link_id", "") or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["progress"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths["progress"]


def mark_link_visited(respan: Path, fov: int, link_id: str) -> dict:
    prog = load_progress(respan, fov)
    ids = list(prog.get("visited_link_ids") or [])
    lid = str(link_id).strip()
    if lid and lid not in ids:
        ids.append(lid)
    prog["visited_link_ids"] = ids
    prog["last_link_id"] = lid
    save_progress(respan, fov, prog)
    return prog
