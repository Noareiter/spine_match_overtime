"""Discover spine CSVs and TIFF stacks from the respan folder tree (location, not filename)."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import animal_config, dendrite_link_store

INVENTORY_FILENAME = "fov_inventory.csv"
INVENTORY_COLUMNS = [
    "row_type",
    "animal_id",
    "fov",
    "timepoint",
    "folder",
    "csv_path",
    "tiff_path",
    "spine_count",
    "dendrite_ids",
    "missing",
    "active",
    "t1_timepoint",
    "t2_timepoint",
    "pair_label",
]

_DETECTED_SPINES_RE = re.compile(r"^fov(\d+).*detected_spines.*\.csv$", re.IGNORECASE)
_TIFF_FOV_RE = re.compile(r"^fov(\d+)(?:[_\-.].*)?\.(?:tif|tiff)$", re.IGNORECASE)
_SKIP_DIR_NAMES = {"tables", "results", "old_inffered", "__pycache__"}


class _SimpleInputProvenance:
    """Fallback layout discovery when input_provenance.py is unavailable."""

    @staticmethod
    def list_respan_timepoint_dirs(respan_root: Path) -> List[Path]:
        """List all direct subdirectories under respan_root (excluding special dirs)."""
        dirs = []
        for child in respan_root.iterdir():
            if not child.is_dir() or child.name.startswith("_"):
                continue
            if child.name.lower() in _SKIP_DIR_NAMES:
                continue
            dirs.append(child)
        return sorted(dirs, key=lambda p: p.name)

    @staticmethod
    def normalize_timepoint_key(label: str) -> str:
        """Normalize a timepoint label for comparison."""
        return label.lower().strip()

    @staticmethod
    def find_spine_csv(respan_root: Path, timepoint: str, fov: int) -> Path:
        """Find the detected spines CSV for a given timepoint and FOV."""
        tp_dir = respan_root / timepoint
        if not tp_dir.is_dir():
            raise FileNotFoundError(f"Timepoint dir not found: {tp_dir}")
        tables = tp_dir / "Tables"
        if tables.is_dir():
            for csv in tables.glob(f"fov{fov}*detected_spines*.csv"):
                return csv
        raise FileNotFoundError(f"No spine CSV for fov{fov} under {tp_dir}")

    @staticmethod
    def resolve_timepoint_dir(respan_root: Path, timepoint: str) -> Path:
        """Resolve a timepoint directory by name."""
        tp_dir = respan_root / timepoint
        if not tp_dir.is_dir():
            raise FileNotFoundError(f"Timepoint directory not found: {tp_dir}")
        return tp_dir


def _import_input_provenance():
    """Try to import input_provenance.py, fall back to simple implementation if not available."""
    from .project_paths import ASSUME_T1_T2_DIR

    prov_path = ASSUME_T1_T2_DIR / "input_provenance.py"
    if prov_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("input_provenance_layout", prov_path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("input_provenance_layout", mod)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            pass
    # Fallback: use simple implementation
    return _SimpleInputProvenance()


def discover_fovs(respan_root: Path) -> List[int]:
    fovs: set[int] = set()
    for child in respan_root.iterdir():
        if not child.is_dir() or child.name.lower() in _SKIP_DIR_NAMES:
            continue
        tables = child / "Tables"
        if not tables.is_dir():
            continue
        for csv in tables.glob("*.csv"):
            m = _DETECTED_SPINES_RE.match(csv.name)
            if m:
                fovs.add(int(m.group(1)))
    return sorted(fovs)


def discover_timepoint_dirs(respan_root: Path, *, preferred_order: Optional[List[str]] = None) -> List[Path]:
    prov = _import_input_provenance()
    found = prov.list_respan_timepoint_dirs(respan_root)
    if not preferred_order:
        return found
    by_key = {prov.normalize_timepoint_key(p.name): p for p in found}
    ordered: List[Path] = []
    seen: set[str] = set()
    for label in preferred_order:
        key = prov.normalize_timepoint_key(label)
        if key in by_key:
            ordered.append(by_key[key])
            seen.add(by_key[key].name.lower())
    for p in found:
        if p.name.lower() not in seen:
            ordered.append(p)
    return ordered


def find_spine_csv(respan_root: Path, timepoint: str, fov: int) -> Path:
    return _import_input_provenance().find_spine_csv(respan_root, timepoint, fov)


def find_tiff_for_fov(timepoint_dir: Path, fov: int) -> Optional[Path]:
    """Find FOV stack under a timepoint folder; shallowest match wins; skips Tables/."""
    if not timepoint_dir.is_dir():
        return None
    candidates: List[Tuple[int, int, Path]] = []
    for p in timepoint_dir.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = {part.lower() for part in p.relative_to(timepoint_dir).parts[:-1]}
        if rel_parts.intersection(_SKIP_DIR_NAMES):
            continue
        m = _TIFF_FOV_RE.match(p.name)
        if not m or int(m.group(1)) != fov:
            continue
        depth = len(p.relative_to(timepoint_dir).parts)
        candidates.append((depth, len(p.name), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2].resolve()


@dataclass
class TimepointFiles:
    name: str
    folder: str
    csv_path: Optional[str] = None
    tiff_path: Optional[str] = None
    spine_count: int = 0
    dendrite_ids: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class FovInventory:
    animal_id: str
    fov: int
    respan_root: str
    timepoints: List[TimepointFiles]
    workflow_pairs: List[Dict[str, str]] = field(default_factory=list)


def _dendrite_ids_from_csv(csv_path: Path) -> Tuple[int, List[str]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "spine_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"spine_id": "id"})
    n = len(df)
    if "dendrite_id" not in df.columns:
        return n, []
    ids = sorted({str(x) for x in df["dendrite_id"].dropna().astype(str) if str(x).strip() not in {"", "nan"}}, key=lambda s: (0, float(s)) if s.replace(".", "", 1).isdigit() else (1, s))
    return n, ids


def discover_available_timepoints(respan_root: Path, fov: int) -> List[str]:
    """Timepoint folder names that have a detection CSV for this FOV."""
    cfg = animal_config.load_config()
    tp_dirs = discover_timepoint_dirs(respan_root, preferred_order=cfg.timepoint_order)
    available: List[str] = []
    for tp_dir in tp_dirs:
        try:
            find_spine_csv(respan_root, tp_dir.name, fov)
            available.append(tp_dir.name)
        except FileNotFoundError:
            continue
    return available


def build_fov_inventory(
    respan_root: Path,
    fov: int,
    *,
    animal_id: str = "",
    selected_timepoints: Optional[List[str]] = None,
) -> FovInventory:
    cfg = animal_config.load_config()
    tp_dirs = discover_timepoint_dirs(respan_root, preferred_order=cfg.timepoint_order)
    all_rows: List[TimepointFiles] = []
    available_names: List[str] = []

    for tp_dir in tp_dirs:
        name = tp_dir.name
        row = TimepointFiles(name=name, folder=str(tp_dir))
        try:
            csv_path = find_spine_csv(respan_root, name, fov)
            row.csv_path = str(csv_path)
            row.spine_count, row.dendrite_ids = _dendrite_ids_from_csv(csv_path)
            available_names.append(name)
        except FileNotFoundError:
            row.missing.append("csv")
        tiff = find_tiff_for_fov(tp_dir, fov)
        if tiff is not None:
            row.tiff_path = str(tiff)
        else:
            row.missing.append("tiff")
        all_rows.append(row)

    active_names = animal_config.resolve_active_timepoints(
        cfg,
        available_names,
        requested=selected_timepoints,
    )
    active_set = set(active_names)
    rows = [r for r in all_rows if r.name in active_set]

    pairs: List[Dict[str, str]] = []
    for t1, t2 in animal_config.workflow_pairs_for_timepoints(active_names):
        pairs.append({"t1_timepoint": t1, "t2_timepoint": t2, "label": f"{t1} → {t2}"})

    inv = FovInventory(
        animal_id=animal_id or cfg.animal_id,
        fov=fov,
        respan_root=str(respan_root),
        timepoints=rows,
        workflow_pairs=pairs,
    )
    write_fov_inventory_csv(inv, all_rows=all_rows, active_set=active_set)
    return inv


def inventory_path(respan_root: Path, fov: int) -> Path:
    return dendrite_link_store.annotator_meta_dir(respan_root, fov) / INVENTORY_FILENAME


def write_fov_inventory_csv(
    inv: FovInventory,
    *,
    all_rows: List[TimepointFiles],
    active_set: set[str],
) -> Path:
    """Document every discovered timepoint (not just the active subset) plus
    the resolved workflow pairs, so the on-disk record matches what the app
    actually saw at load time, regardless of which timepoints are selected.
    """
    path = inventory_path(Path(inv.respan_root), inv.fov)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "row_type": "timepoint",
                    "animal_id": inv.animal_id,
                    "fov": inv.fov,
                    "timepoint": row.name,
                    "folder": row.folder,
                    "csv_path": row.csv_path or "",
                    "tiff_path": row.tiff_path or "",
                    "spine_count": row.spine_count,
                    "dendrite_ids": ";".join(row.dendrite_ids),
                    "missing": ";".join(row.missing),
                    "active": row.name in active_set,
                }
            )
        for pair in inv.workflow_pairs:
            writer.writerow(
                {
                    "row_type": "workflow_pair",
                    "animal_id": inv.animal_id,
                    "fov": inv.fov,
                    "t1_timepoint": pair.get("t1_timepoint", ""),
                    "t2_timepoint": pair.get("t2_timepoint", ""),
                    "pair_label": pair.get("label", ""),
                }
            )
    return path


def list_timepoint_catalog(
    respan_root: Path,
    fov: int,
    *,
    saved_selection: Optional[List[str]] = None,
) -> List[dict]:
    """All known timepoints with availability flags (for UI checkboxes)."""
    cfg = animal_config.load_config()
    tp_dirs = discover_timepoint_dirs(respan_root, preferred_order=cfg.timepoint_order)
    available: List[str] = []
    by_name: Dict[str, TimepointFiles] = {}
    for tp_dir in tp_dirs:
        name = tp_dir.name
        row = TimepointFiles(name=name, folder=str(tp_dir))
        try:
            csv_path = find_spine_csv(respan_root, name, fov)
            row.csv_path = str(csv_path)
            row.spine_count, row.dendrite_ids = _dendrite_ids_from_csv(csv_path)
            available.append(name)
        except FileNotFoundError:
            row.missing.append("csv")
        tiff = find_tiff_for_fov(tp_dir, fov)
        if tiff is not None:
            row.tiff_path = str(tiff)
        else:
            row.missing.append("tiff")
        by_name[name] = row

    active = animal_config.resolve_active_timepoints(
        cfg,
        available,
        saved=saved_selection,
    )
    active_set = set(active)
    catalog: List[dict] = []
    for name in cfg.timepoint_order + [n for n in by_name if n not in cfg.timepoint_order]:
        if name not in by_name:
            continue
        row = by_name[name]
        catalog.append(
            {
                "name": name,
                "has_csv": bool(row.csv_path),
                "has_tiff": bool(row.tiff_path),
                "spine_count": row.spine_count,
                "selected": name in active_set,
                "missing": list(row.missing),
            }
        )
    return catalog


def resolve_pair_files(respan_root: Path, fov: int, t1_timepoint: str, t2_timepoint: str) -> Dict[str, str]:
    t1_csv = find_spine_csv(respan_root, t1_timepoint, fov)
    t2_csv = find_spine_csv(respan_root, t2_timepoint, fov)
    t1_dir = _import_input_provenance().resolve_timepoint_dir(respan_root, t1_timepoint)
    t2_dir = _import_input_provenance().resolve_timepoint_dir(respan_root, t2_timepoint)
    t1_tiff = find_tiff_for_fov(t1_dir, fov)
    t2_tiff = find_tiff_for_fov(t2_dir, fov)
    if t1_tiff is None:
        raise FileNotFoundError(f"No TIFF for FOV {fov} under {t1_dir}")
    if t2_tiff is None:
        raise FileNotFoundError(f"No TIFF for FOV {fov} under {t2_dir}")
    return {
        "t1_tiff_path": str(t1_tiff),
        "t2_tiff_path": str(t2_tiff),
        "t1_csv_path": str(t1_csv),
        "t2_csv_path": str(t2_csv),
        "t1_timepoint": t1_dir.name,
        "t2_timepoint": t2_dir.name,
        "fov": str(fov),
    }
