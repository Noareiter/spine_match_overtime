"""Merge per-FOV annotator outputs into respan/_annotator/Results final/."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import animal_config, animal_layout, dendrite_link_store, spine_qc_store

RESULTS_FINAL_DIRNAME = "Results final"
MANIFEST_FILENAME = "build_manifest.json"

_MERGE_CSV_NAMES = (
    "spine_registry_wide.csv",
    "spine_disposition.csv",
    "spine_catalog.csv",
    "dendrite_links_wide.csv",
)

_COPY_JSON_NAMES = (
    "lineage_decisions.json",
    "duplicate_resolutions.json",
    "ignored_spines.json",
    "oof_segments.json",
    "dendrite_links.json",
    "spine_review_progress.json",
    "timepoint_selection.json",
)


def results_final_dir(respan: Path) -> Path:
    return respan / "_annotator" / RESULTS_FINAL_DIRNAME


def resolve_target_fovs(respan: Path, cfg: animal_config.AnimalConfig) -> List[int]:
    if cfg.fovs:
        return sorted(cfg.fovs)
    annotator = respan / "_annotator"
    found: List[int] = []
    if annotator.is_dir():
        for child in annotator.iterdir():
            if not child.is_dir() or not child.name.lower().startswith("fov"):
                continue
            suffix = child.name[3:]
            if suffix.isdigit():
                found.append(int(suffix))
    if found:
        return sorted(found)
    return animal_layout.discover_fovs(respan)


def _fov_timepoints(respan: Path, fov: int, cfg: animal_config.AnimalConfig) -> List[str]:
    available = animal_layout.discover_available_timepoints(respan, fov)
    if not available:
        return []
    from . import timepoint_selection

    saved = timepoint_selection.load(respan, fov)
    return animal_config.resolve_active_timepoints(
        cfg,
        available,
        saved=saved or None,
    )


def fov_review_status(
    respan: Path,
    fov: int,
    cfg: animal_config.AnimalConfig,
) -> dict:
    meta = dendrite_link_store.annotator_meta_dir(respan, fov)
    tps = _fov_timepoints(respan, fov, cfg)
    if not tps:
        return {
            "fov": fov,
            "complete": False,
            "reason": "no_timepoints",
            "unreviewed": 0,
            "duplicate_unresolved": 0,
            "catalog_total": 0,
        }
    _, summary = spine_qc_store.coverage_and_unreviewed(
        respan,
        fov,
        tps,
        animal_id=cfg.animal_id,
    )
    registry = meta / "spine_registry_wide.csv"
    unreviewed = int(summary.get("unreviewed", 0) or 0)
    duplicate_unresolved = int(summary.get("duplicate_unresolved", 0) or 0)
    catalog_total = int(summary.get("catalog_total", 0) or 0)
    complete = (
        unreviewed == 0
        and duplicate_unresolved == 0
        and catalog_total > 0
        and registry.is_file()
    )
    return {
        "fov": fov,
        "complete": complete,
        "reason": "" if complete else "pending_review",
        "unreviewed": unreviewed,
        "duplicate_unresolved": duplicate_unresolved,
        "catalog_total": catalog_total,
        "tagged": int(summary.get("tagged", 0) or 0),
        "registry_path": str(registry) if registry.is_file() else "",
    }


def all_fovs_status(
    respan: Path,
    cfg: animal_config.AnimalConfig,
    *,
    fovs: Optional[Sequence[int]] = None,
) -> Tuple[List[dict], bool]:
    target = list(fovs if fovs is not None else resolve_target_fovs(respan, cfg))
    statuses = [fov_review_status(respan, fov, cfg) for fov in target]
    all_complete = bool(target) and all(s["complete"] for s in statuses)
    return statuses, all_complete


def _union_csv_header(rows: List[dict]) -> List[str]:
    header: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                header.append(key)
    return header


def _merge_csv_paths(
    labeled_paths: List[Tuple[int, Path]],
    out_path: Path,
) -> int:
    all_rows: List[dict] = []
    for fov, path in labeled_paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                item = dict(row)
                if not str(item.get("fov", "")).strip():
                    item["fov"] = str(fov)
                all_rows.append(item)
    if not all_rows:
        return 0
    header = _union_csv_header(all_rows)
    if "fov" in header:
        header = ["fov"] + [h for h in header if h != "fov"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return len(all_rows)


def _copy_json_snapshot(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def build_results_final(
    respan: Path,
    cfg: animal_config.AnimalConfig,
    *,
    fovs: Optional[Sequence[int]] = None,
) -> dict:
    target_fovs = list(fovs if fovs is not None else resolve_target_fovs(respan, cfg))
    statuses, all_complete = all_fovs_status(respan, cfg, fovs=target_fovs)
    if not all_complete:
        pending = [s["fov"] for s in statuses if not s["complete"]]
        raise ValueError(
            f"Cannot build Results final — FOV(s) still pending review: {', '.join(map(str, pending))}"
        )

    out_dir = results_final_dir(respan)
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_counts: Dict[str, int] = {}
    copied_files: List[str] = []

    for name in _MERGE_CSV_NAMES:
        labeled = [
            (fov, dendrite_link_store.annotator_meta_dir(respan, fov) / name)
            for fov in target_fovs
        ]
        count = _merge_csv_paths(labeled, out_dir / name)
        if count:
            merged_counts[name] = count

    for fov in target_fovs:
        meta = dendrite_link_store.annotator_meta_dir(respan, fov)
        for name in _COPY_JSON_NAMES:
            src = meta / name
            dest = out_dir / f"fov{fov}_{name}"
            if _copy_json_snapshot(src, dest):
                copied_files.append(dest.name)

    manifest = {
        "animal_id": cfg.animal_id,
        "respan_root": str(respan),
        "fovs": target_fovs,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "merged_csv_row_counts": merged_counts,
        "copied_json_files": sorted(copied_files),
        "fov_status": statuses,
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "results_final_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "merged_csv_row_counts": merged_counts,
        "fov_count": len(target_fovs),
        "all_fovs_complete": True,
        "message": (
            f"Results final built for {len(target_fovs)} FOV(s) — "
            f"{merged_counts.get('spine_registry_wide.csv', 0)} lineage row(s) merged."
        ),
    }


def maybe_build_results_final(
    respan: Path,
    cfg: Optional[animal_config.AnimalConfig] = None,
) -> Optional[dict]:
    """Build merged Results final when every configured FOV has finished review."""
    cfg = cfg or animal_config.load_config()
    statuses, all_complete = all_fovs_status(respan, cfg)
    if not all_complete:
        return {
            "all_fovs_complete": False,
            "fov_status": statuses,
            "pending_fovs": [s["fov"] for s in statuses if not s["complete"]],
        }
    return build_results_final(respan, cfg)
