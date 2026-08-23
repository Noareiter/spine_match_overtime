"""Merge per-FOV annotator outputs into respan/_annotator/Results final/."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import (
    animal_config,
    animal_layout,
    dendrite_link_store,
    ignored_spine_store,
    spine_catalog_store,
    spine_lineage_store,
    spine_qc_store,
    spine_qc_tag_store,
)

RESULTS_FINAL_DIRNAME = "Results final"
MANIFEST_FILENAME = "build_manifest.json"

_MERGE_CSV_NAMES = (
    "spine_disposition.csv",
    "spine_catalog.csv",
)

# Wide-format files with one column block per timepoint. These need the
# canonical animal-level timepoint_order for their header, not a naive
# first-seen-across-FOVs union (see _merge_wide_csv_by_timepoint) --
# otherwise a FOV missing a middle timepoint pushes another FOV's columns
# for that timepoint to the end of the header, out of chronological order.
_REGISTRY_FIXED_COLUMNS = (
    "animal_id",
    "fov",
    "lineage_id",
    "lineage_key",
    "pre_spine_id",
    "anchor_timepoint",
    "first_seen_tp",
    "last_seen_tp",
)
_REGISTRY_TP_COLUMN_TEMPLATES = (
    "id_{tp}",
    "local_id_{tp}",
    "status_{tp}",
    "fate_{tp}",
    "artifact_mode_{tp}",
    "{tp}_x",
    "{tp}_y",
    "{tp}_z",
)
_DENDRITE_LINKS_FIXED_COLUMNS = ("animal_id", "fov", "link_id")
_DENDRITE_LINKS_TP_COLUMN_TEMPLATES = ("dendrite_id_{tp}",)

IGNORED_MERGED_FILENAME = "ignored_spines.csv"
IGNORED_COLUMNS = (
    "animal_id",
    "fov",
    "timepoint",
    "global_spine_id",
    "local_spine_id",
    "lineage_key",
    "source",
)

_COPY_JSON_NAMES = (
    "lineage_decisions.json",
    "duplicate_resolutions.json",
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


def _merge_wide_csv_by_timepoint(
    labeled_paths: List[Tuple[int, Path]],
    fixed_columns: Tuple[str, ...],
    tp_column_templates: Tuple[str, ...],
    timepoint_order: List[str],
    out_path: Path,
) -> int:
    """Merge wide per-timepoint-column CSVs using the full canonical
    animal-level timepoint_order for the header, in chronological order --
    every timepoint gets its column block even if a given FOV never touched
    it. _merge_csv_paths's naive first-seen-across-FOVs union would instead
    push a later FOV's columns for a timepoint an earlier FOV lacks to the
    end of the header, out of order.
    """
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
    header = list(fixed_columns)
    for tmpl in tp_column_templates:
        header += [tmpl.format(tp=tp) for tp in timepoint_order]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in header})
    return len(all_rows)


def _ignored_spine_rows_for_fov(respan: Path, fov: int, animal_id: str) -> List[dict]:
    """Union every 'ignore' source for one FOV into flat rows.

    Three independent sources, each a genuinely different action:
    - oof: spine falls inside a drawn out-of-frame region (ignored_spines.json)
    - qc_tag: manually tagged ignore via Timepoint mode / bulk Tab->I
      (spine_qc_tags.json)
    - lineage_fate: a timepoint within a lineage was marked fate="ignore"
      (lineage_decisions.json) -- there is no detection at that timepoint by
      definition (the match must be cleared before fate can be set), so
      global_spine_id/local_spine_id are left empty and lineage_key
      identifies which lineage's timepoint-gap this is instead.
    """
    catalog = spine_catalog_store.load_catalog(respan, fov)

    def _local_id(gid: str) -> str:
        if not catalog:
            return ""
        row = catalog.by_global.get(str(gid))
        return str(row.get("local_spine_id") or "") if row else ""

    rows: List[dict] = []

    by_tp_oof = ignored_spine_store.load_ignored(respan, fov)
    for tp, ids in sorted(by_tp_oof.items()):
        for gid in sorted(ids):
            rows.append({
                "animal_id": animal_id,
                "fov": str(fov),
                "timepoint": tp,
                "global_spine_id": gid,
                "local_spine_id": _local_id(gid),
                "lineage_key": "",
                "source": "oof",
            })

    qc_tags = spine_qc_tag_store.load_tags(respan, fov)
    for tp, tagmap in sorted(qc_tags.items()):
        for gid, tag in sorted(tagmap.items()):
            if tag != "ignore" or not str(gid).strip():
                continue
            rows.append({
                "animal_id": animal_id,
                "fov": str(fov),
                "timepoint": tp,
                "global_spine_id": gid,
                "local_spine_id": _local_id(gid),
                "lineage_key": "",
                "source": "qc_tag",
            })

    decisions = spine_lineage_store.load_decisions(respan, fov)
    for lin in decisions.get("lineages") or []:
        lineage_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()
        per_tp = lin.get("per_tp") or {}
        for tp, td in sorted(per_tp.items()):
            if str((td or {}).get("fate") or "").strip().lower() != "ignore":
                continue
            rows.append({
                "animal_id": animal_id,
                "fov": str(fov),
                "timepoint": tp,
                "global_spine_id": "",
                "local_spine_id": "",
                "lineage_key": lineage_key,
                "source": "lineage_fate",
            })

    return rows


def _merge_ignored_spines(
    target_fovs: List[int],
    respan: Path,
    animal_id: str,
    out_path: Path,
) -> int:
    all_rows: List[dict] = []
    for fov in target_fovs:
        all_rows.extend(_ignored_spine_rows_for_fov(respan, fov, animal_id))
    if not all_rows:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(IGNORED_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in IGNORED_COLUMNS})
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

    timepoint_order = list(cfg.timepoint_order)
    for name, fixed_cols, tp_templates in (
        ("spine_registry_wide.csv", _REGISTRY_FIXED_COLUMNS, _REGISTRY_TP_COLUMN_TEMPLATES),
        ("dendrite_links_wide.csv", _DENDRITE_LINKS_FIXED_COLUMNS, _DENDRITE_LINKS_TP_COLUMN_TEMPLATES),
    ):
        labeled = [
            (fov, dendrite_link_store.annotator_meta_dir(respan, fov) / name)
            for fov in target_fovs
        ]
        count = _merge_wide_csv_by_timepoint(labeled, fixed_cols, tp_templates, timepoint_order, out_dir / name)
        if count:
            merged_counts[name] = count

    ignored_count = _merge_ignored_spines(
        target_fovs, respan, cfg.animal_id, out_dir / IGNORED_MERGED_FILENAME
    )
    if ignored_count:
        merged_counts[IGNORED_MERGED_FILENAME] = ignored_count

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
