#!/usr/bin/env python3
"""
Infer missing pairwise spine classifications from manual annotator exports.

Reads every comparison folder that has a **manual** export (name ends with
``_spine_annotator_export``, not ``inferred`` / ``synthetic`` / etc.). Builds
lineages with union-find on ``matched.csv`` links, then writes **any** missing
chronological pair — including baseline pairs such as pre-droplet vs end-droplet
when pre-mid and mid-end (etc.) are already classified.

Re-run safe: after you validate an inferred pair (manual export appears) and
archive the inferred folder, the next run uses that export too — more spines
get classified. Conflicting IDs and all prior ``ignored`` flags → ``unresolved_manual_review.csv``.

Legacy baseline-only bridge (``infer_pair_via_baseline``) is kept for reference;
the main path uses full lineage inference.

Input layout (case-insensitive RESPAN segment):
  .../IMAGING/<ANIMAL_ID>/respan/results/fovN/<comparison>/<timestamp>_spine_annotator_export/

Output:
  .../fovN/<T_A - T_B>/latest_inferred_spine_annotator_export/
      input_files/   (TIFF stacks copied from baseline comparison logs)
      matched.csv, new.csv, lost.csv, unresolved_manual_review.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from input_provenance import find_spine_csv, stage_session_inputs, validate_or_raise

# --- Configuration (timepoint folder names on disk) ---
BASELINE_TP = "pre-droplet"

TP_ORDER: List[str] = [
    "pre-droplet",
    "mid-droplet",
    "end-droplet",
    "end-lever",
    "return to droplet",
]

TP_ORDER_INDEX: Dict[str, int] = {tp: i for i, tp in enumerate(TP_ORDER)}

NON_BASELINE_TPS: Tuple[str, ...] = tuple(tp for tp in TP_ORDER if tp != BASELINE_TP)

# Comparison folder names for baseline-linked exports (must match annotator / Step 3).
BASELINE_COMPARISON_MAP: Dict[str, str] = {
    "mid-droplet": "pre-mid droplet",
    "end-droplet": "pre-end droplet",
    "end-lever": "pre droplet-end lever",
    "return to droplet": "pre droplet - return to droplet",
}

COMPARISON_BY_PAIR: Dict[Tuple[str, str], str] = {
    ("end-droplet", "return to droplet"): "end droplet - return to droplet",
    ("end-droplet", "end-lever"): "end droplet - end lever",
}

COMPARISON_MAP: Dict[str, Tuple[str, str]] = {
    # Canonical GP04 comparison folder names (10 chronological pairs).
    "pre-mid droplet": ("pre-droplet", "mid-droplet"),
    "pre-droplet - end-droplet": ("pre-droplet", "end-droplet"),
    "pre-droplet - end-lever": ("pre-droplet", "end-lever"),
    "pre-droplet - return to droplet": ("pre-droplet", "return to droplet"),
    "mid-droplet - end-droplet": ("mid-droplet", "end-droplet"),
    "mid-droplet - end-lever": ("mid-droplet", "end-lever"),
    "mid-droplet - return to droplet": ("mid-droplet", "return to droplet"),
    "end-droplet - end-lever": ("end-droplet", "end-lever"),
    "end-droplet - return to droplet": ("end-droplet", "return to droplet"),
    "end-lever - return to droplet": ("end-lever", "return to droplet"),
    # Legacy aliases (older export folder spellings).
    "pre-end droplet": ("pre-droplet", "end-droplet"),
    "pre droplet-end lever": ("pre-droplet", "end-lever"),
    "pre droplet - return to droplet": ("pre-droplet", "return to droplet"),
    "end droplet - return to droplet": ("end-droplet", "return to droplet"),
    "end droplet - end lever": ("end-droplet", "end-lever"),
}

DERIVED_EXPORT_MARKERS = ("inferred", "synthetic", "registry_derived", "transitive")
INFERENCE_SOURCE_LINEAGE = "lineage_union_find"

STATUS_MATCHED = "matched"
STATUS_NEW = "new"
STATUS_LOST = "lost"
STATUS_IGNORED = "ignored"
STATUS_NOT_IN_FOCUS = "not_in_focus"
STATUS_ARTIFACT = "artifact"
STATUS_ABSENT = "absent"
UNCERTAINTY_STATUSES: Set[str] = {STATUS_IGNORED, STATUS_NOT_IN_FOCUS}
INFERRED_EXPORT_SUFFIX = "_inferred_spine_annotator_export"
# Single inferred folder per comparison (replaced on each run).
LATEST_INFERRED_EXPORT_DIRNAME = f"latest{INFERRED_EXPORT_SUFFIX}"
LOG_CANDIDATES = ("matching_activity_log.txt", "matching_activity.log")


@dataclass
class BaselineExportData:
    """One Baseline <-> follow-up timepoint annotator export."""

    comparison: str
    follow_up_tp: str
    export_dir: Path
    matched: Dict[str, str] = field(default_factory=dict)  # baseline_id -> follow_up_id
    lost_baseline: Set[str] = field(default_factory=set)
    new_follow_up: Set[str] = field(default_factory=set)
    ignored_baseline: Set[str] = field(default_factory=set)
    ignored_follow_up: Set[str] = field(default_factory=set)
    removed_baseline: Set[str] = field(default_factory=set)
    removed_follow_up: Set[str] = field(default_factory=set)
    t1_tiff: Optional[Path] = None
    t2_tiff: Optional[Path] = None


@dataclass
class InferredPairTables:
    matched: List[dict] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    lost: List[str] = field(default_factory=list)
    removed_t1: List[str] = field(default_factory=list)
    removed_t2: List[str] = field(default_factory=list)
    unresolved: List[dict] = field(default_factory=list)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def add(self, node: Tuple[str, str]) -> None:
        if node not in self.parent:
            self.parent[node] = node

    def find(self, node: Tuple[str, str]) -> Tuple[str, str]:
        self.add(node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, a: Tuple[str, str], b: Tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        out: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        for node in self.parent:
            out[self.find(node)].append(node)
        return dict(out)


@dataclass
class FovLineageState:
    uf: UnionFind = field(default_factory=UnionFind)
    spine_status: Dict[Tuple[str, str], str] = field(default_factory=dict)
    source_exports: List[str] = field(default_factory=list)
    source_comparisons: List[str] = field(default_factory=list)
    tiff_by_tp: Dict[str, Path] = field(default_factory=dict)


def _canonical_spine_id(value: object) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0+", s):
        return str(int(float(s)))
    return s


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    return not s or s.lower() in {"nan", "none", "nat"}


def _read_id_column(path: Path, *preferred_cols: str) -> List[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    for col in preferred_cols:
        if col in df.columns:
            return [_canonical_spine_id(x) for x in df[col].dropna() if not _is_blank(x)]
    if len(df.columns) == 1:
        col = df.columns[0]
        return [_canonical_spine_id(x) for x in df[col].dropna() if not _is_blank(x)]
    return []


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def comparison_folder_name(t1_tp: str, t2_tp: str) -> str:
    if TP_ORDER_INDEX[t1_tp] > TP_ORDER_INDEX[t2_tp]:
        t1_tp, t2_tp = t2_tp, t1_tp
    pair = (t1_tp, t2_tp)
    if pair in COMPARISON_BY_PAIR:
        return COMPARISON_BY_PAIR[pair]
    for folder, tp_pair in COMPARISON_MAP.items():
        if tp_pair == pair:
            return folder
    return f"{t1_tp} - {t2_tp}"


def _is_derived_export_dir(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in DERIVED_EXPORT_MARKERS)


def _is_inferred_export_dir(path: Path) -> bool:
    name = path.name.lower()
    return "inferred" in name and name.endswith("_spine_annotator_export")


def list_inferred_export_dirs(comp_dir: Path) -> List[Path]:
    if not comp_dir.is_dir():
        return []
    return sorted(
        p for p in comp_dir.glob("*spine_annotator_export") if p.is_dir() and _is_inferred_export_dir(p)
    )


def remove_old_inferred_exports(comp_dir: Path, *, dry_run: bool = False) -> List[str]:
    """Delete every inferred export under a comparison so only the new run remains."""
    removed: List[str] = []
    for old_dir in list_inferred_export_dirs(comp_dir):
        removed.append(old_dir.name)
        if dry_run:
            continue
        shutil.rmtree(old_dir)
    return removed


def resolve_results_root(imaging_root: Path, animal_id: str) -> Path:
    """Locate .../IMAGING/<animal>/respan/results (case-insensitive)."""
    animal_dir = imaging_root / animal_id
    if not animal_dir.is_dir():
        raise FileNotFoundError(f"Animal folder not found: {animal_dir}")

    for child in animal_dir.iterdir():
        if child.is_dir() and child.name.lower() == "respan":
            for sub in child.iterdir():
                if sub.is_dir() and sub.name.lower() == "results":
                    return sub.resolve()
    raise FileNotFoundError(
        f"Could not find respan/results under {animal_dir}. "
        "Expected: <imaging_root>/<animal_id>/respan/results"
    )


def discover_fovs(results_root: Path) -> List[int]:
    fovs: List[int] = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"fov(\d+)", d.name, flags=re.IGNORECASE)
        if m:
            fovs.append(int(m.group(1)))
    return fovs


def _parse_export_timestamp(export_dir: Path) -> Optional[datetime]:
    """Best-effort recency from metadata or folder name."""
    meta_path = export_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        for key in ("generated_at_utc", "exported_at_utc", "inference_run_stamp"):
            raw = meta.get(key)
            if _is_blank(raw):
                continue
            text = str(raw).strip()
            if key == "inference_run_stamp":
                try:
                    return datetime.strptime(text, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text)
            except ValueError:
                continue

    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", export_dir.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _export_recency_key(export_dir: Path) -> float:
    """Sort key: higher = more recently updated export."""
    parsed = _parse_export_timestamp(export_dir)
    if parsed is not None:
        return parsed.timestamp()

    mtimes: List[float] = []
    if export_dir.is_dir():
        mtimes.append(export_dir.stat().st_mtime)
    for name in ("matched.csv", "metadata.json", "matching_activity_log.txt", "matching_activity.log"):
        path = export_dir / name
        if path.is_file():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def list_manual_export_dirs(comp_dir: Path) -> List[Path]:
    exports = [
        p for p in comp_dir.glob("*spine_annotator_export") if p.is_dir() and not _is_derived_export_dir(p)
    ]
    return sorted(exports, key=_export_recency_key)


def latest_manual_export_dir(comp_dir: Path) -> Optional[Path]:
    exports = list_manual_export_dirs(comp_dir)
    return exports[-1] if exports else None


def comparison_dir_has_manual_export(comp_dir: Path) -> bool:
    """True if folder is a source for inference (manual export present, not inferred-only)."""
    return latest_manual_export_dir(comp_dir) is not None


def resolve_comparison_tps(comp_name: str, metadata: dict) -> Optional[Tuple[str, str]]:
    if comp_name in COMPARISON_MAP:
        t1, t2 = COMPARISON_MAP[comp_name]
        if TP_ORDER_INDEX[t1] > TP_ORDER_INDEX[t2]:
            t1, t2 = t2, t1
        return t1, t2
    if " - " in comp_name:
        left, right = comp_name.split(" - ", 1)
        left, right = left.strip(), right.strip()
        if left in TP_ORDER_INDEX and right in TP_ORDER_INDEX:
            t1, t2 = sorted((left, right), key=lambda t: TP_ORDER_INDEX[t])
            return t1, t2
    for key in ("t1_timepoint", "t1_tp"):
        t1 = metadata.get(key)
        t2 = metadata.get("t2_timepoint") or metadata.get("t2_tp")
        if t1 in TP_ORDER_INDEX and t2 in TP_ORDER_INDEX:
            a, b = str(t1), str(t2)
            if TP_ORDER_INDEX[a] > TP_ORDER_INDEX[b]:
                a, b = b, a
            return a, b
    t1_tp = _infer_tp_from_path(str(metadata.get("t1_csv_path", "")))
    t2_tp = _infer_tp_from_path(str(metadata.get("t2_csv_path", "")))
    if t1_tp and t2_tp:
        if TP_ORDER_INDEX[t1_tp] > TP_ORDER_INDEX[t2_tp]:
            t1_tp, t2_tp = t2_tp, t1_tp
        return t1_tp, t2_tp
    return None


def _infer_tp_from_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    parts = Path(path_str.replace("\\", "/")).parts
    for tp in TP_ORDER:
        if tp in parts:
            return tp
    return None


def _path_if_file(value: object) -> Optional[Path]:
    if _is_blank(value):
        return None
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_file() else None


def _register_tiff_paths_from_meta(state: FovLineageState, meta: dict, *, t1_tp: str, t2_tp: str) -> None:
    t1_tiff = _path_if_file(meta.get("t1_tiff_path"))
    t2_tiff = _path_if_file(meta.get("t2_tiff_path"))
    if t1_tiff is not None:
        state.tiff_by_tp.setdefault(t1_tp, t1_tiff)
    if t2_tiff is not None:
        state.tiff_by_tp.setdefault(t2_tp, t2_tiff)


def _set_lineage_status(
    state: FovLineageState, tp: str, spine_id: str, status: str, *, overwrite: bool
) -> None:
    sid = _canonical_spine_id(spine_id)
    if not sid:
        return
    key = (tp, sid)
    state.uf.add(key)
    if overwrite or key not in state.spine_status:
        state.spine_status[key] = status


def _apply_status_list(
    state: FovLineageState,
    tp: str,
    ids: Iterable[str],
    status: str,
    *,
    overwrite: bool,
) -> None:
    for sid in ids:
        _set_lineage_status(state, tp, sid, status, overwrite=overwrite)


def ingest_manual_export_dir(
    state: FovLineageState,
    export_dir: Path,
    *,
    t1_tp: str,
    t2_tp: str,
) -> None:
    """Load one manual export (same status priority as Step 3 apply_export_qc)."""
    _apply_status_list(
        state,
        t2_tp,
        _read_id_column(export_dir / "new.csv", "t2_spine_id", "spine_id"),
        STATUS_NEW,
        overwrite=True,
    )
    _apply_status_list(
        state,
        t1_tp,
        _read_id_column(export_dir / "lost.csv", "t1_spine_id", "spine_id"),
        STATUS_LOST,
        overwrite=True,
    )

    matched_path = export_dir / "matched.csv"
    if matched_path.is_file():
        mdf = pd.read_csv(matched_path)
        for _, row in mdf.iterrows():
            t1_id = _canonical_spine_id(row.get("t1_spine_id", ""))
            t2_id = _canonical_spine_id(row.get("t2_spine_id", ""))
            if t1_id:
                _set_lineage_status(state, t1_tp, t1_id, STATUS_MATCHED, overwrite=True)
            if t2_id:
                _set_lineage_status(state, t2_tp, t2_id, STATUS_MATCHED, overwrite=True)
            if t1_id and t2_id:
                state.uf.union((t1_tp, t1_id), (t2_tp, t2_id))

    _apply_status_list(
        state,
        t1_tp,
        _read_id_column(export_dir / "removed_t1.csv", "t1_spine_id", "spine_id"),
        STATUS_ARTIFACT,
        overwrite=True,
    )
    _apply_status_list(
        state,
        t2_tp,
        _read_id_column(export_dir / "removed_t2.csv", "t2_spine_id", "spine_id"),
        STATUS_ARTIFACT,
        overwrite=True,
    )
    _apply_status_list(
        state,
        t1_tp,
        _read_id_column(export_dir / "ignored_t1.csv", "t1_spine_id", "spine_id"),
        STATUS_IGNORED,
        overwrite=True,
    )
    _apply_status_list(
        state,
        t2_tp,
        _read_id_column(export_dir / "ignored_t2.csv", "t2_spine_id", "spine_id"),
        STATUS_IGNORED,
        overwrite=True,
    )
    _apply_status_list(
        state,
        t2_tp,
        _read_id_column(export_dir / "not_in_t1_focus.csv", "t2_spine_id", "spine_id"),
        STATUS_NOT_IN_FOCUS,
        overwrite=True,
    )


def apply_forward_artifact_propagation(state: FovLineageState) -> int:
    """
    Artifact forward rule: if a spine is artifact at timepoint T, every later timepoint
    in the same lineage (union-find component) is marked artifact using known ID links.

    Skips components with an ID clash at one timepoint (cannot map identity reliably).
    """
    touched = 0
    for nodes in state.uf.components().values():
        ids_by_tp, clash = _ids_by_tp_from_component(nodes)
        if clash:
            continue
        earliest_idx: Optional[int] = None
        for tp, sid in ids_by_tp.items():
            if state.spine_status.get((tp, sid)) != STATUS_ARTIFACT:
                continue
            idx = TP_ORDER_INDEX[tp]
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx
        if earliest_idx is None:
            continue
        for tp, sid in ids_by_tp.items():
            if TP_ORDER_INDEX[tp] < earliest_idx:
                continue
            key = (tp, sid)
            if state.spine_status.get(key) != STATUS_ARTIFACT:
                state.spine_status[key] = STATUS_ARTIFACT
                touched += 1
    return touched


def artifact_ids_by_timepoint(state: FovLineageState) -> Dict[str, Set[str]]:
    """All spine IDs marked artifact per timepoint (after forward propagation)."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for (tp, sid), status in state.spine_status.items():
        if status == STATUS_ARTIFACT and sid:
            out[tp].add(sid)
    return {tp: set(ids) for tp, ids in out.items()}


def build_lineage_artifact_blacklist(fov_dir: Path) -> Dict[str, Set[str]]:
    """Manual-export lineages + forward artifact rule → per-timepoint ID sets."""
    state = build_lineage_state_from_fov(fov_dir)
    apply_forward_artifact_propagation(state)
    return artifact_ids_by_timepoint(state)


def build_lineage_state_from_fov(fov_dir: Path) -> FovLineageState:
    """Union-find lineages from all manual exports under this FOV (inferred folders ignored)."""
    state = FovLineageState()
    skip_dirs = {"old_inffered", "old_inferred"}
    for comp_dir in sorted(p for p in fov_dir.iterdir() if p.is_dir()):
        if comp_dir.name.lower() in skip_dirs:
            continue
        export_dir = latest_manual_export_dir(comp_dir)
        if export_dir is None:
            continue
        all_manual = [
            p for p in comp_dir.glob("*spine_annotator_export")
            if p.is_dir() and not _is_derived_export_dir(p)
        ]
        if len(all_manual) > 1:
            print(
                f"  using latest manual export for {comp_dir.name}: {export_dir.name} "
                f"({len(all_manual) - 1} older folder(s) ignored)"
            )
        meta_path = export_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        resolved = resolve_comparison_tps(comp_dir.name, meta)
        if not resolved:
            print(f"WARNING: Could not resolve timepoints for {comp_dir}; skipping.")
            continue
        t1_tp, t2_tp = resolved
        t1_tiff, t2_tiff = parse_tiff_paths_from_export(export_dir)
        if t1_tiff is not None:
            state.tiff_by_tp.setdefault(t1_tp, t1_tiff)
        if t2_tiff is not None:
            state.tiff_by_tp.setdefault(t2_tp, t2_tiff)
        _register_tiff_paths_from_meta(state, meta, t1_tp=t1_tp, t2_tp=t2_tp)
        ingest_manual_export_dir(state, export_dir, t1_tp=t1_tp, t2_tp=t2_tp)
        state.source_exports.append(str(export_dir))
        state.source_comparisons.append(comp_dir.name)
    return state


def _tp_index(tp: str) -> int:
    return TP_ORDER_INDEX[tp]


def _is_biological(status: str) -> bool:
    if status in {STATUS_ABSENT, STATUS_ARTIFACT}:
        return False
    if status in UNCERTAINTY_STATUSES:
        return False
    return True


def _status_for_component_node(state: FovLineageState, tp: str, sid: str) -> str:
    return state.spine_status.get((tp, sid), STATUS_MATCHED)


def _earliest_timepoint_with_status(status_by_tp: Dict[str, str], status: str) -> Optional[str]:
    for tp in TP_ORDER:
        if status_by_tp.get(tp) == status:
            return tp
    return None


def _first_id_timepoint(ids_by_tp: Dict[str, str]) -> Optional[str]:
    if not ids_by_tp:
        return None
    return min(ids_by_tp.keys(), key=_tp_index)


def _apply_chronological_rules(
    ids_by_tp: Dict[str, str],
    status_by_tp: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Graph-level chronological rules before pairwise routing.

    - Forward artifact from first artifact timepoint onward.
    - Forward death: LOST at T => absent at all later timepoints.
    - Birth: NEW at T_B => absent before T_B; first ID appearance clears earlier TPs.
    """
    ids = dict(ids_by_tp)
    status = dict(status_by_tp)

    art_tp = _earliest_timepoint_with_status(status, STATUS_ARTIFACT)
    if art_tp is not None:
        art_idx = _tp_index(art_tp)
        for tp in TP_ORDER:
            if _tp_index(tp) >= art_idx and tp in ids:
                status[tp] = STATUS_ARTIFACT

    for tp in TP_ORDER:
        if status.get(tp) != STATUS_LOST or tp not in ids:
            continue
        for later in TP_ORDER[_tp_index(tp) + 1 :]:
            ids.pop(later, None)
            status[later] = STATUS_ABSENT

    birth_tp = _earliest_timepoint_with_status(status, STATUS_NEW)
    if birth_tp is not None:
        birth_idx = _tp_index(birth_tp)
        for tp in TP_ORDER:
            if _tp_index(tp) < birth_idx:
                ids.pop(tp, None)
                status[tp] = STATUS_ABSENT

    first_tp = _first_id_timepoint(ids)
    if first_tp is not None:
        first_idx = _tp_index(first_tp)
        for tp in TP_ORDER:
            if _tp_index(tp) < first_idx:
                ids.pop(tp, None)
                status[tp] = STATUS_ABSENT

    return ids, status


def _ignore_breaks_chain(
    status_by_tp: Dict[str, str],
    *,
    t_a: str,
    t_b: str,
) -> bool:
    """True if not-in-focus sits strictly between T_A and T_B on the timeline."""
    ia, ib = _tp_index(t_a), _tp_index(t_b)
    if ib - ia <= 1:
        return False
    for tp in TP_ORDER[ia + 1 : ib]:
        if status_by_tp.get(tp) == STATUS_NOT_IN_FOCUS:
            return True
    return False


def _ignored_unresolved_reason(
    status_by_tp: Dict[str, str],
    *,
    t_a: str,
    t_b: str,
) -> Optional[str]:
    """
    If this lineage was marked ignored in a prior manual session within the pair
    window (T_A, T_B, or strictly between), return an unresolved reason code.
    """
    if status_by_tp.get(t_a) == STATUS_IGNORED:
        return "t1_ignored"
    if status_by_tp.get(t_b) == STATUS_IGNORED:
        return "t2_ignored"
    ia, ib = _tp_index(t_a), _tp_index(t_b)
    for tp in TP_ORDER[ia + 1 : ib]:
        if status_by_tp.get(tp) == STATUS_IGNORED:
            return "ignore_breaks_chain"
    return None


def _collect_component_timeline(
    state: FovLineageState,
    nodes: List[Tuple[str, str]],
) -> Tuple[Dict[str, str], Dict[str, str], Optional[dict]]:
    ids_by_tp, clash = _ids_by_tp_from_component(nodes)
    if clash:
        return ids_by_tp, {}, clash

    status_by_tp: Dict[str, str] = {}
    for tp, sid in ids_by_tp.items():
        status_by_tp[tp] = _status_for_component_node(state, tp, sid)

    ids_by_tp, status_by_tp = _apply_chronological_rules(ids_by_tp, status_by_tp)
    return ids_by_tp, status_by_tp, None


def _ids_by_tp_from_component(
    nodes: List[Tuple[str, str]],
) -> Tuple[Dict[str, str], Optional[dict]]:
    out: Dict[str, str] = {}
    for tp, sid in nodes:
        sid_s = _canonical_spine_id(sid)
        if not sid_s:
            continue
        if tp in out and out[tp] != sid_s:
            return out, {
                "reason": "lineage_id_clash",
                "detail": f"{tp}: {out[tp]!r} vs {sid_s!r}",
            }
        out[tp] = sid_s
    return out, None


def route_lineage_to_pair(
    *,
    ids_by_tp: Dict[str, str],
    status_by_tp: Dict[str, str],
    t1_tp: str,
    t2_tp: str,
) -> InferredPairTables:
    """
    Route one lineage to pairwise CSV rows using chronology-resolved timeline.

    Prior ``ignored`` flags from manual exports always → unresolved_manual_review.
    """
    out = InferredPairTables()
    t1_id = _canonical_spine_id(ids_by_tp.get(t1_tp, ""))
    t2_id = _canonical_spine_id(ids_by_tp.get(t2_tp, ""))
    st1 = status_by_tp.get(t1_tp, STATUS_ABSENT)
    st2 = status_by_tp.get(t2_tp, STATUS_ABSENT)

    ignored_reason = _ignored_unresolved_reason(status_by_tp, t_a=t1_tp, t_b=t2_tp)
    if ignored_reason:
        out.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": t1_id,
                "t2_spine_id": t2_id,
                "reason": ignored_reason,
                "detail": f"{t1_tp} vs {t2_tp}",
            }
        )
        return out

    if _ignore_breaks_chain(status_by_tp, t_a=t1_tp, t_b=t2_tp):
        out.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": t1_id,
                "t2_spine_id": t2_id,
                "reason": "ignore_breaks_chain",
                "detail": f"{t1_tp} vs {t2_tp}",
            }
        )
        return out

    if t1_id and st1 == STATUS_ARTIFACT:
        out.removed_t1.append(t1_id)
    if t2_id and st2 == STATUS_ARTIFACT:
        out.removed_t2.append(t2_id)
    if (t1_id and st1 == STATUS_ARTIFACT) or (t2_id and st2 == STATUS_ARTIFACT):
        return out

    if st1 == STATUS_NOT_IN_FOCUS:
        t1_id, st1 = "", STATUS_ABSENT
    if st2 == STATUS_NOT_IN_FOCUS:
        t2_id, st2 = "", STATUS_ABSENT

    t1_present = bool(t1_id)
    t2_present = bool(t2_id)

    birth_tp = _earliest_timepoint_with_status(status_by_tp, STATUS_NEW)
    if (
        birth_tp is not None
        and _tp_index(t1_tp) < _tp_index(birth_tp) <= _tp_index(t2_tp)
        and t2_present
        and not t1_present
    ):
        out.new.append(t2_id)
        return out

    if t1_present and t2_present:
        if _is_biological(st1) and _is_biological(st2):
            out.matched.append(
                {
                    "t1_spine_id": t1_id,
                    "t2_spine_id": t2_id,
                    "baseline_spine_id": "",
                    "source": INFERENCE_SOURCE_LINEAGE,
                }
            )
        return out

    if t1_present and not t2_present:
        if _is_biological(st1) or st1 == STATUS_LOST:
            out.lost.append(t1_id)
        return out

    if t2_present and not t1_present:
        if _is_biological(st2) or st2 == STATUS_NEW:
            out.new.append(t2_id)
        return out

    return out


def merge_inferred_tables(target: InferredPairTables, chunk: InferredPairTables) -> None:
    target.matched.extend(chunk.matched)
    target.new.extend(chunk.new)
    target.lost.extend(chunk.lost)
    target.removed_t1.extend(chunk.removed_t1)
    target.removed_t2.extend(chunk.removed_t2)
    target.unresolved.extend(chunk.unresolved)


def build_pair_exports_from_lineage(
    state: FovLineageState,
    *,
    t_a: str,
    t_b: str,
) -> InferredPairTables:
    tables = InferredPairTables()
    for nodes in state.uf.components().values():
        ids_by_tp, status_by_tp, clash = _collect_component_timeline(state, nodes)
        if clash:
            tables.unresolved.append(
                {
                    "baseline_spine_id": "",
                    "t1_spine_id": ids_by_tp.get(t_a, ""),
                    "t2_spine_id": ids_by_tp.get(t_b, ""),
                    "reason": clash["reason"],
                    "detail": clash["detail"],
                }
            )
            continue
        merge_inferred_tables(
            tables,
            route_lineage_to_pair(
                ids_by_tp=ids_by_tp,
                status_by_tp=status_by_tp,
                t1_tp=t_a,
                t2_tp=t_b,
            ),
        )
    _append_orphan_ignored_unresolved(state, t_a=t_a, t_b=t_b, tables=tables)
    return tables


def _pair_timepoints(t_a: str, t_b: str) -> List[str]:
    ia, ib = _tp_index(t_a), _tp_index(t_b)
    return TP_ORDER[ia : ib + 1]


def _classified_spine_ids(tables: InferredPairTables) -> Set[str]:
    """Spine IDs already assigned to an output bucket for this pair."""
    ids: Set[str] = set()
    for row in tables.matched:
        for key in ("t1_spine_id", "t2_spine_id"):
            sid = _canonical_spine_id(row.get(key, ""))
            if sid:
                ids.add(sid)
    for sid in tables.new:
        c = _canonical_spine_id(sid)
        if c:
            ids.add(c)
    for sid in tables.lost:
        c = _canonical_spine_id(sid)
        if c:
            ids.add(c)
    for sid in tables.removed_t1 + tables.removed_t2:
        c = _canonical_spine_id(sid)
        if c:
            ids.add(c)
    for row in tables.unresolved:
        for key in ("t1_spine_id", "t2_spine_id"):
            sid = _canonical_spine_id(row.get(key, ""))
            if sid:
                ids.add(sid)
    return ids


def _append_orphan_ignored_unresolved(
    state: FovLineageState,
    *,
    t_a: str,
    t_b: str,
    tables: InferredPairTables,
) -> None:
    """
    Safety net: any ignored spine in the pair window that did not route through
    a lineage component still lands in unresolved_manual_review.
    """
    window = set(_pair_timepoints(t_a, t_b))
    classified = _classified_spine_ids(tables)
    for (tp, sid), status in state.spine_status.items():
        if status != STATUS_IGNORED or tp not in window:
            continue
        canon = _canonical_spine_id(sid)
        if not canon or canon in classified:
            continue
        if tp == t_a:
            reason = "t1_ignored"
            t1_id, t2_id = canon, ""
        elif tp == t_b:
            reason = "t2_ignored"
            t1_id, t2_id = "", canon
        else:
            reason = "ignore_breaks_chain"
            t1_id, t2_id = "", ""
        tables.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": t1_id,
                "t2_spine_id": t2_id,
                "reason": reason,
                "detail": f"{t_a} vs {t_b} ({tp})",
            }
        )
        classified.add(canon)


def resolve_pair_tiff_sources(
    state: FovLineageState,
    baseline_exports: Dict[str, BaselineExportData],
    *,
    t_a: str,
    t_b: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    t_a_src = state.tiff_by_tp.get(t_a)
    t_b_src = state.tiff_by_tp.get(t_b)
    if (t_a_src is None or not t_a_src.is_file()) and t_a in baseline_exports:
        t_a_src = baseline_exports[t_a].t2_tiff
    if (t_b_src is None or not t_b_src.is_file()) and t_b in baseline_exports:
        t_b_src = baseline_exports[t_b].t2_tiff
    return t_a_src, t_b_src


def _extract_json_objects(text: str) -> List[dict]:
    """Pull JSON objects from annotator activity logs (--- separated blocks)."""
    objects: List[dict] = []
    for block in re.split(r"\n---\n", text):
        block = block.strip()
        if not block:
            continue
        # Drop leading timestamp line if present.
        lines = block.splitlines()
        json_start = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("{"):
                json_start = i
                break
        payload = "\n".join(lines[json_start:]).strip()
        if not payload.startswith("{"):
            continue
        try:
            objects.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return objects


def _paths_from_log_dict(obj: dict) -> Tuple[Optional[Path], Optional[Path]]:
    for key in ("selected_files", "files", "paths"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            obj = {**obj, **nested}
    t1 = obj.get("t1_tiff_path")
    t2 = obj.get("t2_tiff_path")
    p1 = Path(str(t1)).expanduser() if not _is_blank(t1) else None
    p2 = Path(str(t2)).expanduser() if not _is_blank(t2) else None
    return (
        p1.resolve() if p1 is not None and p1.is_file() else None,
        p2.resolve() if p2 is not None and p2.is_file() else None,
    )


def parse_tiff_paths_from_export(export_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Read T1/T2 TIFF paths from matching log, then metadata.json."""
    for log_name in LOG_CANDIDATES:
        log_path = export_dir / log_name
        if not log_path.is_file() or log_path.stat().st_size == 0:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        t1_path: Optional[Path] = None
        t2_path: Optional[Path] = None
        for obj in _extract_json_objects(text):
            p1, p2 = _paths_from_log_dict(obj)
            if p1 is not None:
                t1_path = p1
            if p2 is not None:
                t2_path = p2
        if t1_path or t2_path:
            return t1_path, t2_path

    meta_path = export_dir / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        p1, p2 = _paths_from_log_dict(meta)
        return p1, p2

    return None, None


def load_baseline_export(comp_dir: Path, follow_up_tp: str) -> Optional[BaselineExportData]:
    export_dir = latest_manual_export_dir(comp_dir)
    if export_dir is None:
        return None

    data = BaselineExportData(
        comparison=comp_dir.name,
        follow_up_tp=follow_up_tp,
        export_dir=export_dir,
    )

    matched_path = export_dir / "matched.csv"
    if matched_path.is_file():
        mdf = pd.read_csv(matched_path)
        for _, row in mdf.iterrows():
            b_id = _canonical_spine_id(row.get("t1_spine_id", ""))
            fu_id = _canonical_spine_id(row.get("t2_spine_id", ""))
            if b_id and fu_id:
                data.matched[b_id] = fu_id

    data.lost_baseline = set(_read_id_column(export_dir / "lost.csv", "t1_spine_id", "spine_id"))
    data.new_follow_up = set(_read_id_column(export_dir / "new.csv", "t2_spine_id", "spine_id"))
    data.ignored_baseline = set(_read_id_column(export_dir / "ignored_t1.csv", "t1_spine_id", "spine_id"))
    data.ignored_follow_up = set(
        _read_id_column(export_dir / "ignored_t2.csv", "t2_spine_id", "spine_id")
    )
    data.removed_baseline = set(_read_id_column(export_dir / "removed_t1.csv", "t1_spine_id", "spine_id"))
    data.removed_follow_up = set(
        _read_id_column(export_dir / "removed_t2.csv", "t2_spine_id", "spine_id")
    )

    t1_tiff, t2_tiff = parse_tiff_paths_from_export(export_dir)
    data.t1_tiff = t1_tiff
    data.t2_tiff = t2_tiff
    return data


def _baseline_status(data: BaselineExportData, baseline_id: str) -> str:
    """How baseline spine `baseline_id` relates to the follow-up timepoint."""
    if baseline_id in data.matched:
        return "matched"
    if baseline_id in data.lost_baseline:
        return "lost"
    return "absent"


def _follow_up_has_qc_issue(data: BaselineExportData, baseline_id: str, fu_id: str) -> Optional[str]:
    if baseline_id in data.ignored_baseline or baseline_id in data.removed_baseline:
        return "baseline_ignored_or_removed"
    if fu_id in data.ignored_follow_up or fu_id in data.removed_follow_up:
        return "follow_up_ignored_or_removed"
    return None


def infer_pair_via_baseline(
    export_a: BaselineExportData,
    export_b: BaselineExportData,
    *,
    t_a: str,
    t_b: str,
) -> InferredPairTables:
    """
    Infer T_A (earlier) vs T_B (later) from two Baseline <-> T exports.

    T_A is the earlier chronological timepoint; CSV columns follow annotator convention
    (t1 = T_A, t2 = T_B).
    """
    tables = InferredPairTables()
    seen_unresolved_fu: Set[str] = set()

    all_baseline_ids: Set[str] = set()
    all_baseline_ids |= set(export_a.matched) | export_a.lost_baseline
    all_baseline_ids |= set(export_b.matched) | export_b.lost_baseline
    all_baseline_ids |= export_a.ignored_baseline | export_a.removed_baseline
    all_baseline_ids |= export_b.ignored_baseline | export_b.removed_baseline

    for b_id in sorted(all_baseline_ids, key=lambda x: (len(x), x)):
        st_a = _baseline_status(export_a, b_id)
        st_b = _baseline_status(export_b, b_id)
        id_a = export_a.matched.get(b_id, "")
        id_b = export_b.matched.get(b_id, "")

        qc_a = _follow_up_has_qc_issue(export_a, b_id, id_a) if id_a else (
            "baseline_ignored_or_removed" if b_id in export_a.ignored_baseline | export_a.removed_baseline else None
        )
        qc_b = _follow_up_has_qc_issue(export_b, b_id, id_b) if id_b else (
            "baseline_ignored_or_removed" if b_id in export_b.ignored_baseline | export_b.removed_baseline else None
        )
        if qc_a or qc_b:
            tables.unresolved.append(
                {
                    "baseline_spine_id": b_id,
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "reason": qc_a or qc_b,
                    "detail": f"{t_a} vs {t_b}",
                }
            )
            continue

        if st_a == "matched" and st_b == "matched" and id_a and id_b:
            tables.matched.append(
                {
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "baseline_spine_id": b_id,
                    "source": "baseline_bridge",
                }
            )
        elif st_a == "matched" and st_b == "lost" and id_a:
            tables.lost.append(id_a)
        elif st_a == "lost" and st_b == "matched" and id_b:
            tables.new.append(id_b)
        elif st_a == "lost" and st_b == "lost":
            pass  # absent at both follow-ups — no row
        elif st_a == "absent" and st_b == "absent":
            pass
        else:
            tables.unresolved.append(
                {
                    "baseline_spine_id": b_id,
                    "t1_spine_id": id_a,
                    "t2_spine_id": id_b,
                    "reason": f"ambiguous_{st_a}_{st_b}",
                    "detail": f"{t_a} vs {t_b}",
                }
            )

    # Spines new at T_A cannot be bridged through baseline.
    for fu_id in sorted(export_a.new_follow_up, key=lambda x: (len(x), x)):
        if fu_id in seen_unresolved_fu:
            continue
        seen_unresolved_fu.add(fu_id)
        tables.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": fu_id,
                "t2_spine_id": "",
                "reason": "new_at_t1_no_baseline_bridge",
                "detail": t_a,
            }
        )

    # Spines new at T_B without baseline (not reachable via bridge).
    for fu_id in sorted(export_b.new_follow_up, key=lambda x: (len(x), x)):
        if fu_id in export_a.new_follow_up:
            continue  # already flagged if also new at T_A
        tables.unresolved.append(
            {
                "baseline_spine_id": "",
                "t1_spine_id": "",
                "t2_spine_id": fu_id,
                "reason": "new_at_t2_no_baseline_bridge",
                "detail": t_b,
            }
        )

    # Follow-up IDs flagged as ignored/removed without appearing in matched rows.
    for fu_id in sorted(export_a.ignored_follow_up | export_a.removed_follow_up):
        if fu_id not in export_a.matched.values():
            tables.unresolved.append(
                {
                    "baseline_spine_id": "",
                    "t1_spine_id": fu_id,
                    "t2_spine_id": "",
                    "reason": "t1_ignored_or_removed",
                    "detail": t_a,
                }
            )
    for fu_id in sorted(export_b.ignored_follow_up | export_b.removed_follow_up):
        if fu_id not in export_b.matched.values():
            tables.unresolved.append(
                {
                    "baseline_spine_id": "",
                    "t1_spine_id": "",
                    "t2_spine_id": fu_id,
                    "reason": "t2_ignored_or_removed",
                    "detail": t_b,
                }
            )

    return tables


def _dedupe_sorted(ids: Iterable[str]) -> List[str]:
    return sorted({x for x in (_canonical_spine_id(i) for i in ids) if x})


def _dedupe_matched(
    rows: List[dict],
    *,
    clash_unresolved: Optional[List[dict]] = None,
    pair_detail: str = "",
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t1_spine_id", "t2_spine_id", "baseline_spine_id", "source"])
    kept: List[dict] = []
    seen_t2: Dict[str, str] = {}
    seen_t1: Dict[str, str] = {}
    for row in rows:
        t1 = _canonical_spine_id(row.get("t1_spine_id", ""))
        t2 = _canonical_spine_id(row.get("t2_spine_id", ""))
        if not t1 or not t2:
            continue
        clash_reason: Optional[str] = None
        if t2 in seen_t2 and seen_t2[t2] != t1:
            clash_reason = "matched_t2_clash"
        elif t1 in seen_t1 and seen_t1[t1] != t2:
            clash_reason = "matched_t1_clash"
        if clash_reason:
            if clash_unresolved is not None:
                clash_unresolved.append(
                    {
                        "baseline_spine_id": "",
                        "t1_spine_id": t1,
                        "t2_spine_id": t2,
                        "reason": clash_reason,
                        "detail": pair_detail or f"t2={t2} t1 was {seen_t2.get(t2, seen_t1.get(t1, ''))}",
                    }
                )
            continue
        seen_t2[t2] = t1
        seen_t1[t1] = t2
        kept.append(row)
    if not kept:
        return pd.DataFrame(columns=["t1_spine_id", "t2_spine_id", "baseline_spine_id", "source"])
    df = pd.DataFrame(kept)
    return df.drop_duplicates(subset=["t1_spine_id", "t2_spine_id"], keep="first").sort_values(
        ["t1_spine_id", "t2_spine_id"]
    )


def _dedupe_unresolved(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["baseline_spine_id", "t1_spine_id", "t2_spine_id", "reason", "detail"]
        )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        subset=["baseline_spine_id", "t1_spine_id", "t2_spine_id", "reason"], keep="first"
    ).sort_values(["reason", "t1_spine_id", "t2_spine_id"])


def copy_tiff_to_input_files(
    src: Path,
    dest_dir: Path,
    *,
    role: str,
    tp: str,
    fov: int,
) -> Path:
    suffix = src.suffix if src.suffix else ".tif"
    safe_tp = re.sub(r"[^\w\-]+", "_", tp)
    dest_name = f"{role}_{safe_tp}_fov{fov}{suffix}"
    dest = dest_dir / dest_name
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)
    return dest.resolve()


def stage_input_files(
    export_dir: Path,
    *,
    fov: int,
    t_a: str,
    t_b: str,
    t_a_src: Optional[Path],
    t_b_src: Optional[Path],
    symlink: bool,
) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    input_dir = export_dir / "input_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if t_a_src is None or not t_a_src.is_file():
        warnings.append(f"missing TIFF for {t_a}")
    if t_b_src is None or not t_b_src.is_file():
        warnings.append(f"missing TIFF for {t_b}")

    t_a_dest: Optional[Path] = None
    t_b_dest: Optional[Path] = None
    if t_a_src is not None and t_a_src.is_file():
        if symlink:
            safe_a = re.sub(r"[^\w\-]+", "_", t_a)
            dest = input_dir / f"t1_{safe_a}_fov{fov}{t_a_src.suffix or '.tif'}"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(t_a_src)
            t_a_dest = dest.resolve()
        else:
            t_a_dest = copy_tiff_to_input_files(t_a_src, input_dir, role="t1", tp=t_a, fov=fov)
    if t_b_src is not None and t_b_src.is_file():
        if symlink:
            safe_b = re.sub(r"[^\w\-]+", "_", t_b)
            dest = input_dir / f"t2_{safe_b}_fov{fov}{t_b_src.suffix or '.tif'}"
            if dest.exists():
                dest.unlink()
            dest.symlink_to(t_b_src)
            t_b_dest = dest.resolve()
        else:
            t_b_dest = copy_tiff_to_input_files(t_b_src, input_dir, role="t2", tp=t_b, fov=fov)

    readme_lines = [
        f"Inferred pairwise review stacks (FOV {fov}).",
        f"T1 timepoint: {t_a}",
        f"T2 timepoint: {t_b}",
        "",
    ]
    if t_a_dest:
        readme_lines.append(f"T1 file: {t_a_dest.name}")
        readme_lines.append(f"  source: {t_a_src}")
    if t_b_dest:
        readme_lines.append(f"T2 file: {t_b_dest.name}")
        readme_lines.append(f"  source: {t_b_src}")
    (input_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return t_a_dest, t_b_dest, warnings


def write_inferred_export(
    tables: InferredPairTables,
    *,
    export_dir: Path,
    respan_root: Path,
    fov: int,
    t_a: str,
    t_b: str,
    comparison: str,
    lineage: FovLineageState,
    baseline_exports: Dict[str, BaselineExportData],
    symlink_tiffs: bool,
    skip_tiffs: bool,
    skip_validation: bool = True,
) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)

    pair_detail = f"{t_a} vs {t_b}"
    matched_df = _dedupe_matched(
        tables.matched, clash_unresolved=tables.unresolved, pair_detail=pair_detail
    )
    new_ids = _dedupe_sorted(tables.new)
    lost_ids = _dedupe_sorted(tables.lost)
    removed_t1 = _dedupe_sorted(tables.removed_t1)
    removed_t2 = _dedupe_sorted(tables.removed_t2)
    unresolved_df = _dedupe_unresolved(tables.unresolved)

    _safe_write_csv(matched_df, export_dir / "matched.csv")
    _safe_write_csv(pd.DataFrame({"t2_spine_id": new_ids}), export_dir / "new.csv")
    _safe_write_csv(pd.DataFrame({"t1_spine_id": lost_ids}), export_dir / "lost.csv")
    _safe_write_csv(pd.DataFrame({"t1_spine_id": removed_t1}), export_dir / "removed_t1.csv")
    _safe_write_csv(pd.DataFrame({"t2_spine_id": removed_t2}), export_dir / "removed_t2.csv")
    _safe_write_csv(unresolved_df, export_dir / "unresolved_manual_review.csv")

    metadata: dict = {
        "source": "baseline_bridge_inference",
        "inference_method": INFERENCE_SOURCE_LINEAGE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_run_stamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "animal_comparison_folder": comparison,
        "fov": fov,
        "t1_timepoint": t_a,
        "t2_timepoint": t_b,
        "baseline_timepoint": BASELINE_TP,
        "source_manual_exports": list(lineage.source_exports),
        "source_comparisons": list(lineage.source_comparisons),
        "source_baseline_exports": {
            tp: str(data.export_dir) for tp, data in baseline_exports.items()
        },
        "matched_count": int(len(matched_df)),
        "new_count": int(len(new_ids)),
        "lost_count": int(len(lost_ids)),
        "removed_t1_count": int(len(removed_t1)),
        "removed_t2_count": int(len(removed_t2)),
        "unresolved_count": int(len(unresolved_df)),
    }

    t_a_tiff, t_b_tiff = resolve_pair_tiff_sources(lineage, baseline_exports, t_a=t_a, t_b=t_b)

    if not skip_tiffs:
        t1_dest, t2_dest, warnings = stage_input_files(
            export_dir,
            fov=fov,
            t_a=t_a,
            t_b=t_b,
            t_a_src=t_a_tiff,
            t_b_src=t_b_tiff,
            symlink=symlink_tiffs,
        )
        if t1_dest:
            metadata["t1_tiff_path"] = str(t1_dest)
        if t2_dest:
            metadata["t2_tiff_path"] = str(t2_dest)
        if warnings:
            metadata["tiff_warnings"] = warnings

        if not skip_validation:
            prov = validate_or_raise(
                export_dir=export_dir,
                respan_root=respan_root,
                fov=fov,
                t1_timepoint=t_a,
                t2_timepoint=t_b,
                input_dir=export_dir / "input_files",
            )
            metadata["t1_csv_path"] = str(prov.t1_csv_path)
            metadata["t2_csv_path"] = str(prov.t2_csv_path)
            metadata["provenance_validation"] = {
                "ok": True,
                "t1_spine_count": prov.t1_spine_count,
                "t2_spine_count": prov.t2_spine_count,
                "matched_ids_checked": prov.matched_ids_checked,
                "new_ids_checked": prov.new_ids_checked,
                "lost_ids_checked": prov.lost_ids_checked,
                "unresolved_ids_checked": prov.unresolved_ids_checked,
                "baseline_ids_checked": prov.baseline_ids_checked,
            }
        else:
            t1_csv = find_spine_csv(respan_root, t_a, fov)
            t2_csv = find_spine_csv(respan_root, t_b, fov)
            stage_session_inputs(export_dir, t1_csv_src=t1_csv, t2_csv_src=t2_csv)
            metadata["t1_csv_path"] = str(t1_csv)
            metadata["t2_csv_path"] = str(t2_csv)
    else:
        t1_csv = find_spine_csv(respan_root, t_a, fov)
        t2_csv = find_spine_csv(respan_root, t_b, fov)
        metadata["t1_csv_path"] = str(t1_csv)
        metadata["t2_csv_path"] = str(t2_csv)

    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def pair_already_has_manual_export(fov_dir: Path, comparison: str) -> bool:
    comp_dir = fov_dir / comparison
    return latest_manual_export_dir(comp_dir) is not None if comp_dir.is_dir() else False


def all_chronological_pairs_to_generate(
    existing_manual: Set[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """All T1<T2 timepoint pairs not yet covered by a manual export."""
    pairs: List[Tuple[str, str]] = []
    for i, t_a in enumerate(TP_ORDER):
        for t_b in TP_ORDER[i + 1 :]:
            if (t_a, t_b) not in existing_manual:
                pairs.append((t_a, t_b))
    return pairs


def existing_manual_pairs(fov_dir: Path) -> Set[Tuple[str, str]]:
    """Pairs that already have a manual (non-inferred) export folder."""
    found: Set[Tuple[str, str]] = set()
    for comp_dir in sorted(p for p in fov_dir.iterdir() if p.is_dir()):
        if not list_manual_export_dirs(comp_dir):
            continue
        name = comp_dir.name
        if name in COMPARISON_MAP:
            found.add(COMPARISON_MAP[name])
            continue
        if name in COMPARISON_BY_PAIR.values():
            for pair, folder in COMPARISON_BY_PAIR.items():
                if folder == name:
                    found.add(pair)
            continue
        if " - " in name:
            left, right = name.split(" - ", 1)
            left, right = left.strip(), right.strip()
            if left in TP_ORDER_INDEX and right in TP_ORDER_INDEX:
                t1, t2 = sorted((left, right), key=lambda t: TP_ORDER_INDEX[t])
                found.add((t1, t2))
    return found


def load_baseline_exports_for_fov(fov_dir: Path) -> Dict[str, BaselineExportData]:
    exports: Dict[str, BaselineExportData] = {}
    for tp, comp_name in BASELINE_COMPARISON_MAP.items():
        comp_dir = fov_dir / comp_name
        if not comp_dir.is_dir():
            continue
        data = load_baseline_export(comp_dir, tp)
        if data is not None:
            exports[tp] = data
    return exports


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Infer non-baseline pairwise spine exports from all manual annotator exports "
            "under results/fovN (union-find lineages). Re-run after validating more pairs."
        )
    )
    p.add_argument(
        "--animal-id",
        default="GP04",
        help="Animal folder name under imaging root (e.g. GP04, GP08).",
    )
    p.add_argument(
        "--imaging-root",
        type=Path,
        default=None,
        help=(
            "Parent of animal folders (e.g. E:/.../Imaging). "
            "Default: search for IMAGING under cwd and script parents."
        ),
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override path to respan/results (skips imaging-root/animal-id discovery).",
    )
    p.add_argument(
        "--fov-dir",
        type=Path,
        default=None,
        help="Run on one FOV folder only (e.g. results/fov1 - Copy). Overrides --results-root layout.",
    )
    p.add_argument(
        "--fov-number",
        type=int,
        default=None,
        help="FOV index for Tables/TIFF lookup when --fov-dir name is not fovN (default: parsed or 1).",
    )
    p.add_argument("--fovs", type=int, nargs="*", default=None, help="Optional FOV subset.")
    p.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help='Only these pairs, e.g. "mid-droplet - end-lever" (chronological T1 - T2).',
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Also infer pairs that already have a manual export (default: skip those).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned writes only.")
    p.add_argument("--no-tiffs", action="store_true", help="Skip copying TIFFs into input_files/.")
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run optional ID/coordinate sanity check against Tables (off by default).",
    )
    p.add_argument(
        "--symlink-tiffs",
        action="store_true",
        help="Symlink TIFFs into input_files/ instead of copying.",
    )
    return p.parse_args(argv)


def _default_imaging_root() -> Optional[Path]:
    candidates = [
        Path(r"E:\Noa\Pons - layer 5\Imaging"),
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for base in candidates:
        for name in ("IMAGING", "Imaging", "imaging"):
            p = base / name if (base / name).exists() else base
            if p.is_dir() and any(p.glob("GP*")):
                return p
    return None


def parse_fov_number_from_dir(fov_dir: Path, explicit: Optional[int] = None) -> int:
    if explicit is not None:
        return int(explicit)
    m = re.search(r"fov\s*(\d+)", fov_dir.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1


def resolve_fov_jobs(
    *,
    results_root: Path,
    fov_dir_arg: Optional[Path],
    fov_number_arg: Optional[int],
    fovs_filter: Optional[List[int]],
) -> List[Tuple[int, Path]]:
    """Return list of (fov_number, absolute_fov_dir) to process."""
    if fov_dir_arg is not None:
        fov_dir = fov_dir_arg.expanduser().resolve()
        if not fov_dir.is_dir():
            raise FileNotFoundError(f"FOV directory not found: {fov_dir}")
        fov = parse_fov_number_from_dir(fov_dir, fov_number_arg)
        return [(fov, fov_dir)]

    fovs = discover_fovs(results_root)
    if fovs_filter:
        wanted = set(fovs_filter)
        fovs = [f for f in fovs if f in wanted]
    return [(f, results_root / f"fov{f}") for f in fovs]


def parse_pair_arg(raw: str) -> Tuple[str, str]:
    if " - " not in raw:
        raise argparse.ArgumentTypeError(f"Invalid pair {raw!r}; expected 'T1 - T2'.")
    left, right = raw.split(" - ", 1)
    left, right = left.strip(), right.strip()
    if left not in TP_ORDER_INDEX or right not in TP_ORDER_INDEX:
        raise argparse.ArgumentTypeError(f"Unknown timepoint in {raw!r}")
    if TP_ORDER_INDEX[left] >= TP_ORDER_INDEX[right]:
        raise argparse.ArgumentTypeError(f"T1 must be earlier than T2: {raw!r}")
    return left, right


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.results_root:
        results_root = args.results_root.expanduser().resolve()
    else:
        imaging_root = args.imaging_root or _default_imaging_root()
        if imaging_root is None:
            print(
                "ERROR: Could not locate imaging root. Pass --imaging-root or --results-root.",
                file=sys.stderr,
            )
            return 1
        results_root = resolve_results_root(imaging_root.expanduser().resolve(), args.animal_id)

    if args.fov_dir:
        results_root = args.fov_dir.expanduser().resolve().parent
    elif not results_root.is_dir():
        print(f"ERROR: Results root not found: {results_root}", file=sys.stderr)
        return 1

    try:
        fov_jobs = resolve_fov_jobs(
            results_root=results_root,
            fov_dir_arg=args.fov_dir,
            fov_number_arg=args.fov_number,
            fovs_filter=args.fovs,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not fov_jobs:
        print(f"ERROR: No fov* directories under {results_root}", file=sys.stderr)
        return 1

    if args.pairs:
        target_pairs = [parse_pair_arg(p) for p in args.pairs]
    else:
        target_pairs = None  # filled per FOV

    respan_root = results_root.parent
    print(f"Animal:       {args.animal_id}")
    print(f"Results root: {results_root}")
    print(f"Respan root:  {respan_root}")
    print(f"FOV jobs:     {[(f, str(d)) for f, d in fov_jobs]}")
    print()

    manifest: List[dict] = []

    for fov, fov_dir in fov_jobs:
        lineage = build_lineage_state_from_fov(fov_dir)
        if len(lineage.source_exports) < 2:
            print(
                f"fov{fov}: need >=2 manual exports (non-inferred); "
                f"found {len(lineage.source_exports)} — skipping."
            )
            continue

        baseline_exports = load_baseline_exports_for_fov(fov_dir)

        pairs_for_fov = target_pairs or all_chronological_pairs_to_generate(
            existing_manual_pairs(fov_dir)
        )
        if not pairs_for_fov:
            print(f"fov{fov}: all chronological pairs already have manual exports.")
            continue

        art_prop = apply_forward_artifact_propagation(lineage)
        print(
            f"fov{fov}: {len(lineage.source_exports)} manual export(s) from "
            f"{', '.join(lineage.source_comparisons)}; "
            f"{len(lineage.uf.components())} lineage component(s)."
            + (f" Forward artifact: {art_prop} spine×TP updated." if art_prop else "")
        )

        for t_a, t_b in pairs_for_fov:
            comparison = comparison_folder_name(t_a, t_b)
            comp_dir = fov_dir / comparison
            export_dir = comp_dir / LATEST_INFERRED_EXPORT_DIRNAME

            if pair_already_has_manual_export(fov_dir, comparison) and not args.force:
                print(f"  SKIP {comparison} (manual export exists; use --force)")
                continue

            removed_inferred = remove_old_inferred_exports(comp_dir, dry_run=args.dry_run)
            if removed_inferred:
                verb = "would remove" if args.dry_run else "removed"
                print(f"  {verb.upper()} old inferred in {comparison}: {', '.join(removed_inferred)}")

            tables = build_pair_exports_from_lineage(lineage, t_a=t_a, t_b=t_b)

            summary = {
                "fov": fov,
                "comparison": comparison,
                "t1_tp": t_a,
                "t2_tp": t_b,
                "inference_method": INFERENCE_SOURCE_LINEAGE,
                "matched": len(
                    _dedupe_matched(tables.matched, clash_unresolved=tables.unresolved, pair_detail=comparison)
                ),
                "new": len(_dedupe_sorted(tables.new)),
                "lost": len(_dedupe_sorted(tables.lost)),
                "removed_t1": len(_dedupe_sorted(tables.removed_t1)),
                "removed_t2": len(_dedupe_sorted(tables.removed_t2)),
                "unresolved": len(_dedupe_unresolved(tables.unresolved)),
                "export_dir": str(export_dir),
            }
            manifest.append(summary)

            if args.dry_run:
                print(f"  DRY-RUN {comparison}: {summary}")
                continue

            try:
                meta = write_inferred_export(
                    tables,
                    export_dir=export_dir,
                    respan_root=respan_root,
                    fov=fov,
                    t_a=t_a,
                    t_b=t_b,
                    comparison=comparison,
                    lineage=lineage,
                    baseline_exports=baseline_exports,
                    symlink_tiffs=args.symlink_tiffs,
                    skip_tiffs=args.no_tiffs,
                    skip_validation=not args.validate,
                )
            except ValueError as exc:
                print(f"  FAILED {comparison}: {exc}", file=sys.stderr)
                summary["status"] = "validation_failed"
                manifest.append(summary)
                continue

            prov = meta.get("provenance_validation", {})
            print(
                f"  WROTE {comparison}: matched={meta['matched_count']} "
                f"new={meta['new_count']} lost={meta['lost_count']} "
                f"removed_t1={meta['removed_t1_count']} removed_t2={meta['removed_t2_count']} "
                f"unresolved={meta['unresolved_count']} "
                f"| provenance OK (T1={prov.get('t1_spine_count','?')} T2={prov.get('t2_spine_count','?')} spines) "
                f"-> {export_dir}"
            )
            summary["status"] = "ok"

    if manifest and not args.dry_run:
        manifest_path = results_root / "baseline_bridge_inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nManifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
