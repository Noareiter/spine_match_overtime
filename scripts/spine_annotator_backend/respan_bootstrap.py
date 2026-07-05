"""Ensure respan/ layout exists under a single workspace path (no searching elsewhere)."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DETECTED_SPINES_RE = re.compile(r"^fov(\d+).*detected_spines.*\.csv$", re.IGNORECASE)


@dataclass
class BootstrapReport:
    workspace: Path
    respan: Path
    created: List[str] = field(default_factory=list)
    existing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"workspace: {self.workspace}", f"respan: {self.respan}"]
        if self.created:
            parts.append(f"created {len(self.created)} folder(s)")
        else:
            parts.append("structure OK (nothing new created)")
        return " | ".join(parts)


def _import_infer_module():
    from .project_paths import ASSUME_T1_T2_DIR

    infer_dir = ASSUME_T1_T2_DIR
    infer_path = infer_dir / "infer_baseline_bridged_pairs.py"
    if not infer_path.is_file():
        return None
    infer_dir_str = str(infer_dir)
    if infer_dir_str not in sys.path:
        sys.path.insert(0, infer_dir_str)
    spec = importlib.util.spec_from_file_location("infer_bootstrap", infer_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("infer_bootstrap", mod)
    spec.loader.exec_module(mod)
    return mod


def chronological_comparisons(timepoint_order: List[str]) -> List[str]:
    infer = _import_infer_module()
    if infer is not None:
        out: List[str] = []
        seen: set[str] = set()
        tp_order = getattr(infer, "TP_ORDER", timepoint_order)
        comp_fn = getattr(infer, "comparison_folder_name", None)
        if comp_fn is not None:
            for i, t_a in enumerate(tp_order):
                for t_b in tp_order[i + 1 :]:
                    name = comp_fn(t_a, t_b)
                    if name not in seen:
                        seen.add(name)
                        out.append(name)
            return out
    # Fallback: simple hyphenated pair names
    out = []
    for i, t_a in enumerate(timepoint_order):
        for t_b in timepoint_order[i + 1 :]:
            out.append(f"{t_a} - {t_b}")
    return out


def discover_fovs(respan: Path) -> List[int]:
    fovs: set[int] = set()
    for child in respan.iterdir():
        if not child.is_dir() or child.name.lower() == "results":
            continue
        tables = child / "Tables"
        if not tables.is_dir():
            continue
        for csv in tables.glob("*.csv"):
            m = DETECTED_SPINES_RE.match(csv.name)
            if m:
                fovs.add(int(m.group(1)))
    return sorted(fovs)


def resolve_respan_in_workspace(workspace: Path) -> Path:
    """
    Single rule — no searching outside workspace:
      1. workspace/respan/ exists  → use it
      2. workspace is named respan → use workspace itself
      3. otherwise                 → create workspace/respan/
    """
    ws = workspace.expanduser().resolve()
    if not ws.is_dir():
        raise FileNotFoundError(
            f"Workspace folder does not exist: {ws}\n"
            f"Create it first, then point config/annotator.json workspace to it."
        )
    if ws.name.lower() == "respan":
        return ws
    respan = ws / "respan"
    if respan.is_dir():
        return respan.resolve()
    respan.mkdir(parents=True, exist_ok=True)
    return respan.resolve()


def _mkdir(path: Path, report: BootstrapReport) -> None:
    rel = str(path.relative_to(report.respan))
    if path.is_dir():
        if rel != ".":
            report.existing.append(rel)
        return
    path.mkdir(parents=True, exist_ok=True)
    report.created.append(rel)


def ensure_respan_layout(
    workspace: Path,
    *,
    timepoint_order: List[str],
    fovs: Optional[List[int]] = None,
    ensure_results_skeleton: bool = True,
) -> BootstrapReport:
    """Create missing timepoint / Tables / results folders under respan."""
    ws = workspace.expanduser().resolve()
    respan = resolve_respan_in_workspace(ws)
    report = BootstrapReport(workspace=ws, respan=respan)

    for tp in timepoint_order:
        tp_dir = respan / tp
        _mkdir(tp_dir, report)
        _mkdir(tp_dir / "Tables", report)

    _mkdir(respan / "results", report)

    fov_list = sorted(set(fovs or [])) or discover_fovs(respan)

    for tp in timepoint_order:
        tp_dir = respan / tp
        for fov in fov_list:
            _mkdir(tp_dir / f"fov{fov}Swc", report)

    if ensure_results_skeleton and fov_list:
        comparisons = chronological_comparisons(timepoint_order)
        for fov in fov_list:
            fov_dir = respan / "results" / f"fov{fov}"
            _mkdir(fov_dir, report)
            for comp in comparisons:
                _mkdir(fov_dir / comp, report)
    elif ensure_results_skeleton and not fov_list:
        report.warnings.append(
            "No FOVs yet (add Tables/fovN_detected_spines.csv or set fovs in config)."
        )

    return report
