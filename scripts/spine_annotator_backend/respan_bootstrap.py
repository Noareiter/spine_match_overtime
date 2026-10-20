"""Ensure respan/ layout exists under a single workspace path (no searching elsewhere)."""

from __future__ import annotations

import re
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
) -> BootstrapReport:
    """Create missing timepoint / Tables folders under respan."""
    ws = workspace.expanduser().resolve()
    respan = resolve_respan_in_workspace(ws)
    report = BootstrapReport(workspace=ws, respan=respan)

    for tp in timepoint_order:
        tp_dir = respan / tp
        _mkdir(tp_dir, report)
        _mkdir(tp_dir / "Tables", report)

    fov_list = sorted(set(fovs or [])) or discover_fovs(respan)

    for tp in timepoint_order:
        tp_dir = respan / tp
        for fov in fov_list:
            _mkdir(tp_dir / f"fov{fov}Swc", report)

    if not fov_list:
        report.warnings.append(
            "No FOVs yet (add Tables/fovN_detected_spines.csv or set fovs in config)."
        )

    return report
