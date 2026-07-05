"""Per-animal workspace config: one path → respan/ inside it (create or reuse)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import respan_bootstrap
from .project_paths import CONFIG_DIR, PROJECT_ROOT

ANNOTATOR_DIR = PROJECT_ROOT
DEFAULT_CONFIG_PATH = CONFIG_DIR / "annotator.json"

DEFAULT_TIMEPOINT_ORDER = [
    "pre-droplet",
    "mid-droplet",
    "end-droplet",
    "end-lever",
    "return to droplet",
]

WORKFLOW_PAIRS = [
    ("pre-droplet", "mid-droplet"),
    ("mid-droplet", "end-droplet"),
    ("end-droplet", "end-lever"),
    ("end-lever", "return to droplet"),
]


@dataclass
class AnimalConfig:
    animal_id: str = ""
    workspace: str = ""
    default_fov: int = 1
    fovs: List[int] = field(default_factory=list)
    folder_open_depth: int = 2
    timepoint_order: List[str] = field(default_factory=lambda: list(DEFAULT_TIMEPOINT_ORDER))
    active_timepoints: List[str] = field(default_factory=list)
    skip_orphan_phases: bool = False
    config_path: str = ""

    def is_configured(self) -> bool:
        return bool(self.workspace.strip())


def workflow_pairs_for_timepoints(timepoint_names: List[str]) -> List[tuple[str, str]]:
    """Consecutive pairs within the active timepoint list."""
    pairs: List[tuple[str, str]] = []
    for i in range(len(timepoint_names) - 1):
        pairs.append((timepoint_names[i], timepoint_names[i + 1]))
    return pairs


def resolve_active_timepoints(
    cfg: AnimalConfig,
    available_with_csv: List[str],
    *,
    requested: Optional[List[str]] = None,
    saved: Optional[List[str]] = None,
) -> List[str]:
    """Pick ordered active TPs: UI request > per-FOV save > config > all available."""
    order = list(cfg.timepoint_order) or list(available_with_csv)
    choose = requested or saved or list(cfg.active_timepoints) or None
    if not choose:
        out = [t for t in order if t in available_with_csv]
        for t in available_with_csv:
            if t not in out:
                out.append(t)
        return out
    choose_set = {str(t).strip() for t in choose if str(t).strip()}
    out = [t for t in order if t in choose_set and t in available_with_csv]
    for t in available_with_csv:
        if t in choose_set and t not in out:
            out.append(t)
    if not out:
        raise ValueError(
            "No selected timepoints have detection CSVs. "
            f"Available: {', '.join(available_with_csv) or '(none)'}"
        )
    return out


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_fovs(raw: object) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return sorted({_coerce_int(x, 0) for x in raw if _coerce_int(x, 0) > 0})
    if isinstance(raw, str):
        import re

        return sorted({int(x) for x in re.findall(r"\d+", raw)})
    return []


def load_config(config_path: Optional[Path | str] = None) -> AnimalConfig:
    path = Path(config_path or os.environ.get("SPINE_CONFIG") or DEFAULT_CONFIG_PATH)
    cfg = AnimalConfig(config_path=str(path.resolve()) if path.exists() else str(path))

    if path.is_file():
        raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        cfg.animal_id = str(raw.get("animal_id", "") or "").strip()
        # Single path field; legacy alias data_root
        cfg.workspace = str(raw.get("workspace") or raw.get("data_root") or "").strip()
        cfg.default_fov = _coerce_int(raw.get("default_fov"), 1)
        cfg.fovs = _parse_fovs(raw.get("fovs"))
        cfg.folder_open_depth = _coerce_int(raw.get("folder_open_depth"), 2)
        tps = raw.get("timepoint_order")
        if isinstance(tps, list) and tps:
            cfg.timepoint_order = [str(x).strip() for x in tps if str(x).strip()]
        active = raw.get("active_timepoints")
        if isinstance(active, list) and active:
            cfg.active_timepoints = [str(x).strip() for x in active if str(x).strip()]
        cfg.skip_orphan_phases = bool(raw.get("skip_orphan_phases", False))

    if os.environ.get("SPINE_WORKSPACE"):
        cfg.workspace = os.environ["SPINE_WORKSPACE"].strip()
    elif os.environ.get("SPINE_DATA_ROOT"):
        cfg.workspace = os.environ["SPINE_DATA_ROOT"].strip()
    if os.environ.get("SPINE_ANIMAL_ID"):
        cfg.animal_id = os.environ["SPINE_ANIMAL_ID"].strip()
    if os.environ.get("SPINE_DEFAULT_FOV"):
        cfg.default_fov = _coerce_int(os.environ["SPINE_DEFAULT_FOV"], cfg.default_fov)

    return cfg


def resolve_workspace_path(cfg: AnimalConfig) -> Path:
    """Expand workspace; relative paths are under D:/spine_match_overtime."""
    raw = str(cfg.workspace or "").strip()
    if not raw:
        raise ValueError("Set workspace in config/annotator.json")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def bootstrap(cfg: Optional[AnimalConfig] = None) -> respan_bootstrap.BootstrapReport:
    cfg = cfg or load_config()
    ws = resolve_workspace_path(cfg)
    fovs = cfg.fovs or ([cfg.default_fov] if cfg.default_fov > 0 else [])
    return respan_bootstrap.ensure_respan_layout(
        ws,
        timepoint_order=cfg.timepoint_order,
        fovs=fovs or None,
    )


def require_respan(cfg: Optional[AnimalConfig] = None) -> Path:
    report = bootstrap(cfg)
    return report.respan


def resolve_paths(cfg: AnimalConfig) -> Dict[str, str]:
    if not cfg.workspace:
        return {
            "animal_id": cfg.animal_id,
            "workspace": "",
            "respan_root": "",
            "results_root": "",
            "default_fov": str(cfg.default_fov),
            "folder_open_depth": str(cfg.folder_open_depth),
            "config_path": cfg.config_path,
        }
    try:
        report = bootstrap(cfg)
        respan = report.respan
        return {
            "animal_id": cfg.animal_id,
            "workspace": str(report.workspace),
            "respan_root": str(respan),
            "results_root": str(respan / "results"),
            "default_fov": str(cfg.default_fov),
            "folder_open_depth": str(cfg.folder_open_depth),
            "config_path": cfg.config_path,
            "bootstrap_created": len(report.created),
            "bootstrap_summary": report.summary(),
        }
    except Exception as exc:
        try:
            ws = str(resolve_workspace_path(cfg))
        except Exception:
            ws = str(Path(cfg.workspace).expanduser())
        return {
            "animal_id": cfg.animal_id,
            "workspace": ws,
            "respan_root": "",
            "results_root": "",
            "error": str(exc),
            "config_path": cfg.config_path,
        }


def require_respan_root(cfg: Optional[AnimalConfig] = None) -> Path:
    """Alias used by animal_layout and app routes."""
    return require_respan(cfg)


def list_newly_created_folders(report: respan_bootstrap.BootstrapReport) -> List[Path]:
    """Return absolute paths for folders created during bootstrap (skip existing)."""
    seen: set[str] = set()
    out: List[Path] = []
    for rel in report.created:
        p = (report.respan / rel).resolve()
        if not p.is_dir():
            continue
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def list_folders_to_open(cfg: AnimalConfig, *, max_depth: Optional[int] = None) -> List[Path]:
    """Legacy: walk respan tree. Prefer list_newly_created_folders after bootstrap."""
    report = bootstrap(cfg)
    depth = max(0, min(max_depth if max_depth is not None else cfg.folder_open_depth, 5))
    respan = report.respan
    workspace = report.workspace

    seen: set[str] = set()
    out: List[Path] = []

    def add(p: Path) -> None:
        if not p.is_dir():
            return
        key = str(p.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        out.append(p.resolve())

    if workspace.resolve() != respan.resolve():
        add(workspace)
    add(respan)

    def walk(p: Path, level: int) -> None:
        if level >= depth:
            return
        try:
            children = sorted((c for c in p.iterdir() if c.is_dir()), key=lambda x: x.name.lower())
        except OSError:
            return
        for child in children:
            if child.name.lower() in {"__pycache__", ".git"}:
                continue
            add(child)
            walk(child, level + 1)

    walk(respan, 0)
    return out
