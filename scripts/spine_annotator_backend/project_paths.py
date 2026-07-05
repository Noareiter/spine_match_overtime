"""Project layout: D:/spine_match_overtime/{config,scripts,results}."""

from __future__ import annotations

import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = _BACKEND_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSUME_T1_T2_DIR = SCRIPTS_DIR / "assume_t1_t2_onPre"

_DEFAULT_TRACKING_SCRIPTS = Path(r"D:\learning_project_spines\scripts")


def tracking_scripts_root() -> Path:
    """Hybrid spine-matching utilities (track_hybrid, spine_matching_tool)."""
    raw = (
        os.environ.get("SPINE_TRACKING_SCRIPTS")
        or os.environ.get("SPINE_LEGACY_SCRIPTS")
        or _DEFAULT_TRACKING_SCRIPTS
    )
    return Path(raw).expanduser().resolve()
