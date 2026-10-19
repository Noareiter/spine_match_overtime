"""Project layout: D:/spine_match_overtime/{config,scripts,results}."""

from __future__ import annotations

from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = _BACKEND_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"

VENDOR_DIR = _BACKEND_DIR / "vendor"


def tracking_scripts_root() -> Path:
    """Vendored hybrid spine-matching utilities (track_hybrid, spine_matching_utils)."""
    return VENDOR_DIR
