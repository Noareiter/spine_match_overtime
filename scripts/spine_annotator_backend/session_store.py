from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from typing import List
from typing import Set
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LoadedSession:
    t1_tiff_path: Path
    t2_tiff_path: Path
    t1_csv_path: Path
    t2_csv_path: Path
    t1_stack: np.ndarray
    t2_stack: np.ndarray
    t1_df: pd.DataFrame
    t2_df: pd.DataFrame
    t1_lookup: Dict[str, Dict[str, object]] = field(default_factory=dict)
    t2_lookup: Dict[str, Dict[str, object]] = field(default_factory=dict)
    review_decisions: Dict[str, Dict[str, object]] = field(default_factory=dict)
    dendrite_links: List[Dict[str, object]] = field(default_factory=list)
    algo_matches: Dict[str, Dict[str, object]] = field(default_factory=dict)
    confirmed_matches: List[Dict[str, str]] = field(default_factory=list)
    matched_t1_ids: Set[str] = field(default_factory=set)
    matched_t2_ids: Set[str] = field(default_factory=set)
    rejected_pairs: Set[str] = field(default_factory=set)  # key format: "t1_id|t2_id"
    lost_t1_ids: Set[str] = field(default_factory=set)
    new_t2_ids: Set[str] = field(default_factory=set)
    removed_t1_ids: Set[str] = field(default_factory=set)
    removed_t2_ids: Set[str] = field(default_factory=set)
    ignored_t1_ids: Set[str] = field(default_factory=set)
    ignored_t2_ids: Set[str] = field(default_factory=set)
    # Unlinked-dendrite spines the user has visually acknowledged in Final Spine Step.
    reviewed_unlinked_t1_ids: Set[str] = field(default_factory=set)
    reviewed_unlinked_t2_ids: Set[str] = field(default_factory=set)
    # Optional explicit QC on unlinked spines: key "t1:<id>" / "t2:<id>" -> "artifact" | "ignore".
    unlinked_spine_dispositions: Dict[str, str] = field(default_factory=dict)
    manual_t2_click_spines: List[Dict[str, object]] = field(default_factory=list)
    manual_t1_click_spines: List[Dict[str, object]] = field(default_factory=list)
    manual_click_matches: List[Dict[str, object]] = field(default_factory=list)
    viewer_state: Dict[str, int] = field(default_factory=lambda: {"modal_slice_t1": 10, "modal_slice_t2": 10})
    max_match_z_gap: float = 7.0
    action_history: List[Dict[str, Any]] = field(default_factory=list)


_ACTIVE_SESSION: Optional[LoadedSession] = None


def set_active_session(session: LoadedSession) -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session


def get_active_session() -> Optional[LoadedSession]:
    return _ACTIVE_SESSION


def clear_active_session() -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = None


def require_active_session() -> LoadedSession:
    if _ACTIVE_SESSION is None:
        raise RuntimeError("No active session. Call /load-session first.")
    return _ACTIVE_SESSION

