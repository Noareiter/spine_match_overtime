"""
Suggestion rules (rules review 19 / 20 / 20a / 23).

- With Phase-0 links: only linked dendrite pairs are candidates; never cross.
- With links that give no candidate: no suggestion (no unrestricted fallback).
- With no links at all: the whole FOV is searched.
- Local-registration nearest search respects MAX_MATCH_Z_GAP and links.

Run with: python -m unittest tests.test_suggestion_rules -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend import baseline_adapter  # noqa: E402
from spine_annotator_backend import mtp_spine_matching as m  # noqa: E402


def _frame(rows):
    df = pd.DataFrame(rows)
    for key in baseline_adapter._hybrid.FEATURE_KEYS:
        df[key] = float("nan")
    return df

PRE = _frame([{"id": "1", "x": 10.0, "y": 10.0, "z": 2.0, "dendrite_id": "1"}])
MID = _frame([
    {"id": "10", "x": 11.0, "y": 10.0, "z": 2.0, "dendrite_id": "7"},   # different dendrite, close
    {"id": "11", "x": 60.0, "y": 60.0, "z": 2.0, "dendrite_id": "3"},   # linked dendrite, far
])
LINK_1_3 = [{"link_id": "L1", "members": {"pre": ["1"], "mid": ["3"]}}]
LINK_1_9 = [{"link_id": "L1", "members": {"pre": ["1"], "mid": ["9"]}}]


def _best(links):
    return m._score_pairwise_best(
        "1", PRE, MID, cross_links=links, t1_tp="pre", t2_tp="mid", allow_cross=True
    )


class TestSuggestionRules(unittest.TestCase):
    def test_links_restrict_even_when_cross_is_closer(self):
        hit = _best(LINK_1_3)
        self.assertEqual(hit["spine_id"], "11")
        self.assertFalse(hit["cross_dendrite"])

    def test_links_without_candidate_give_no_suggestion(self):
        self.assertIsNone(_best(LINK_1_9))

    def test_no_links_searches_whole_fov(self):
        self.assertEqual(_best([])["spine_id"], "10")

    def test_pre_mid_queue_has_no_cross_queue(self):
        main, cross = m.build_pre_mid_queues(
            PRE, MID, LINK_1_9, pre_tp="pre", mid_tp="mid"
        )
        self.assertEqual(main, [])
        self.assertEqual(cross, [])

    def test_local_registration_respects_z_gate(self):
        lookup = {
            "pre": {"1": {"x": 10.0, "y": 10.0, "z": 2.0, "dendrite_id": "1"}},
            "mid": {
                "10": {"x": 10.0, "y": 10.0, "z": 2.0, "dendrite_id": "1"},
                "20": {"x": 30.0, "y": 30.0, "z": 2.0, "dendrite_id": "1"},
                "21": {"x": 30.5, "y": 30.0, "z": 20.0, "dendrite_id": "1"},
            },
            "end": {
                "21": {"x": 10.2, "y": 10.0, "z": 20.0, "dendrite_id": "1"},
                "22": {"x": 40.0, "y": 40.0, "z": 2.5, "dendrite_id": "1"},
            },
        }
        positions = {
            "pre": {"spine_id": "1", "x": 10.0, "y": 10.0, "z": 2.0, "source": "pre_base"},
            "mid": {"spine_id": "10", "x": 10.0, "y": 10.0, "z": 2.0, "source": "algo_chain"},
            "end": {"spine_id": None, "x": 10.0, "y": 10.0, "z": 2.0, "source": "coord_region"},
        }
        out, _, _ = m.apply_local_registration(
            positions, pre_tp="pre", pre_spine_id="1", timepoint_names=["pre", "mid", "end"],
            spine_lookup=lookup, cross_links=[], linked_dendrites_fn=lambda *a: set(),
        )
        # "21" is nearest in XY but 18 z-planes away: must be skipped.
        self.assertEqual(out["end"]["spine_id"], "22")


if __name__ == "__main__":
    unittest.main()
