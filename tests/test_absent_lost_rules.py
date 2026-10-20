"""
ABSENT / NEW / LOST lineage rules.

A valid lineage is: [ABSENT...] -> one continuous block of matches (NEW at its
first timepoint unless that is the first session timepoint) -> [LOST at the
first missing timepoint] -> [ABSENT...].

- LOST is written once, at the first missing timepoint; later ones are ABSENT.
- The empty timepoint after the last match is always LOST (never IGNORE from an
  OOF region - censoring is manual only).
- A lineage is never split: a gap between two matches blocks the save.
- Stored fates (IGNORE / ARTIFACT / censored / NEW) are dropped and re-inferred.

Run with: python -m unittest tests.test_absent_lost_rules -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend import spine_lineage_store as sls  # noqa: E402

TPS = ["T1", "T2", "T3", "T4", "T5"]
EMPTY: dict = {}


def _match(i: int) -> dict:
    return {"spine_id": f"S_{i:04d}", "x": 1.0, "y": 1.0, "z": 1.0}


def _statuses(seq):
    per_tp = {tp: dict(td) for tp, td in zip(TPS, seq)}
    out, _last, lost_at, _cens, _rc = sls.finalize_lineage_per_tp(
        per_tp, TPS, pre_spine_id=""
    )
    return [sls.status_for_tp_data(out[tp]) for tp in TPS], lost_at


class TestStatuses(unittest.TestCase):
    def test_lost_once_then_absent(self):
        statuses, lost_at = _statuses([_match(1), _match(2), EMPTY, EMPTY, EMPTY])
        self.assertEqual(statuses, ["matched", "matched", "lost", "absent", "absent"])
        self.assertEqual(lost_at, "T3")

    def test_absent_before_new(self):
        statuses, lost_at = _statuses([EMPTY, _match(2), _match(3), _match(4), _match(5)])
        self.assertEqual(statuses, ["absent", "new", "matched", "matched", "matched"])
        self.assertEqual(lost_at, "")

    def test_new_then_gap_is_lost_not_absent(self):
        statuses, lost_at = _statuses([EMPTY, _match(2), EMPTY, EMPTY, EMPTY])
        self.assertEqual(statuses, ["absent", "new", "lost", "absent", "absent"])
        self.assertEqual(lost_at, "T3")

    def test_legacy_propagated_lost_is_rederived(self):
        lost3 = {"fate": "lost", "source": "lost_inferred_after_T2"}
        lostp = {"fate": "lost", "source": "lost_propagated_after_T2"}
        statuses, lost_at = _statuses([_match(1), _match(2), lost3, dict(lostp), dict(lostp)])
        self.assertEqual(statuses, ["matched", "matched", "lost", "absent", "absent"])
        self.assertEqual(lost_at, "T3")

    def test_legacy_inferred_oof_ignore_becomes_lost(self):
        ign = {"fate": "ignore", "source": "ignore_inferred_after_T2"}
        ignp = {"fate": "ignore", "source": "ignore_propagated_after_T2"}
        statuses, _ = _statuses([_match(1), _match(2), ign, ignp, EMPTY])
        self.assertEqual(statuses, ["matched", "matched", "lost", "absent", "absent"])

    def test_gap_in_oof_region_is_lost(self):
        oof = [{
            "segment_id": "seg1", "pre_tp": "T1",
            "pre_shape": {"shape": "rect", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
            "per_tp": {tp: {"shape": "rect", "x0": 0, "y0": 0, "x1": 10, "y1": 10} for tp in TPS},
        }]
        per_tp = {tp: dict(td) for tp, td in zip(TPS, [_match(1), _match(2), EMPTY, EMPTY, EMPTY])}
        out, *_ = sls.finalize_lineage_per_tp(
            per_tp, TPS, pre_spine_id="", pre_tp="T1", oof_segments=oof
        )
        self.assertEqual(sls.status_for_tp_data(out["T3"]), "lost")
        self.assertEqual(sls.status_for_tp_data(out["T4"]), "absent")


    def test_split_lineage_keeps_anchor_block(self):
        # Conflict resolution / legacy data may split a lineage; finalize keeps
        # the block holding the anchor and releases the rest (bug 31).
        per_tp = {tp: dict(td) for tp, td in zip(TPS, [_match(1), EMPTY, _match(3), _match(4), EMPTY])}
        out, *_ = sls.finalize_lineage_per_tp(per_tp, TPS, pre_spine_id="", anchor_timepoint="T3")
        statuses = [sls.status_for_tp_data(out[tp]) for tp in TPS]
        self.assertEqual(statuses, ["absent", "absent", "new", "matched", "lost"])
        self.assertEqual(out["T1"].get("removed_spine_id"), "S_0001")


class TestSaveValidation(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="absent_lost_rules_"))

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def _save(self, seq):
        per_tp = {tp: dict(td) for tp, td in zip(TPS, seq)}
        return sls.save_decision(
            self._root, 1,
            animal_id="TEST", pre_spine_id="S_0001", pre_timepoint="T1",
            timepoint_names=TPS, per_tp=per_tp, lineage_key="S_0001",
        )

    def test_gap_between_matches_blocks_save(self):
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            self._save([_match(1), EMPTY, _match(3), EMPTY, EMPTY])
        self.assertEqual(sls.load_decisions(self._root, 1).get("lineages") or [], [])

    def test_artifact_inside_lineage_blocks_save(self):
        blind = {"fate": "artifact", "artifact_mode": "blind_spot"}
        with self.assertRaisesRegex(ValueError, "cannot be split"):
            self._save([_match(1), blind, _match(3), EMPTY, EMPTY])

    def test_lineage_level_censoring_is_dropped(self):
        # No lineage-level CENSORED / IGNORE (rules review 29): a legacy
        # blind-spot mark after the last match is dropped -> LOST then ABSENT.
        blind = {"fate": "artifact", "artifact_mode": "blind_spot"}
        self._save([_match(1), _match(2), EMPTY, blind, EMPTY])
        lin = sls.load_decisions(self._root, 1)["lineages"][0]
        statuses = [sls.status_for_tp_data(lin["per_tp"][tp]) for tp in TPS]
        self.assertEqual(statuses, ["matched", "matched", "lost", "absent", "absent"])

    def test_sticky_new_is_rederived(self):
        # Bug 27: a stored NEW must not block matching an earlier timepoint.
        new2 = {**_match(2), "fate": "new", "source": "inferred_new"}
        self._save([_match(1), new2, _match(3), EMPTY, EMPTY])
        lin = sls.load_decisions(self._root, 1)["lineages"][0]
        statuses = [sls.status_for_tp_data(lin["per_tp"][tp]) for tp in TPS]
        self.assertEqual(statuses, ["matched", "matched", "matched", "lost", "absent"])

    def test_legacy_ignore_before_first_match_is_dropped(self):
        # Bug 26 / 28: only ABSENT before the first match; no IGNORE propagation.
        ign = {"fate": "ignore", "source": "manual"}
        self._save([ign, _match(2), _match(3), EMPTY, EMPTY])
        lin = sls.load_decisions(self._root, 1)["lineages"][0]
        statuses = [sls.status_for_tp_data(lin["per_tp"][tp]) for tp in TPS]
        self.assertEqual(statuses, ["absent", "new", "matched", "lost", "absent"])

    def test_valid_lineage_saves(self):
        self._save([_match(1), _match(2), EMPTY, EMPTY, EMPTY])
        lin = sls.load_decisions(self._root, 1)["lineages"][0]
        self.assertEqual(lin["per_tp"]["T3"].get("fate"), "lost")
        self.assertFalse(lin["per_tp"]["T4"].get("fate"))
        self.assertFalse(lin["per_tp"]["T5"].get("fate"))


if __name__ == "__main__":
    unittest.main()
