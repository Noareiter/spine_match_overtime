"""
OOF applies only to the timepoint it was drawn on (rules review 12).

- A new segment covers its drawn timepoint only, every dendrite inside it.
- Legacy segments (no drawn_tp) cover their per_tp timepoints only; their
  pre_shape is a projection and does not apply to pre_tp unless per_tp is empty.
- A shape cannot be edited onto a timepoint the segment does not cover.

Run with: python -m unittest tests.test_oof_single_timepoint -v
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

from spine_annotator_backend import oof_segment_store as oof  # noqa: E402

BOX = oof.rect_shape(0, 0, 10, 10)


class TestOofSingleTimepoint(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="oof_single_tp_"))

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def test_new_segment_covers_drawn_timepoint_only(self):
        seg = oof.upsert_segment(
            self._root, 1, animal_id="T", segment_id="s1", dendrite_id_pre="2",
            pre_tp="end-lever", pre_shape=BOX, per_tp={"end-lever": BOX}, drawn_tp="end-lever",
        )
        segs = [seg]
        self.assertTrue(oof.coords_in_oof(5, 5, "end-lever", segs, dendrite_id="99"))
        self.assertFalse(oof.coords_in_oof(5, 5, "pre-droplet", segs))
        self.assertFalse(oof.coords_in_oof(5, 5, "end-droplet", segs))

    def test_legacy_segment_uses_per_tp_not_pre_shape(self):
        legacy = {"segment_id": "old", "pre_tp": "pre-droplet", "pre_shape": BOX,
                  "per_tp": {"end-lever": BOX}}
        self.assertEqual(oof.segment_timepoints(legacy), ["end-lever"])
        self.assertFalse(oof.coords_in_oof(5, 5, "pre-droplet", [legacy]))
        only_pre = {"segment_id": "old2", "pre_tp": "pre-droplet", "pre_shape": BOX, "per_tp": {}}
        self.assertEqual(oof.segment_timepoints(only_pre), ["pre-droplet"])

    def test_cannot_edit_shape_onto_other_timepoint(self):
        oof.upsert_segment(
            self._root, 1, animal_id="T", segment_id="s1", dendrite_id_pre="2",
            pre_tp="end-lever", pre_shape=BOX, per_tp={"end-lever": BOX}, drawn_tp="end-lever",
        )
        with self.assertRaises(ValueError):
            oof.update_tp_shape(self._root, 1, animal_id="T", segment_id="s1",
                                tp="pre-droplet", shape=BOX)


if __name__ == "__main__":
    unittest.main()
