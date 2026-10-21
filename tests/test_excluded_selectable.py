"""
Excluded detections (A/I tag, OOF, ignored dendrite) stay selectable.

Rule (2026-10-21): an excluded detection is never queued (review completes
without it), is drawn blue, and can still be picked into a lineage by hand.
Picking it is an explicit decision that supersedes the exclusion: Save
succeeds, an A/I tag on it is cleared, its disposition becomes lineage_*
and it leaves excluded_spines.csv.

Run with: python -m unittest tests.test_excluded_selectable -v
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures  # noqa: E402
from spine_annotator_backend import results_final_store, spine_qc_tag_store  # noqa: E402

TPS = ["pre-droplet", "mid-droplet", "end-droplet"]
HEAD = "spine_id,x,y,z,dendrite_id\n"
CSV = {
    "pre-droplet": HEAD + "1,100,100,2,1\n",
    "mid-droplet": HEAD + "1,101,100,2,1\n",
    "end-droplet": HEAD + "1,102,100,2,1\n",
}
END_GID = "S_00003"  # end-droplet spine 1


class TestExcludedSelectable(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="excluded_sel_"))
        self.respan = self._root / "respan"
        for tp in TPS:
            (self.respan / tp / "Tables").mkdir(parents=True)
            (self.respan / tp / "Tables" / "fov1_detected_spines.csv").write_text(CSV[tp], encoding="utf-8")
        _fixtures.add_tiffs(self.respan)
        cfg = {"animal_id": "T", "workspace": str(self._root), "default_fov": 1,
               "fovs": [1], "timepoint_order": TPS, "active_timepoints": []}
        cfg_path = self._root / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        self._prior = os.environ.get("SPINE_CONFIG")
        os.environ["SPINE_CONFIG"] = str(cfg_path)
        from fastapi.testclient import TestClient
        from spine_annotator_backend.app import app
        self.client = TestClient(app)

    def tearDown(self):
        if self._prior is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior
        shutil.rmtree(self._root, ignore_errors=True)

    def _load_and_select(self):
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": r.json()["spine_ids"][0]})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _marker(self, tp, gid):
        p = self.client.post("/mtp/viewer/panel", json={"timepoint": tp, "mode": "zoom", "x": 102.0, "y": 100.0, "z": 2})
        self.assertEqual(p.status_code, 200, p.text)
        return next((m for m in p.json().get("markers", []) if m["spine_id"] == gid), None)

    def _disposition(self, tp, gid):
        rows, _ = __import__("spine_annotator_backend.spine_qc_store", fromlist=["x"]).build_disposition_rows(
            self.respan, 1, animal_id="T", timepoint_names=TPS
        )
        return next(r["disposition"] for r in rows if r["timepoint"] == tp and r["global_spine_id"] == gid)

    def test_artifact_tagged_spine_can_be_selected_and_saved(self):
        self._load_and_select()
        spine_qc_tag_store.save_tag(self.respan, 1, timepoint="end-droplet", spine_id=END_GID, tag="artifact", animal_id="T")
        j = self._load_and_select()
        self.assertIsNone(j["positions"]["end-droplet"]["spine_id"])  # not suggested
        m = self._marker("end-droplet", END_GID)
        self.assertIsNotNone(m, "excluded spine must still be drawn")
        self.assertTrue(m.get("excluded"))
        r = self.client.post("/mtp/viewer/set-spine", json={"timepoint": "end-droplet", "spine_id": END_GID})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("excluded", (r.json().get("message") or "").lower())
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)
        tags = spine_qc_tag_store.load_tags(self.respan, 1)
        self.assertEqual(spine_qc_tag_store.get_tag(tags, "end-droplet", END_GID), "")
        self.assertEqual(self._disposition("end-droplet", END_GID), "lineage_matched")
        rows = results_final_store._excluded_spine_rows_for_fov(self.respan, 1, "T")
        self.assertEqual([r for r in rows if r["global_spine_id"] == END_GID], [])

    def test_oof_spine_can_be_selected_and_wins_in_disposition(self):
        self._load_and_select()
        r = self.client.post("/mtp/viewer/oof-segment", json={
            "timepoint": "end-droplet", "shape": "rect", "x0": 95, "y0": 95, "x1": 110, "y1": 105,
        })
        self.assertEqual(r.status_code, 200, r.text)
        self._load_and_select()
        m = self._marker("end-droplet", END_GID)
        self.assertIsNotNone(m)
        self.assertTrue(m.get("excluded"))
        # Click on the image (focus snap) picks it too.
        r = self.client.post("/mtp/viewer/focus", json={"timepoint": "end-droplet", "stack_x": 102.0, "stack_y": 100.0, "z_plane": 2})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["positions"]["end-droplet"]["spine_id"], END_GID)
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._disposition("end-droplet", END_GID), "lineage_matched")
        rows = results_final_store._excluded_spine_rows_for_fov(self.respan, 1, "T")
        self.assertEqual([r for r in rows if r["global_spine_id"] == END_GID], [])

    def test_excluded_spine_is_not_required_for_review_complete(self):
        self._load_and_select()
        spine_qc_tag_store.save_tag(self.respan, 1, timepoint="end-droplet", spine_id=END_GID, tag="ignore", animal_id="T")
        self._load_and_select()
        self.assertEqual(self._disposition("end-droplet", END_GID), "qc_ignore")


if __name__ == "__main__":
    unittest.main()
