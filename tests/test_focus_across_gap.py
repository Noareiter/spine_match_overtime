"""
/focus always moves the panel, and may snap across a gap.

A gap is allowed while editing (2026-10-21): clicking near a spine at a
timepoint after a gap snaps to it like anywhere else. Only Save refuses the
split lineage, with the reason.

Run with: python -m unittest tests.test_focus_across_gap -v
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

CSV = "spine_id,x,y,z,dendrite_id\n1,100.0,100.0,2.0,1\n"
TPS = ["pre-droplet", "mid-droplet", "end-droplet"]


class TestFocusAcrossGap(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="focus_gap_"))
        respan = self._root / "respan"
        for tp in TPS:
            (respan / tp / "Tables").mkdir(parents=True)
            (respan / tp / "Tables" / "fov1_detected_spines.csv").write_text(CSV, encoding="utf-8")
        _fixtures.add_tiffs(respan)
        config = {"animal_id": "T", "workspace": str(self._root), "default_fov": 1,
                  "fovs": [1], "timepoint_order": TPS, "active_timepoints": []}
        self._config_path = self._root / "config.json"
        self._config_path.write_text(json.dumps(config), encoding="utf-8")
        self._prior = os.environ.get("SPINE_CONFIG")
        os.environ["SPINE_CONFIG"] = str(self._config_path)
        from fastapi.testclient import TestClient
        from spine_annotator_backend.app import app
        self.client = TestClient(app)

    def tearDown(self):
        if self._prior is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior
        shutil.rmtree(self._root, ignore_errors=True)

    def test_focus_after_gap_snaps_and_save_refuses_the_split(self):
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200, r.text)
        sid = r.json()["spine_ids"][0]
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": sid})
        self.client.post("/mtp/viewer/clear-tp", json={"timepoint": "mid-droplet"})
        self.client.post("/mtp/viewer/clear-tp", json={"timepoint": "end-droplet"})
        r = self.client.post("/mtp/viewer/focus", json={
            "timepoint": "end-droplet", "stack_x": 101.0, "stack_y": 100.0, "z_plane": 2,
        })
        self.assertEqual(r.status_code, 200, r.text)
        end = r.json()["positions"]["end-droplet"]
        self.assertTrue(end.get("spine_id"))
        self.assertEqual(end.get("source"), "click_nearest")
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("cannot be split", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
