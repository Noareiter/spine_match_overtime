"""
Clear in Lineage mode (user rule 2026-10-20).

Clearing a timepoint after the spine has appeared clears every later timepoint
too: the spine disappeared there, and a lineage is never split. Clearing
before the first match clears only that timepoint (the spine starts later).

Run with: python -m unittest tests.test_clear_tail -v
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


class TestClearTail(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="clear_tail_"))
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

    def _start(self):
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200, r.text)
        sid = r.json()["spine_ids"][0]
        r = self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": sid})
        return r.json()["positions"]

    def test_clear_after_first_match_clears_the_tail(self):
        pos = self._start()
        self.assertTrue(all(pos[tp]["spine_id"] for tp in TPS))
        r = self.client.post("/mtp/viewer/clear-tp", json={"timepoint": "mid-droplet"})
        pos = r.json()["positions"]
        self.assertTrue(pos["pre-droplet"]["spine_id"])
        self.assertIsNone(pos["mid-droplet"]["spine_id"])
        self.assertIsNone(pos["end-droplet"]["spine_id"])
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)

    def test_clear_before_first_match_clears_only_that_timepoint(self):
        self._start()
        r = self.client.post("/mtp/viewer/clear-tp", json={"timepoint": "pre-droplet"})
        pos = r.json()["positions"]
        self.assertIsNone(pos["pre-droplet"]["spine_id"])
        self.assertTrue(pos["mid-droplet"]["spine_id"])
        self.assertTrue(pos["end-droplet"]["spine_id"])


    def test_oof_on_open_panel_spine_clears_it_and_save_refuses_it(self):
        pos = self._start()
        mid = pos["mid-droplet"]
        r = self.client.post("/mtp/viewer/oof-segment", json={
            "timepoint": "mid-droplet", "shape": "rect",
            "x0": mid["x"] - 5, "y0": mid["y"] - 5, "x1": mid["x"] + 5, "y1": mid["y"] + 5,
        })
        self.assertEqual(r.status_code, 200, r.text)
        state = self.client.get("/mtp/viewer/state").json()
        self.assertIsNone(state["positions"]["mid-droplet"]["spine_id"])
        # Even if the client puts it back, Save refuses an unavailable spine.
        r = self.client.post("/mtp/viewer/set-spine", json={"timepoint": "mid-droplet", "spine_id": mid["spine_id"]})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
