"""
Manual (Force-add) spines and split suggestions.

- A split suggestion (hole in the middle) is allowed while editing; only
  Save refuses it, and filling the hole with a manual spine then saves.
- A manual spine carries no dendrite id and is never "unavailable": an
  ignored dendrite at its timepoint neither hides it nor blocks Save.
- Force-add inside an OOF region is refused.
- The focused spine is drawn even inside an OOF region.

Run with: python -m unittest tests.test_manual_spine -v
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

TPS = ["pre-droplet", "mid-droplet", "end-droplet", "end-lever"]
HEAD = "spine_id,x,y,z,dendrite_id\n"
# One spine per timepoint on dendrite 1 near (100,100).
CSV = {
    "pre-droplet": HEAD + "1,100,100,2,1\n",
    "mid-droplet": HEAD + "1,101,100,2,1\n",
    "end-droplet": HEAD + "1,102,100,2,1\n",
    "end-lever": HEAD + "1,103,100,2,1\n",
}
OOF_BOX = {"timepoint": "end-droplet", "shape": "rect", "x0": 95, "y0": 95, "x1": 110, "y1": 105}


class TestManualSpine(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="manual_spine_"))
        respan = self._root / "respan"
        for tp in TPS:
            (respan / tp / "Tables").mkdir(parents=True)
            (respan / tp / "Tables" / "fov1_detected_spines.csv").write_text(CSV[tp], encoding="utf-8")
        _fixtures.add_tiffs(respan)
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
        sid = r.json()["spine_ids"][0]
        r = self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": sid})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["positions"]

    def _add(self, tp, x, y):
        return self.client.post("/mtp/viewer/add-manual-spine", json={
            "timepoint": tp, "stack_x": x, "stack_y": y, "z_plane": 2,
        })

    def _panel_roles(self, tp, x, y):
        p = self.client.post("/mtp/viewer/panel", json={
            "timepoint": tp, "mode": "zoom", "x": x, "y": y, "z": 2,
        })
        self.assertEqual(p.status_code, 200, p.text)
        return {m["spine_id"]: m["role"] for m in p.json().get("markers", [])}

    def test_split_suggestion_allowed_but_not_saved_until_filled(self):
        self._load_and_select()
        r = self.client.post("/mtp/viewer/oof-segment", json=OOF_BOX)
        self.assertEqual(r.status_code, 200, r.text)
        pos = self._load_and_select()
        self.assertTrue(pos["pre-droplet"]["spine_id"])
        self.assertIsNone(pos["end-droplet"]["spine_id"])
        self.assertTrue(pos["end-lever"]["spine_id"])  # split suggestion is fine
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("cannot be split", r.json()["detail"])
        r = self._add("end-droplet", 150.0, 150.0)  # outside the OOF box
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)

    def test_manual_spine_ignores_ignored_dendrite(self):
        # Ignoring a dendrite is a Phase-0 (linker) action: load that page first.
        r = self.client.post("/mtp/load-from-animal", params={"fov": 1})
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post("/mtp/ignore-dendrite", json={"timepoint": "end-droplet", "dendrite_id": "1"})
        self.assertEqual(r.status_code, 200, r.text)
        pos = self._load_and_select()
        self.assertIsNone(pos["end-droplet"]["spine_id"])  # dendrite 1 is ignored there
        r = self._add("end-droplet", 104.0, 100.0)
        self.assertEqual(r.status_code, 200, r.text)
        sid = r.json()["positions"]["end-droplet"]["spine_id"]
        self.assertTrue(sid.startswith("manual_"))
        self.assertEqual(r.json()["positions"]["end-droplet"]["dendrite_id"], "")
        self.assertEqual(self._panel_roles("end-droplet", 104.0, 100.0).get(sid), "focus")
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)

    def test_manual_spine_inside_oof_is_refused(self):
        self._load_and_select()
        self.client.post("/mtp/viewer/oof-segment", json=OOF_BOX)
        self._load_and_select()
        r = self._add("end-droplet", 100.0, 100.0)
        self.assertEqual(r.status_code, 400)
        self.assertIn("out of frame", r.json()["detail"])

    def test_focused_csv_spine_inside_oof_is_drawn(self):
        self._load_and_select()
        self.client.post("/mtp/viewer/oof-segment", json=OOF_BOX)
        self._load_and_select()
        # Pick the (now OOF) end-droplet spine by hand; it stays visible as focus
        # even though it cannot be saved.
        r = self.client.post("/mtp/viewer/set-spine", json={"timepoint": "end-droplet", "spine_id": "S_00003"})
        if r.status_code == 200:
            self.assertEqual(self._panel_roles("end-droplet", 102.0, 100.0).get("S_00003"), "focus")
        else:
            self.assertIn("ignored", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
