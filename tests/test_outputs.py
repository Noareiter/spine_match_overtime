"""
Output files after the October 2026 outputs review.

- spine_registry_wide.csv: no fate_ / artifact_mode_ columns; x/y/z only where
  a spine was observed.
- spine_observations_long.csv: one row per lineage x timepoint, RESPAN
  measures filled for observed spines, empty at absent / lost.
- spine_disposition.csv: no status / fate / event columns.
- lineage_decisions.json: no censored_from_tp / right_censored.
- fov_inventory.csv: timepoint rows only, no workflow-pair columns.

Run with: python -m unittest tests.test_outputs -v
"""

from __future__ import annotations

import csv
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

TPS = ["pre-droplet", "mid-droplet", "end-droplet"]
CSV = {
    "pre-droplet": "spine_id,x,y,z,dendrite_id,spine_vol,head_vol\n1,100,100,2,1,0.5,0.2\n",
    "mid-droplet": "spine_id,x,y,z,dendrite_id,spine_vol,head_vol\n7,101,100,2,1,0.6,0.3\n",
    "end-droplet": "spine_id,x,y,z,dendrite_id,spine_vol,head_vol\n9,180,180,2,1,0.9,0.4\n",
}


def _read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="outputs_"))
        self.respan = self._root / "respan"
        for tp in TPS:
            (self.respan / tp / "Tables").mkdir(parents=True)
            (self.respan / tp / "Tables" / "fov1_detected_spines.csv").write_text(CSV[tp], encoding="utf-8")
        _fixtures.add_tiffs(self.respan)
        config = {"animal_id": "T", "workspace": str(self._root), "default_fov": 1,
                  "fovs": [1], "timepoint_order": TPS, "active_timepoints": []}
        cfg_path = self._root / "config.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        self._prior = os.environ.get("SPINE_CONFIG")
        os.environ["SPINE_CONFIG"] = str(cfg_path)
        from fastapi.testclient import TestClient
        from spine_annotator_backend.app import app
        self.client = TestClient(app)
        self.meta = self.respan / "_annotator" / "fov1"

    def tearDown(self):
        if self._prior is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior
        shutil.rmtree(self._root, ignore_errors=True)

    def _save_pre_mid_lineage(self):
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": "S_00001"})
        self.client.post("/mtp/viewer/clear-tp", json={"timepoint": "end-droplet"})
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200, r.text)

    def test_registry_columns_and_coordinates(self):
        self._save_pre_mid_lineage()
        rows = _read(self.meta / "spine_registry_wide.csv")
        header = list(rows[0].keys())
        self.assertFalse(any(h.startswith(("fate_", "artifact_mode_")) for h in header))
        row = [r for r in rows if r["lineage_key"] == "S_00001"][0]
        self.assertEqual([row[f"status_{tp}"] for tp in TPS], ["matched", "matched", "lost"])
        self.assertEqual(row["mid-droplet_x"], "101.0")
        self.assertEqual(row["end-droplet_x"], "")

    def test_observations_long_has_measures(self):
        self._save_pre_mid_lineage()
        rows = _read(self.meta / "spine_observations_long.csv")
        by_tp = {r["timepoint"]: r for r in rows if r["lineage_key"] == "S_00001"}
        self.assertEqual(by_tp["mid-droplet"]["local_spine_id"], "7")
        self.assertEqual(by_tp["mid-droplet"]["spine_vol"], "0.6")
        self.assertEqual(by_tp["end-droplet"]["status"], "lost")
        self.assertEqual(by_tp["end-droplet"]["spine_vol"], "")

    def test_disposition_lineage_json_and_inventory_are_clean(self):
        self._save_pre_mid_lineage()
        header = list(_read(self.meta / "spine_disposition.csv")[0].keys())
        for gone in ("status", "fate", "event"):
            self.assertNotIn(gone, header)
        lin = json.loads((self.meta / "lineage_decisions.json").read_text(encoding="utf-8"))["lineages"][0]
        self.assertNotIn("censored_from_tp", lin)
        self.assertNotIn("right_censored", lin)
        inv = _read(self.meta / "fov_inventory.csv")
        self.assertEqual([r["timepoint"] for r in inv], TPS)
        self.assertNotIn("pair_label", inv[0])


if __name__ == "__main__":
    unittest.main()
