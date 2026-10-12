"""
Feature test: ignoring a whole dendrite in the Phase 0 dendrite-link step
excludes every spine on it from any match candidacy, and lists them in the
merged ignored_spines.csv under source "dendrite".

Covers:
  - POST /mtp/ignore-dendrite persists the (timepoint, dendrite_id) pair and
    is reflected in /mtp/state's ignored_dendrites map.
  - Match ranking (build_lineage_positions et al., via /mtp/viewer/select-spine)
    never offers a spine on an ignored dendrite - it falls through to the
    next-best free candidate.
  - Manual override (/mtp/viewer/set-spine) is hard-blocked for an ignored
    spine, not just skipped by ranking.
  - results_final_store._ignored_spine_rows_for_fov lists the ignored
    dendrite's spines with source "dendrite".

Uses a small synthetic 2-timepoint fixture (no external repo dependency).

Run with: python -m unittest tests.test_ignore_whole_dendrite -v
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

MID_CSV = """spine_id,x,y,z,dendrite_id
1,100.0,100.0,5.0,1
"""

# dendrite 2 (spines 15, 16) will be ignored; dendrite 5 (spine 20) stays free.
END_CSV = """spine_id,x,y,z,dendrite_id
15,101.0,101.0,5.0,2
16,103.0,103.0,5.0,2
20,300.0,300.0,5.0,5
"""


class TestIgnoreWholeDendrite(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="ignore_dendrite_"))
        respan = self._root / "respan"
        (respan / "mid-droplet" / "Tables").mkdir(parents=True)
        (respan / "end-droplet" / "Tables").mkdir(parents=True)
        (respan / "mid-droplet" / "Tables" / "fov1_detected_spines.csv").write_text(
            MID_CSV, encoding="utf-8"
        )
        (respan / "end-droplet" / "Tables" / "fov1_detected_spines.csv").write_text(
            END_CSV, encoding="utf-8"
        )
        config = {
            "animal_id": "TESTANIMAL",
            "workspace": str(self._root),
            "default_fov": 1,
            "fovs": [1],
            "timepoint_order": ["mid-droplet", "end-droplet"],
            "active_timepoints": [],
        }
        self._config_path = self._root / "config.json"
        self._config_path.write_text(json.dumps(config), encoding="utf-8")

        self._prior_spine_config = os.environ.get("SPINE_CONFIG")
        os.environ["SPINE_CONFIG"] = str(self._config_path)
        os.environ.pop("SPINE_MATCHER_CHECKPOINT", None)
        os.environ.pop("SPINE_MATCHER_W_APP", None)

        from fastapi.testclient import TestClient
        from spine_annotator_backend.app import app

        self.client = TestClient(app)
        self.respan = respan

    def tearDown(self):
        if self._prior_spine_config is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior_spine_config
        shutil.rmtree(self._root, ignore_errors=True)

    def _catalog_rows(self):
        import csv as csvmod

        cat_path = self.respan / "_annotator" / "fov1" / "spine_catalog.csv"
        with open(cat_path, newline="", encoding="utf-8-sig") as f:
            return list(csvmod.DictReader(f))

    def test_ignored_dendrite_excluded_everywhere(self):
        # Ignore dendrite 2 at end-droplet via the Phase 0 linker route.
        r = self.client.post("/mtp/load-from-animal", params={"fov": 1})
        self.assertEqual(r.status_code, 200)

        r = self.client.post(
            "/mtp/ignore-dendrite", json={"timepoint": "end-droplet", "dendrite_id": "2"}
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["ignored_dendrites"].get("end-droplet"), ["2"])

        # Ranking must skip dendrite 2's spines entirely and fall through to
        # dendrite 5's spine (local 20), the only free candidate.
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200)

        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": "S_00001"}
        )
        end_pos = (r.json().get("positions") or {}).get("end-droplet") or {}
        rows = self._catalog_rows()
        suggested_local = next(
            (row["local_spine_id"] for row in rows if row["global_spine_id"] == end_pos.get("spine_id")),
            None,
        )
        self.assertEqual(suggested_local, "20")

        # Manual override of an ignored spine must be hard-blocked, not just
        # skipped by ranking.
        gid_15 = next(
            row["global_spine_id"]
            for row in rows
            if row["timepoint"] == "end-droplet" and row["local_spine_id"] == "15"
        )
        r = self.client.post(
            "/mtp/viewer/set-spine", json={"timepoint": "end-droplet", "spine_id": gid_15}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("ignored", r.text.lower())

        # The merged ignore CSV must list both dendrite-2 spines under source
        # "dendrite".
        from spine_annotator_backend import results_final_store

        ignore_rows = results_final_store._ignored_spine_rows_for_fov(
            self.respan, 1, "TESTANIMAL"
        )
        dendrite_rows = [row for row in ignore_rows if row["source"] == "dendrite"]
        local_ids = sorted(row["local_spine_id"] for row in dendrite_rows)
        self.assertEqual(local_ids, ["15", "16"])

    def test_unignore_dendrite_restores_candidacy(self):
        r = self.client.post("/mtp/load-from-animal", params={"fov": 1})
        self.assertEqual(r.status_code, 200)
        self.client.post(
            "/mtp/ignore-dendrite", json={"timepoint": "end-droplet", "dendrite_id": "2"}
        )
        r = self.client.delete("/mtp/ignore-dendrite/end-droplet/2")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("2", r.json()["ignored_dendrites"].get("end-droplet", []))

        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": "S_00001"}
        )
        end_pos = (r.json().get("positions") or {}).get("end-droplet") or {}
        rows = self._catalog_rows()
        suggested_local = next(
            (row["local_spine_id"] for row in rows if row["global_spine_id"] == end_pos.get("spine_id")),
            None,
        )
        # Back to the naturally closest match (local 15), now that dendrite 2
        # is no longer ignored.
        self.assertEqual(suggested_local, "15")


if __name__ == "__main__":
    unittest.main()
