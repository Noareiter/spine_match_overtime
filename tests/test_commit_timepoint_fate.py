"""
Timepoint mode: S + A on one spine must affect ONLY that spine at that
timepoint, and must be written (and exported) when saved.

User report: marking spine 24 pre-droplet as artifact (S then A) still dragged
the rest of its lineage (41 mid, 21 end-droplet, ...) along, and the artifact
never reached the final export.

Now: A only stages; /commit-timepoint-fate (Shift+Enter) writes the QC tag,
deletes any saved lineage claiming that spine at that timepoint (other
timepoints return to the pool), never saves the open panel as a lineage, and is
undone by /undo-last-decision. Artifact tags are exported to artifact_spines.csv.

Run with: python -m unittest tests.test_commit_timepoint_fate -v
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

# A second, far-away mid-droplet spine keeps the main phase from finishing when
# spine 1's lineage is saved, so spine 1 can be re-opened afterwards.
MID_CSV = """spine_id,x,y,z,dendrite_id
1,100.0,100.0,5.0,1
2,500.0,500.0,5.0,3
"""
END_LEVER_CSV = """spine_id,x,y,z,dendrite_id
5,101.0,101.0,5.0,1
6,900.0,900.0,5.0,9
"""


class TestCommitTimepointFate(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="commit_tp_fate_"))
        respan = self._root / "respan"
        for tp, body in (("mid-droplet", MID_CSV), ("end-lever", END_LEVER_CSV)):
            (respan / tp / "Tables").mkdir(parents=True)
            (respan / tp / "Tables" / "fov1_detected_spines.csv").write_text(
                body, encoding="utf-8"
            )
        config = {
            "animal_id": "TESTANIMAL",
            "workspace": str(self._root),
            "default_fov": 1,
            "fovs": [1],
            "timepoint_order": ["mid-droplet", "end-lever"],
            "active_timepoints": [],
        }
        self._config_path = self._root / "config.json"
        self._config_path.write_text(json.dumps(config), encoding="utf-8")
        self._prior = os.environ.get("SPINE_CONFIG")
        os.environ["SPINE_CONFIG"] = str(self._config_path)
        os.environ.pop("SPINE_MATCHER_CHECKPOINT", None)
        os.environ.pop("SPINE_MATCHER_W_APP", None)

        from fastapi.testclient import TestClient
        from spine_annotator_backend.app import app

        self.client = TestClient(app)
        self.respan = respan

    def tearDown(self):
        if self._prior is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior
        shutil.rmtree(self._root, ignore_errors=True)

    def _gid(self, tp: str, local: str) -> str:
        cat = self.respan / "_annotator" / "fov1" / "spine_catalog.csv"
        with open(cat, newline="", encoding="utf-8-sig") as f:
            return next(
                r["global_spine_id"]
                for r in csv.DictReader(f)
                if r["timepoint"] == tp and r["local_spine_id"] == local
            )

    def _lineages(self):
        from spine_annotator_backend import spine_lineage_store

        return spine_lineage_store.load_decisions(self.respan, 1).get("lineages") or []

    def _save_lineage_1_to_5(self):
        self.assertEqual(self.client.post("/mtp/viewer/load", params={"fov": 1}).status_code, 200)
        r = self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": self._gid("mid-droplet", "1")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.post("/mtp/viewer/confirm-lineage", json={}).status_code, 200)
        self.assertEqual(len(self._lineages()), 1)

    def test_commit_artifact_deletes_lineage_and_tags_only_that_spine(self):
        from spine_annotator_backend import spine_qc_tag_store

        self._save_lineage_1_to_5()
        gid1 = self._gid("mid-droplet", "1")
        gid5 = self._gid("end-lever", "5")

        # Re-open the saved lineage, switch to timepoint mode, mark artifact.
        r = self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": gid1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            self.client.post("/mtp/viewer/review-mode", json={"mode": "timepoint"}).status_code, 200
        )
        r = self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "mid-droplet", "fate": "artifact"},
        )
        self.assertEqual(r.status_code, 200, r.text)

        tags = spine_qc_tag_store.load_tags(self.respan, 1)
        self.assertEqual(tags.get("mid-droplet", {}).get(gid1), "artifact")
        # The lineage's other timepoint is NOT tagged and NOT carried along.
        self.assertNotIn(gid5, tags.get("end-lever", {}))
        self.assertEqual(self._lineages(), [])

    def test_unsaved_panel_is_not_saved_as_lineage(self):
        from spine_annotator_backend import spine_qc_tag_store

        self.client.post("/mtp/viewer/load", params={"fov": 1})
        gid1 = self._gid("mid-droplet", "1")
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": gid1})
        self.client.post("/mtp/viewer/review-mode", json={"mode": "timepoint"})
        r = self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "mid-droplet", "fate": "artifact"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._lineages(), [])
        self.assertEqual(
            spine_qc_tag_store.load_tags(self.respan, 1)["mid-droplet"][gid1], "artifact"
        )
        # Its would-be partner is untouched and still free.
        self.assertNotIn(self._gid("end-lever", "5"), spine_qc_tag_store.load_tags(self.respan, 1).get("end-lever", {}))

    def test_undo_restores_lineage_and_clears_tag(self):
        from spine_annotator_backend import spine_qc_tag_store

        self._save_lineage_1_to_5()
        gid1 = self._gid("mid-droplet", "1")
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": gid1})
        self.client.post("/mtp/viewer/review-mode", json={"mode": "timepoint"})
        self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "mid-droplet", "fate": "artifact"},
        )
        r = self.client.post("/mtp/viewer/undo-last-decision")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(len(self._lineages()), 1)
        self.assertNotIn(gid1, spine_qc_tag_store.load_tags(self.respan, 1).get("mid-droplet", {}))

    def test_artifact_rows_exported(self):
        from spine_annotator_backend import results_final_store

        self._save_lineage_1_to_5()
        gid1 = self._gid("mid-droplet", "1")
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": gid1})
        self.client.post("/mtp/viewer/review-mode", json={"mode": "timepoint"})
        self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "mid-droplet", "fate": "artifact"},
        )
        rows = results_final_store._artifact_spine_rows_for_fov(self.respan, 1, "TESTANIMAL")
        self.assertEqual([(r["timepoint"], r["global_spine_id"]) for r in rows], [("mid-droplet", gid1)])

    def test_commit_rejects_wrong_timepoint_and_missing_spine(self):
        self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.client.post("/mtp/viewer/select-spine", json={"t1_spine_id": self._gid("mid-droplet", "1")})
        r = self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "end-lever", "fate": "artifact"},
        )
        self.assertEqual(r.status_code, 400)
        r = self.client.post(
            "/mtp/viewer/commit-timepoint-fate",
            json={"timepoint": "mid-droplet", "fate": "lost"},
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
