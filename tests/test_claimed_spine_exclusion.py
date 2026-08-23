"""
Regression test: a spine already claimed by one confirmed lineage must never
be suggested as the candidate match for a different, still-unmatched spine
at the same timepoint.

Found via a user-supplied reproduction: mid-droplet spine 1 was matched to
end-droplet spine 15 and confirmed. Selecting a *different* mid-droplet spine
(spine 2) still suggested end-droplet spine 15 as its candidate match, even
though it was already claimed - the ranking in mtp_spine_matching.py had no
concept of "already claimed", only the separate orphan-phase queue did.
Confirming that bad suggestion didn't corrupt data (the app's duplicate-
conflict detector caught it), but it created unnecessary conflict-resolution
work and offered no warning, unlike a manual claim via /set-spine.

Fix: ranking/suggestion (build_lineage_positions, build_lineage_positions_
from_anchor, build_pre_mid_queues) now excludes already-claimed candidates
entirely, falling through to the next-best free candidate or a coordinate
fallback - mirroring what the orphan-phase queue already did correctly.

Uses a small synthetic 2-timepoint fixture (no external repo/checkpoint
dependency) so this can run anywhere.

Run with: python -m unittest tests.test_claimed_spine_exclusion -v
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
2,150.0,150.0,5.0,1
"""

END_CSV = """spine_id,x,y,z,dendrite_id
15,101.0,101.0,5.0,1
20,200.0,200.0,5.0,1
"""


class TestClaimedSpineExcludedFromSuggestion(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="claimed_spine_excl_"))
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

    def tearDown(self):
        if self._prior_spine_config is None:
            os.environ.pop("SPINE_CONFIG", None)
        else:
            os.environ["SPINE_CONFIG"] = self._prior_spine_config
        shutil.rmtree(self._root, ignore_errors=True)

    def test_second_lineage_does_not_suggest_already_claimed_spine(self):
        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200)

        # Spine 1 (mid-droplet) naturally matches spine 15 (end-droplet, closest).
        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": "S_00001"}
        )
        positions = r.json().get("positions") or {}
        self.assertEqual(positions.get("end-droplet", {}).get("spine_id"), "S_00003")

        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("saved"))

        # Spine 2 (mid-droplet, unrelated) must NOT be offered the now-claimed
        # spine 15 (S_00003) as its candidate - it should fall through to the
        # only remaining free candidate, spine 20 (S_00004).
        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": "S_00002"}
        )
        positions2 = r.json().get("positions") or {}
        end_sid = positions2.get("end-droplet", {}).get("spine_id")
        self.assertNotEqual(
            end_sid,
            "S_00003",
            "already-claimed spine was suggested again as a fresh candidate",
        )
        self.assertEqual(end_sid, "S_00004")

        # Confirming it must not create a duplicate-conflict record - it's a
        # genuinely free match now, not a contested one.
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200)
        coverage = r.json().get("coverage") or {}
        self.assertEqual(coverage.get("duplicate_unresolved", 0), 0)


if __name__ == "__main__":
    unittest.main()
