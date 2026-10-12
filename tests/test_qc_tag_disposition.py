"""
Regression test: ignoring a spine via Timepoint mode (press S, then I) must
permanently remove it from the "unreviewed" sweep - not just from ranking.

Found via user reproduction: reached spine 6 (end-lever, otherwise unclaimed),
pressed S to switch to Timepoint mode, then set fate=ignore. The QC tag was
saved correctly to spine_qc_tags.json, but the spine still reappeared as if
unreviewed.

Root cause was two-fold:
  1. spine_qc_store.build_disposition_rows() never looked at spine_qc_tags.json
     at all (or ignored_dendrites.json, same gap) - only OOF pool-ignore and
     lineage assignments could ever mark a spine as not-unreviewed.
  2. The disposition cache's staleness check only compared against
     lineage_decisions.json's mtime, so even fixing (1) wouldn't help until
     some unrelated lineage save happened to invalidate the cache.
  3. set_fate()/set_fate_all_tps() never refreshed the in-memory phase
     queues after saving a QC tag, so the current session kept offering it
     even before considering the cache.

All three are fixed: build_disposition_rows now checks QC tags and ignored
dendrites (new DISPOSITION_QC_IGNORE/QC_ARTIFACT/DENDRITE_IGNORE values),
_disposition_cache_valid checks all three ignore-source file mtimes, and
set_fate()/set_fate_all_tps() rebuild the phase queues immediately after
tagging.

Run with: python -m unittest tests.test_qc_tag_disposition -v
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

# spine 5 is the natural match for mid-droplet spine 1; spine 6 is isolated
# (different dendrite, far away) and stays genuinely unclaimed.
END_LEVER_CSV = """spine_id,x,y,z,dendrite_id
5,101.0,101.0,5.0,1
6,900.0,900.0,5.0,9
"""


class TestQcTagDisposition(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="qc_tag_disposition_"))
        respan = self._root / "respan"
        (respan / "mid-droplet" / "Tables").mkdir(parents=True)
        (respan / "end-lever" / "Tables").mkdir(parents=True)
        (respan / "mid-droplet" / "Tables" / "fov1_detected_spines.csv").write_text(
            MID_CSV, encoding="utf-8"
        )
        (respan / "end-lever" / "Tables" / "fov1_detected_spines.csv").write_text(
            END_LEVER_CSV, encoding="utf-8"
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

    def test_timepoint_mode_ignore_removes_spine_from_unreviewed(self):
        from spine_annotator_backend import spine_qc_store

        r = self.client.post("/mtp/viewer/load", params={"fov": 1})
        self.assertEqual(r.status_code, 200)

        # Confirm mid-droplet spine 1 -> end-lever spine 5, leaving spine 6
        # genuinely unclaimed.
        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": "S_00001"}
        )
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/mtp/viewer/confirm-lineage", json={})
        self.assertEqual(r.status_code, 200)

        # Resolve spine 6's global id.
        import csv as csvmod

        cat_path = self.respan / "_annotator" / "fov1" / "spine_catalog.csv"
        with open(cat_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csvmod.DictReader(f))
        gid_6 = next(
            row["global_spine_id"]
            for row in rows
            if row["timepoint"] == "end-lever" and row["local_spine_id"] == "6"
        )

        # Reproduce exactly: select spine 6, press S (timepoint mode), press I (ignore).
        r = self.client.post(
            "/mtp/viewer/select-spine", json={"t1_spine_id": gid_6}
        )
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/mtp/viewer/review-mode", json={"mode": "timepoint"})
        self.assertEqual(r.status_code, 200)
        r = self.client.post(
            "/mtp/viewer/set-fate",
            json={"timepoint": "end-lever", "fate": "ignore", "review_mode": "timepoint"},
        )
        self.assertEqual(r.status_code, 200)

        # Disposition must reflect the QC tag, not "unreviewed".
        disp_rows, _ = spine_qc_store.build_disposition_rows(
            self.respan, 1, animal_id="TESTANIMAL",
            timepoint_names=["mid-droplet", "end-lever"],
        )
        row6 = next(row for row in disp_rows if row["global_spine_id"] == gid_6)
        self.assertEqual(row6["disposition"], spine_qc_store.DISPOSITION_QC_IGNORE)

        # It must not reappear in the unreviewed sweep.
        unrev = spine_qc_store.unreviewed_queue(
            self.respan, 1, ["mid-droplet", "end-lever"]
        )
        self.assertNotIn(gid_6, [u["global_spine_id"] for u in unrev])


if __name__ == "__main__":
    unittest.main()
