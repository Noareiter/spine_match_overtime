"""
Reproducibility (persistence) test for the appearance score.

Verifies that when a lineage decision is confirmed while an appearance-model
checkpoint is configured, `score_app` (the calibrated appearance score for
that confirmed pair) and `score_app_checkpoint` (which model produced it)
are actually written into lineage_decisions.json on disk -- not just held
transiently in a DataFrame during ranking.

This exercises the real production call path:
    mtp_spine_viewer._annotate_score_app()  (mutates per_tp in place)
        -> baseline_adapter.score_appearance_pair()  (real model inference)
    spine_lineage_store.save_decision()  (persists to lineage_decisions.json)

against a scratch copy of the demo workspace and the DEMO10 checkpoint
(trained/held-out on demo FOV1), so this is a real end-to-end check, not a
fabricated sample file.

Run with: python -m unittest tests.test_score_app_persistence -v
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

DEEPNET_ROOT = Path(r"D:\DeepNetSpineMatching")
DEMO_RESPAN = DEEPNET_ROOT / "results" / "demo_spine_logic_lab" / "respan"
CHECKPOINT = DEEPNET_ROOT / "data" / "checkpoints" / "DEMO10" / "fold_fov1_best.pt"

_ENV_CHECKPOINT = "SPINE_MATCHER_CHECKPOINT"


@unittest.skipUnless(DEEPNET_ROOT.is_dir(), f"reference repo not found at {DEEPNET_ROOT}")
@unittest.skipUnless(CHECKPOINT.is_file(), f"checkpoint not found at {CHECKPOINT}")
@unittest.skipUnless(DEMO_RESPAN.is_dir(), f"demo workspace not found at {DEMO_RESPAN}")
class TestScoreAppPersistence(unittest.TestCase):
    def setUp(self):
        self._prior_checkpoint_env = os.environ.get(_ENV_CHECKPOINT)
        os.environ[_ENV_CHECKPOINT] = str(CHECKPOINT)
        self._scratch = Path(tempfile.mkdtemp(prefix="score_app_persist_"))

    def tearDown(self):
        if self._prior_checkpoint_env is None:
            os.environ.pop(_ENV_CHECKPOINT, None)
        else:
            os.environ[_ENV_CHECKPOINT] = self._prior_checkpoint_env
        shutil.rmtree(self._scratch, ignore_errors=True)

    def test_score_app_persisted_for_confirmed_pair(self):
        from spine_annotator_backend import mtp_spine_viewer as v
        from spine_annotator_backend import spine_lineage_store

        tiff_pre = DEMO_RESPAN / "pre-droplet" / "fov1.tif"
        tiff_end = DEMO_RESPAN / "end-droplet" / "fov1.tif"
        self.assertTrue(tiff_pre.is_file())
        self.assertTrue(tiff_end.is_file())

        v._STATE.timepoint_names = ["pre-droplet", "end-droplet"]
        v._STATE.files = {
            "pre-droplet": {"tiff": str(tiff_pre)},
            "end-droplet": {"tiff": str(tiff_end)},
        }

        per_tp = {
            "pre-droplet": {"spine_id": "1", "x": 192.0, "y": 275.0, "z": 4.0},
            "end-droplet": {"spine_id": "5", "x": 195.0, "y": 270.0, "z": 5.0},
        }

        # Real production call: mutates per_tp in place with score_app/checkpoint.
        v._annotate_score_app(per_tp)

        self.assertIn("score_app", per_tp["end-droplet"])
        self.assertIsInstance(per_tp["end-droplet"]["score_app"], float)
        self.assertTrue(0.0 <= per_tp["end-droplet"]["score_app"] <= 1.0)
        # Anchor timepoint has nothing preceding it - must never get a score_app.
        self.assertNotIn("score_app", per_tp["pre-droplet"])

        spine_lineage_store.save_decision(
            self._scratch,
            1,
            animal_id="DEMO",
            pre_spine_id="1",
            pre_timepoint="pre-droplet",
            timepoint_names=["pre-droplet", "end-droplet"],
            per_tp=per_tp,
            lineage_key="S_00001",
        )

        decisions_path = self._scratch / "_annotator" / "fov1" / "lineage_decisions.json"
        self.assertTrue(decisions_path.is_file(), "lineage_decisions.json was not written")
        data = json.loads(decisions_path.read_text(encoding="utf-8"))

        lineages = data.get("lineages") or []
        self.assertEqual(len(lineages), 1)
        saved_tp_data = lineages[0]["per_tp"]["end-droplet"]

        self.assertIn("score_app", saved_tp_data)
        score = saved_tp_data["score_app"]
        self.assertIsInstance(score, float)
        self.assertIsNotNone(score)
        self.assertTrue(0.0 <= score <= 1.0)

        self.assertIn("score_app_checkpoint", saved_tp_data)
        self.assertIn("DEMO10", saved_tp_data["score_app_checkpoint"])

    def test_score_app_absent_without_checkpoint_configured(self):
        """No checkpoint -> no fabricated score_app; the field is simply absent."""
        os.environ.pop(_ENV_CHECKPOINT, None)

        from spine_annotator_backend import mtp_spine_viewer as v

        tiff_pre = DEMO_RESPAN / "pre-droplet" / "fov1.tif"
        tiff_end = DEMO_RESPAN / "end-droplet" / "fov1.tif"
        v._STATE.timepoint_names = ["pre-droplet", "end-droplet"]
        v._STATE.files = {
            "pre-droplet": {"tiff": str(tiff_pre)},
            "end-droplet": {"tiff": str(tiff_end)},
        }
        per_tp = {
            "pre-droplet": {"spine_id": "1", "x": 192.0, "y": 275.0, "z": 4.0},
            "end-droplet": {"spine_id": "5", "x": 195.0, "y": 270.0, "z": 5.0},
        }
        v._annotate_score_app(per_tp)
        self.assertNotIn("score_app", per_tp["end-droplet"])
        self.assertNotIn("score_app_checkpoint", per_tp["end-droplet"])


if __name__ == "__main__":
    unittest.main()
