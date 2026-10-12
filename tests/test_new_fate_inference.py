"""
Regression test: "new" (formation) fate inference must be exempted only at
the session's TRUE first timepoint - never at whichever timepoint a specific
lineage's review happened to start from.

Found via real GP04 production data: a lineage anchored at mid-droplet (via
that timepoint's own orphan-review phase, because the spine wasn't picked up
naturally from pre-droplet) never got "new" at mid-droplet, even though
pre-droplet showed a genuine "no match, no fate" observation for it - i.e. a
confirmed absence, not an unknown/unreviewed gap. The domain owner confirmed:
if a spine wasn't found at an earlier real timepoint, that IS grounds for
inferring "new" at the timepoint it does appear - the exemption should only
protect the session's actual first timepoint, which has no earlier data to
compare against by definition.

Fix: _finalize_lineage_per_tp's NEW-inference call now always exempts
timepoint_names[0] (the session's true first timepoint), regardless of which
timepoint a given lineage's own pre_timepoint/anchor happens to be. The
lost-inference call (a separate concern) is untouched and still uses the
lineage's own anchor.

Run with: python -m unittest tests.test_new_fate_inference -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spine_annotator_backend import spine_lineage_store  # noqa: E402


class TestNewFateInference(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="new_fate_infer_"))

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def test_new_inferred_when_lineage_anchored_later_than_session_start(self):
        """A lineage anchored at mid-droplet, with a confirmed 'not there' at
        pre-droplet (session's true first TP), must get 'new' at mid-droplet."""
        per_tp = {
            "pre-droplet": {
                "spine_id": None,
                "local_spine_id": None,
                "x": 982.0, "y": 210.0, "z": 32.0,
                "dendrite_id": "", "source": "coord_region", "fate": None,
            },
            "mid-droplet": {
                "spine_id": "S_00004",
                "local_spine_id": "100",
                "x": 982.0, "y": 210.0, "z": 32.0,
                "dendrite_id": "10", "source": "anchor_base", "fate": None,
            },
        }
        spine_lineage_store.save_decision(
            self._root, 1,
            animal_id="GP04", pre_spine_id="100", pre_timepoint="mid-droplet",
            timepoint_names=["pre-droplet", "mid-droplet"],
            per_tp=per_tp, lineage_key="S_00004",
        )
        data = spine_lineage_store.load_decisions(self._root, 1)
        lin = data["lineages"][0]
        self.assertEqual(lin["per_tp"]["mid-droplet"].get("fate"), "new")

    def test_session_first_timepoint_never_gets_new(self):
        """A lineage genuinely anchored at the session's true first timepoint
        must never get 'new' there, regardless of what precedes it (nothing
        does - there is no earlier data by definition)."""
        per_tp = {
            "pre-droplet": {
                "spine_id": "S_00001",
                "local_spine_id": "1",
                "x": 100.0, "y": 100.0, "z": 5.0,
                "dendrite_id": "1", "source": "pre_base", "fate": None,
            },
            "mid-droplet": {
                "spine_id": "S_00003",
                "local_spine_id": "10",
                "x": 105.0, "y": 105.0, "z": 5.0,
                "dendrite_id": "1", "source": "pre_mid_match", "fate": None,
            },
        }
        spine_lineage_store.save_decision(
            self._root, 1,
            animal_id="GP04", pre_spine_id="1", pre_timepoint="pre-droplet",
            timepoint_names=["pre-droplet", "mid-droplet"],
            per_tp=per_tp, lineage_key="S_00001",
        )
        data = spine_lineage_store.load_decisions(self._root, 1)
        lin = data["lineages"][0]
        self.assertIsNone(lin["per_tp"]["pre-droplet"].get("fate"))


if __name__ == "__main__":
    unittest.main()
