"""
Regression test: a spine that is "new" at one timepoint and genuinely absent
at the next must be inferred as "lost" there - not silently wiped back to an
unassigned/no-fate state.

Found via real GP04 production data: spines new at end-droplet, with a true
gap (no detection at all) at end-lever, ended up with fate=None at end-lever
instead of fate="lost", even though save_decision()'s own return value
(lost_inferred=True) said a loss WAS inferred.

Root cause: _finalize_lineage_per_tp calls release_unbound_matches() twice -
once before infer_lost_after_last_match(), once after. The second call's
"NEW at TPk" branch scans forward from the NEW timepoint to find where the
tracked span ends, and stops when it hits an already-terminal fate (lost/
ignore/artifact) - correctly recognizing that as the end of the span - but
left `end_i` pointing at the last *present* index instead of the terminal
fate's own index. The very "lost" entry infer_lost_after_last_match had just
written one step earlier in the same call was therefore inside the release
range and got wiped back to a blank "returned_to_pool" placeholder.

Run with: python -m unittest tests.test_lost_after_new -v
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


class TestLostAfterNew(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="lost_after_new_"))

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def test_genuine_gap_after_new_becomes_lost_not_blank(self):
        per_tp = {
            "pre-droplet": {
                "spine_id": None, "local_spine_id": None,
                "x": 1.0, "y": 1.0, "z": 1.0,
                "dendrite_id": "", "source": "coord_region", "fate": None,
            },
            "mid-droplet": {
                "spine_id": None, "local_spine_id": None,
                "x": 1.0, "y": 1.0, "z": 1.0,
                "dendrite_id": "", "source": "coord_region", "fate": None,
            },
            "end-droplet": {
                "spine_id": "S_0020", "local_spine_id": "20",
                "x": 1.0, "y": 1.0, "z": 1.0,
                "dendrite_id": "2", "source": "anchor_base", "fate": None,
            },
            "end-lever": {
                "spine_id": None, "local_spine_id": None,
                "x": 1.0, "y": 1.0, "z": 1.0,
                "dendrite_id": "", "source": "coord_region", "fate": None,
            },
        }
        spine_lineage_store.save_decision(
            self._root, 1,
            animal_id="GP04", pre_spine_id="20", pre_timepoint="end-droplet",
            timepoint_names=["pre-droplet", "mid-droplet", "end-droplet", "end-lever"],
            per_tp=per_tp, lineage_key="S_0020",
        )
        data = spine_lineage_store.load_decisions(self._root, 1)
        lin = data["lineages"][0]
        self.assertEqual(lin["per_tp"]["end-droplet"].get("fate"), "new")
        self.assertEqual(lin["per_tp"]["end-lever"].get("fate"), "lost")


if __name__ == "__main__":
    unittest.main()
