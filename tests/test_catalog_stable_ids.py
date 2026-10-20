"""
Catalog rebuild keeps S_* ids (rules review bug 4).

A CSV change used to renumber every S_* id, so saved lineages pointed at the
wrong spines. Now a detection keeps its id while (timepoint, local id, position)
is unchanged; new or moved detections get fresh ids above every used one.

Run with: python -m unittest tests.test_catalog_stable_ids -v
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

from spine_annotator_backend import spine_catalog_store as cat  # noqa: E402

HEAD = "spine_id,x,y,z,dendrite_id\n"


class TestCatalogStableIds(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="catalog_ids_"))
        self.pre = self._root / "pre.csv"
        self.mid = self._root / "mid.csv"
        self.pre.write_text(HEAD + "1,10,10,1,1\n2,20,20,1,1\n")
        self.mid.write_text(HEAD + "1,11,10,1,1\n")

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def _ensure(self, rebuild=False):
        c, _, _ = cat.ensure_catalog(
            self._root, 1, animal_id="T", timepoint_names=["pre", "mid"],
            csv_by_tp={"pre": self.pre, "mid": self.mid}, rebuild=rebuild,
        )
        return c

    def test_added_row_does_not_renumber(self):
        first = self._ensure()
        self.assertEqual(first.global_id("mid", "1"), "S_00003")
        self.pre.write_text(HEAD + "0,5,5,1,1\n1,10,10,1,1\n2,20,20,1,1\n")
        second = self._ensure(rebuild=True)
        self.assertEqual(second.global_id("pre", "1"), "S_00001")
        self.assertEqual(second.global_id("pre", "2"), "S_00002")
        self.assertEqual(second.global_id("mid", "1"), "S_00003")
        self.assertEqual(second.global_id("pre", "0"), "S_00004")

    def test_moved_detection_gets_new_id(self):
        self._ensure()
        self.pre.write_text(HEAD + "1,10,10,1,1\n2,80,80,1,1\n")
        second = self._ensure(rebuild=True)
        self.assertEqual(second.global_id("pre", "1"), "S_00001")
        self.assertEqual(second.global_id("pre", "2"), "S_00004")


if __name__ == "__main__":
    unittest.main()
