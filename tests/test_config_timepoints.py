"""
Timepoints come only from config, and the whole animal must be complete.

- Subfolders of respan/ that are not in timepoint_order are ignored.
- active_timepoints (if set) is a subset of timepoint_order, kept in its order.
- missing_animal_files lists every missing CSV / TIFF across all FOVs.

Run with: python -m unittest tests.test_config_timepoints -v
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures  # noqa: E402
from spine_annotator_backend import animal_config, animal_layout  # noqa: E402

CSV = "spine_id,x,y,z,dendrite_id\n1,10.0,10.0,2.0,1\n"


class TestConfigTimepoints(unittest.TestCase):
    def setUp(self):
        self._root = Path(tempfile.mkdtemp(prefix="config_tps_"))
        self.respan = self._root / "respan"
        for tp in ("pre-droplet", "mid-droplet", "backup copy"):
            (self.respan / tp / "Tables").mkdir(parents=True)
            for fov in (1, 2):
                (self.respan / tp / "Tables" / f"fov{fov}_detected_spines.csv").write_text(CSV)
        (self.respan / "ROI").mkdir()
        _fixtures.add_tiffs(self.respan, fov=1)
        _fixtures.add_tiffs(self.respan, fov=2)
        self.cfg = animal_config.AnimalConfig(
            animal_id="T", workspace=str(self._root), fovs=[1, 2],
            timepoint_order=["pre-droplet", "mid-droplet"],
        )

    def tearDown(self):
        shutil.rmtree(self._root, ignore_errors=True)

    def test_only_config_folders_are_timepoints(self):
        dirs = animal_layout.discover_timepoint_dirs(
            self.respan, preferred_order=self.cfg.timepoint_order
        )
        self.assertEqual([d.name for d in dirs], ["pre-droplet", "mid-droplet"])

    def test_active_timepoints_subset_in_config_order(self):
        self.cfg.active_timepoints = ["mid-droplet", "pre-droplet"]
        self.assertEqual(animal_config.selected_timepoints(self.cfg), ["pre-droplet", "mid-droplet"])
        self.cfg.active_timepoints = ["end-lever"]
        with self.assertRaises(ValueError):
            animal_config.selected_timepoints(self.cfg)

    def test_complete_animal_has_nothing_missing(self):
        self.assertEqual(animal_config.missing_animal_files(self.cfg, self.respan), [])

    def test_missing_tiff_in_one_fov_is_reported(self):
        (self.respan / "mid-droplet" / "fov2.tif").unlink()
        missing = animal_config.missing_animal_files(self.cfg, self.respan)
        self.assertEqual(len(missing), 1)
        self.assertIn("FOV 2", missing[0])
        self.assertIn("mid-droplet", missing[0])
        self.assertIn("TIFF", missing[0])


if __name__ == "__main__":
    unittest.main()
