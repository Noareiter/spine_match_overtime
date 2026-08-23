"""
Crop-provenance parity test.

Guards against `spine_match_overtime`'s crop-extraction pipeline silently
drifting from `DeepNetSpineMatching`'s copy. The appearance-scoring model was
trained on crops produced by DeepNetSpineMatching's `crop_service.py` +
`spine_matcher/cache_crops.py`. If this repo's own crop_service.py (used to
build the *live* crops fed into `score_coords()` at annotation time) ever
diverges from that reference, the appearance score becomes silently
miscalibrated: the encoder would see crops shaped differently than what it
was trained/calibrated on, with no error and no log line.

Two checks:
  1. Functional parity: `centered_crop()` loaded from both repos' copies of
     crop_service.py, run against the same real demo TIFF stack and the same
     coordinates, must produce byte-identical arrays (numpy.testing.assert_array_equal)
     at both the model input shape (41x41x5) and the display/UI shape (96x96x13).
  2. Provenance parity: both repos must resolve `SPINE_UTILS_PATH` (the
     external, shared `load_stack` implementation) to the exact same file on
     disk, via the same env-var-or-default logic in `project_paths.py`.

crop_service.py has no relative imports (pure numpy + typing), so both
copies can be loaded side by side as distinctly-named modules via
importlib, without triggering the package-relative-import errors that
baseline_adapter.py or project_paths.py would raise if loaded the same way.

Run with: python -m unittest tests.test_crop_parity -v
(pytest is not installed in this environment; unittest is stdlib-only.)
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "scripts" / "spine_annotator_backend"

DEEPNET_ROOT = Path(r"D:\DeepNetSpineMatching")
DEEPNET_BACKEND_DIR = DEEPNET_ROOT / "scripts" / "spine_annotator_backend"

DEMO_TIFF = (
    DEEPNET_ROOT
    / "results"
    / "demo_spine_logic_lab"
    / "respan"
    / "pre-droplet"
    / "fov1.tif"
)


def _load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _extract_def_block(source: str, def_name: str) -> str:
    """Return a whitespace-normalized text of a top-level `def def_name(...):` block."""
    lines = source.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {def_name}("):
            start = i
            break
    if start is None:
        raise AssertionError(f"def {def_name} not found in source")
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line and not line[0].isspace():
            break
        block.append(line)
    normalized = "\n".join(l.rstrip() for l in block).strip()
    return normalized


@unittest.skipUnless(
    DEEPNET_ROOT.is_dir(),
    f"reference repo not found at {DEEPNET_ROOT} on this machine",
)
class TestCropServiceParity(unittest.TestCase):
    """Functional parity of centered_crop() between the two repos' crop_service.py."""

    @classmethod
    def setUpClass(cls):
        this_repo_crop_service = BACKEND_DIR / "crop_service.py"
        deepnet_crop_service = DEEPNET_BACKEND_DIR / "crop_service.py"

        if not this_repo_crop_service.is_file():
            raise unittest.SkipTest(f"missing {this_repo_crop_service}")
        if not deepnet_crop_service.is_file():
            raise unittest.SkipTest(f"missing {deepnet_crop_service}")
        if not DEMO_TIFF.is_file():
            raise unittest.SkipTest(f"missing demo TIFF {DEMO_TIFF}")

        cls.mod_here = _load_module_from_path(
            "crop_service_this_repo", this_repo_crop_service
        )
        cls.mod_deepnet = _load_module_from_path(
            "crop_service_deepnet_repo", deepnet_crop_service
        )
        cls.stack = tifffile.imread(str(DEMO_TIFF))

    def test_centered_crop_model_input_shape_matches(self):
        # 41x41x5 is the appearance model's trained crop shape.
        x, y, z = 60.0, 45.0, 6.0
        crop_here, meta_here = self.mod_here.centered_crop(
            self.stack, x, y, z, width=41, height=41, depth=5
        )
        crop_deepnet, meta_deepnet = self.mod_deepnet.centered_crop(
            self.stack, x, y, z, width=41, height=41, depth=5
        )
        np.testing.assert_array_equal(crop_here, crop_deepnet)
        self.assertEqual(meta_here, meta_deepnet)

    def test_centered_crop_display_shape_matches(self):
        # 96x96x13 is the UI's display/preview crop shape.
        x, y, z = 60.0, 45.0, 6.0
        crop_here, meta_here = self.mod_here.centered_crop(
            self.stack, x, y, z, width=96, height=96, depth=13
        )
        crop_deepnet, meta_deepnet = self.mod_deepnet.centered_crop(
            self.stack, x, y, z, width=96, height=96, depth=13
        )
        np.testing.assert_array_equal(crop_here, crop_deepnet)
        self.assertEqual(meta_here, meta_deepnet)

    def test_centered_crop_edge_clamping_matches(self):
        # Coordinates near the stack boundary exercise the clamping branch
        # in _axis_bounds - the most likely place for a silent divergence.
        z_max, y_max, x_max = self.stack.shape
        x, y, z = float(x_max - 1), float(y_max - 1), float(z_max - 1)
        crop_here, meta_here = self.mod_here.centered_crop(
            self.stack, x, y, z, width=41, height=41, depth=5
        )
        crop_deepnet, meta_deepnet = self.mod_deepnet.centered_crop(
            self.stack, x, y, z, width=41, height=41, depth=5
        )
        np.testing.assert_array_equal(crop_here, crop_deepnet)
        self.assertEqual(meta_here, meta_deepnet)
        self.assertTrue(meta_here["clamped"]["x"] or meta_here["clamped"]["y"])


@unittest.skipUnless(
    DEEPNET_ROOT.is_dir(),
    f"reference repo not found at {DEEPNET_ROOT} on this machine",
)
class TestLoadStackProvenanceParity(unittest.TestCase):
    """
    baseline_adapter.py can't be loaded standalone via importlib (it does
    package-relative imports like `from .project_paths import ...`), so
    parity here is checked at the source level: both repos' `load_stack`
    must delegate to the exact same external utils.py, resolved through
    the same env-var-or-default logic.
    """

    def test_load_stack_definitions_are_textually_identical(self):
        here_src = _read_source(BACKEND_DIR / "baseline_adapter.py")
        deepnet_src = _read_source(DEEPNET_BACKEND_DIR / "baseline_adapter.py")
        here_block = _extract_def_block(here_src, "load_stack")
        deepnet_block = _extract_def_block(deepnet_src, "load_stack")
        self.assertEqual(here_block, deepnet_block)

    def test_tracking_scripts_root_resolves_identically(self):
        here_src = _read_source(BACKEND_DIR / "project_paths.py")
        deepnet_src = _read_source(DEEPNET_BACKEND_DIR / "project_paths.py")
        here_block = _extract_def_block(here_src, "tracking_scripts_root")
        deepnet_block = _extract_def_block(deepnet_src, "tracking_scripts_root")
        self.assertEqual(here_block, deepnet_block)

        # Both repos must default to (or env-override to) the same
        # external utils.py - that's the actual load_stack implementation.
        self.assertIn(r"D:\learning_project_spines\scripts", here_src)
        self.assertIn(r"D:\learning_project_spines\scripts", deepnet_src)

    def test_spine_utils_path_suffix_is_identical(self):
        here_src = _read_source(BACKEND_DIR / "baseline_adapter.py")
        deepnet_src = _read_source(DEEPNET_BACKEND_DIR / "baseline_adapter.py")
        needle = 'step2-spine tracking" / "spine_matching_tool" / "utils.py"'
        self.assertIn(needle, here_src)
        self.assertIn(needle, deepnet_src)


if __name__ == "__main__":
    unittest.main()
