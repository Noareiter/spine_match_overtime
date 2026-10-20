"""Shared helpers for tests that build a throwaway respan/ workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def add_tiffs(respan: Path, fov: int = 1) -> None:
    """Write a small blank fov<N>.tif into every timepoint folder that has Tables/.

    The server refuses to load an animal unless every selected timepoint has
    both its detection CSV and its TIFF.
    """
    for tp_dir in respan.iterdir():
        if (tp_dir / "Tables").is_dir():
            tifffile.imwrite(tp_dir / f"fov{fov}.tif", np.zeros((5, 260, 260), dtype=np.uint16))
