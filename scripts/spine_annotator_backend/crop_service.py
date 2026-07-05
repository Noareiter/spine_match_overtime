from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _axis_bounds(center: int, size: int, axis_len: int) -> Tuple[int, int, bool]:
    half = size // 2
    start = center - half
    end = start + size
    clamped = False
    if start < 0:
        start = 0
        end = min(size, axis_len)
        clamped = True
    if end > axis_len:
        end = axis_len
        start = max(0, end - size)
        clamped = True
    return start, end, clamped


def centered_crop(
    stack: np.ndarray,
    x: float,
    y: float,
    z: float,
    width: int = 96,
    height: int = 96,
    depth: int = 13,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int | bool]]]:
    zc = int(round(z))
    yc = int(round(y))
    xc = int(round(x))

    z0, z1, z_clamped = _axis_bounds(zc, depth, stack.shape[0])
    y0, y1, y_clamped = _axis_bounds(yc, height, stack.shape[1])
    x0, x1, x_clamped = _axis_bounds(xc, width, stack.shape[2])

    crop = stack[z0:z1, y0:y1, x0:x1]

    meta = {
        "source_bounds": {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "clamped": {"z": z_clamped, "y": y_clamped, "x": x_clamped},
        "center_index_source": {"z": zc, "y": yc, "x": xc},
        "center_index_local": {"z": zc - z0, "y": yc - y0, "x": xc - x0},
    }
    return crop, meta


def xy_plane_at_stack_z(
    stack: np.ndarray,
    *,
    z_plane: int,
    x: float,
    y: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int | bool]]]:
    """Single Z plane from the full stack with XY centered on (x, y)."""
    raw_z = int(z_plane)
    zp = int(np.clip(raw_z, 0, max(int(stack.shape[0]) - 1, 0)))
    z_clamped = raw_z != zp
    yc = int(round(float(y)))
    xc = int(round(float(x)))
    y0, y1, y_clamped = _axis_bounds(yc, height, stack.shape[1])
    x0, x1, x_clamped = _axis_bounds(xc, width, stack.shape[2])
    plane = np.asarray(stack[zp, y0:y1, x0:x1], dtype=float)
    meta = {
        "source_bounds": {"z0": zp, "z1": zp + 1, "y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "clamped": {"z": z_clamped, "y": y_clamped, "x": x_clamped},
        "center_index_source": {"z": zp, "y": yc, "x": xc},
        "center_index_local": {"z": 0, "y": yc - y0, "x": xc - x0},
    }
    return plane, meta

