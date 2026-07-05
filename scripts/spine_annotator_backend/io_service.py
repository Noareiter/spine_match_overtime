from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_WORKER = Path(__file__).resolve().parent / "_tk_dialog_worker.py"


def _run_tk_dialog(payload: Dict[str, Any]) -> str:
    """Run Tk in a fresh subprocess so dialogs are not on the Uvicorn worker thread."""
    if not _WORKER.is_file():
        raise FileNotFoundError(f"Missing Tk dialog helper: {_WORKER}")

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as jf:
        json.dump(payload, jf)
        json_path = jf.name
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as out_f:
        out_path = out_f.name

    try:
        proc = subprocess.run(
            [sys.executable, str(_WORKER), json_path, out_path],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"File dialog helper exited with code {proc.returncode}")
        return Path(out_path).read_text(encoding="utf-8").strip()
    finally:
        try:
            Path(json_path).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(out_path).unlink(missing_ok=True)
        except OSError:
            pass


def _pick_file(title: str, filetypes: Tuple[Tuple[str, str], ...]) -> str:
    fts: List[List[str]] = [list(x) for x in filetypes]
    file_path = _run_tk_dialog({"mode": "open", "title": title, "filetypes": fts})
    if not file_path:
        raise ValueError(f"Selection cancelled for: {title}")
    return str(Path(file_path).resolve())


def pick_directory_via_dialog(title: str = "Select export destination folder") -> str:
    folder = _run_tk_dialog({"mode": "dir", "title": title, "mustexist": True})
    if not folder:
        raise ValueError("Selection cancelled for export destination folder.")
    return str(Path(folder).resolve())


def select_files_via_dialog(
    t1_tiff_path: Optional[str],
    t2_tiff_path: Optional[str],
    t1_csv_path: Optional[str],
    t2_csv_path: Optional[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}

    resolved["t1_tiff_path"] = t1_tiff_path or _pick_file(
        "Select T1 TIFF stack",
        (("TIFF files", "*.tif *.tiff"), ("All files", "*.*")),
    )
    resolved["t1_csv_path"] = t1_csv_path or _pick_file(
        "Select T1 RESPAN CSV",
        (("CSV files", "*.csv"), ("All files", "*.*")),
    )
    resolved["t2_tiff_path"] = t2_tiff_path or _pick_file(
        "Select T2 TIFF stack",
        (("TIFF files", "*.tif *.tiff"), ("All files", "*.*")),
    )
    resolved["t2_csv_path"] = t2_csv_path or _pick_file(
        "Select T2 RESPAN CSV",
        (("CSV files", "*.csv"), ("All files", "*.*")),
    )
    return resolved


def validate_existing_path(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path
