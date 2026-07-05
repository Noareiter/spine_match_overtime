"""Run Tk file/folder dialogs on this process's main thread (subprocess of the server).

The FastAPI server must not call tkinter directly: worker threads + Tcl cause
"Tcl_AsyncDelete: async handler deleted by the wrong thread" and flaky 400s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: _tk_dialog_worker.py <payload.json> <out.txt>", file=sys.stderr)
        sys.exit(2)
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mode = str(payload.get("mode", "open"))
    title = str(payload.get("title", ""))
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = ""
    try:
        if mode == "open":
            fts = [tuple(x) for x in payload.get("filetypes", [])]
            path = filedialog.askopenfilename(title=title, filetypes=fts) or ""
        elif mode == "dir":
            path = filedialog.askdirectory(title=title, mustexist=bool(payload.get("mustexist", True))) or ""
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        root.destroy()
    Path(sys.argv[2]).write_text(path, encoding="utf-8")


if __name__ == "__main__":
    main()
