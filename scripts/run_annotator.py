#!/usr/bin/env python3
"""Launch the dendritic spine annotator (FastAPI + browser UI)."""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dendritic spine annotator")
    parser.add_argument(
        "--config",
        default=os.environ.get("SPINE_CONFIG", str(PROJECT_ROOT / "config" / "annotator.json")),
        help="Path to annotator.json (animal_id + workspace)",
    )
    parser.add_argument("--animal-id", default=os.environ.get("SPINE_ANIMAL_ID"))
    parser.add_argument("--imaging-root", default=os.environ.get("SPINE_IMAGING_ROOT"))
    parser.add_argument("--data-root", default=os.environ.get("SPINE_DATA_ROOT"), help="Alias for --workspace")
    parser.add_argument("--workspace", default=os.environ.get("SPINE_WORKSPACE"))
    parser.add_argument("--fov", type=int, default=None, help="Default FOV (overrides config default_fov)")
    parser.add_argument("--open-folders", action="store_true", help="Open Explorer only for folders newly created by bootstrap")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading annotator...", flush=True)

    os.environ["SPINE_CONFIG"] = str(Path(args.config).expanduser().resolve())
    if args.animal_id:
        os.environ["SPINE_ANIMAL_ID"] = args.animal_id
    if args.imaging_root:
        os.environ["SPINE_IMAGING_ROOT"] = args.imaging_root
    if args.data_root:
        os.environ["SPINE_WORKSPACE"] = args.data_root
    if args.workspace:
        os.environ["SPINE_WORKSPACE"] = args.workspace
    if args.fov is not None:
        os.environ["SPINE_DEFAULT_FOV"] = str(args.fov)

    from spine_annotator_backend.animal_config import load_config, resolve_paths

    cfg = load_config()
    ws = Path(cfg.workspace).expanduser() if cfg.workspace else None
    if ws and not ws.is_absolute():
        ws = (PROJECT_ROOT / ws).resolve()
    if ws:
        print(f"Checking workspace: {ws}", flush=True)
        if not ws.exists():
            print(
                f"\nERROR: Workspace folder not found:\n  {ws}\n\n"
                "Connect the drive (e.g. E:) or edit workspace in config/annotator.json.\n",
                flush=True,
            )
            raise SystemExit(1)
    print("Bootstrapping respan layout (may take a moment on network drives)...", flush=True)
    paths = resolve_paths(cfg)
    url = f"http://{args.host}:{args.port}/mtp/"

    print(f"Config: {paths.get('config_path', args.config)}")
    print(f"Animal: {paths.get('animal_id') or '(not set)'}")
    if paths.get("workspace"):
        print(f"Workspace: {paths['workspace']}")
    if paths.get("respan_root"):
        print(f"Respan:  {paths['respan_root']}")
        if paths.get("bootstrap_created"):
            print(f"  (bootstrap created {paths['bootstrap_created']} new folder(s))")
    elif paths.get("error"):
        print(f"Respan:  ERROR — {paths['error']}")
    else:
        print("Respan:  (set workspace in config/annotator.json)")
    print(f"Open {url}  (Phase 1 · Dendrite Links)")
    print(f"Spine tracker: http://{args.host}:{args.port}/mtp/viewer/")
    help_path = (PROJECT_ROOT / "QUICK_REFERENCE.html").resolve()
    if help_path.is_file():
        print(f"Quick reference: {help_path}")

    if args.open_folders:
        try:
            from open_animal_folders import main as open_folders_main
            rc = open_folders_main()
            if rc != 0:
                print("Folder open skipped — create workspace folder or fix config/annotator.json")
        except ModuleNotFoundError:
            print("(Folder open skipped — open_animal_folders module not available)")

    if not args.no_browser:
        help_path = (PROJECT_ROOT / "QUICK_REFERENCE.html").resolve()
        if help_path.is_file():
            webbrowser.open(help_path.as_uri())
        webbrowser.open(url)

    import uvicorn
    from spine_annotator_backend.app import app

    print(f"Starting server on http://{args.host}:{args.port}/mtp/ ...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
