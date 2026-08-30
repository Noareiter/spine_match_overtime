"""One-time materializer: ReSpAn_ready per-dendrite output -> this app's respan/ layout.

Source shape (per animal): <source>/Dendrite_NN/Tables/Detected_spines_4D.csv,
one CSV per dendrite covering all timepoints, with a stable cross-timepoint
`spine_id` (ReSpAn's own automatic lineage link) plus a per-dendrite registered
4D image stack at <source>/Dendrite_NN/Registered/Registered_images_4d.tif
(axes TZYX).

This app expects the opposite shape: <workspace>/respan/<timepoint>/Tables/
fov<N>_detected_spines.csv + fov<N>.tif, one file per timepoint. Rather than
touch the app's discovery/session/review code, this script converts the data
once into that shape, mapping Dendrite_NN -> fov N, and pre-seeds two files
the app already knows how to read as an "offer to approve/reject" without
marking anything as decided:

  - spine_catalog.csv: assigns the SAME S_XXXXX global id to every detection
    that shares a ReSpAn spine_id, directly encoding ReSpAn's ground-truth
    link as this app's own global id scheme (spine_catalog_store.ensure_catalog
    loads an existing, non-stale catalog as-is rather than rebuilding it).
  - spine_registry_wide.csv: one row per lineage with the shared id per
    timepoint where detected. mtp_spine_viewer._parse_registry reads this at
    load time and mtp_spine_matching.build_lineage_positions[...] uses it to
    pre-fill the suggested cross-TP position for a spine -- still routed
    through the normal select-spine -> confirm-lineage approve/reject flow,
    since nothing here touches lineage_decisions.json (queue exclusion only
    comes from confirmed decisions).

Also writes dendrite_links.json with a single full-span link (one dendrite
per folder, already linked across all timepoints by ReSpAn's registration),
and a companion respan_ready_gap_flags.csv flagging lineages that are absent
at some timepoint(s) between two present ones, for manual double-checking.

A second, simpler source shape is also handled ("single_tp" mode): if a
dendrite folder has no Detected_spines_4D.csv but does have per-timepoint
<Dendrite_NN>_T##_detected_spines.csv files (ReSpAn run per-timepoint, no
cross-TP tracking, no Registered/ stack), this is just a folder reshape --
each per-TP csv/tif is copied straight into the respan/<TP>/... layout, with
no catalog/registry/dendrite_links pre-seed, since there's no known cross-TP
link to offer. Real per-spine dendrite_id passes through untouched, so Phase
0 dendrite linking is meaningful here, same as a normal fresh dataset.

Usage (manual, explicit paths):
    python -m scripts.spine_annotator_backend.respan_ready_import \
        --source "E:/Noa/Pons - layer 5/Imaging/GP04/ReSpAn_ready" \
        --dest "E:/Noa/Pons - layer 5/Imaging/GP04/respan_ready_bootstrap" \
        --animal-id GP04

Usage (auto, driven by the app's own config -- what start_annotator.bat calls
on every launch): reads `workspace` from the given annotator.json; if it looks
like a pre-matched source (contains Dendrite_NN/Tables/Detected_spines_4D.csv),
materializes it into a sibling `respan_ready_bootstrap` folder next to it and
rewrites the config's workspace/animal_id/fovs/timepoint_order to point at the
result. If `workspace` doesn't look like a pre-matched source, this is a no-op
(covers a normal already-respan-shaped workspace, e.g. a manually built one).
Idempotent: once rewritten, the destination no longer looks like a matched
source, so later launches skip straight through.

    python -m scripts.spine_annotator_backend.respan_ready_import \
        --from-config config/annotator.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tifffile

DENDRITE_ID = "1"
ARTIFACT_CLIP_THRESHOLD = 2000  # see note in materialize_dendrite

CATALOG_COLUMNS = [
    "global_spine_id",
    "animal_id",
    "fov",
    "timepoint",
    "local_spine_id",
    "x",
    "y",
    "z",
    "dendrite_id",
    "source_csv",
]


def _format_global_id(n: int) -> str:
    return f"S_{n:05d}"


def _dendrite_folders(source_root: Path) -> List[Path]:
    return sorted(
        p for p in source_root.iterdir() if p.is_dir() and p.name.lower().startswith("dendrite_")
    )


def _fov_from_dendrite_name(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse dendrite number from folder name: {name}")
    return int(m.group(1))


_SINGLE_TP_CSV_RE = re.compile(r"^(?P<dendrite>.+)_T(?P<tp>\d+)_detected_spines\.csv$", re.IGNORECASE)


def _dendrite_mode(dendrite_dir: Path) -> Optional[str]:
    """"tracked" (Detected_spines_4D.csv, cross-TP spine_id linked, registered
    4D stack), "single_tp" (per-TP detected_spines csv, real per-spine
    dendrite_id, no cross-TP link, raw per-TP tif, no registration), or None."""
    tables = dendrite_dir / "Tables"
    if (tables / "Detected_spines_4D.csv").is_file():
        return "tracked"
    if tables.is_dir():
        for f in tables.iterdir():
            if _SINGLE_TP_CSV_RE.match(f.name):
                return "single_tp"
    return None


def _detect_source_mode(folder: Path) -> Optional[str]:
    """Mode of the first Dendrite_NN child that has one, or None if `folder`
    doesn't directly contain any recognizable dendrite output."""
    if not folder.is_dir():
        return None
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or not child.name.lower().startswith("dendrite_"):
            continue
        mode = _dendrite_mode(child)
        if mode:
            return mode
    return None


def _discover_timepoint_names(dendrite_dirs: List[Path]) -> List[str]:
    """Max `timepoint` value across all dendrites' CSVs -> ["T00", "T01", ...]."""
    max_tp = 0
    for d in dendrite_dirs:
        csv_path = d / "Tables" / "Detected_spines_4D.csv"
        if not csv_path.is_file():
            continue
        tp_col = pd.read_csv(csv_path, usecols=["timepoint"])["timepoint"]
        if len(tp_col):
            max_tp = max(max_tp, int(tp_col.max()))
    if max_tp <= 0:
        raise ValueError("Could not determine timepoint count from any Detected_spines_4D.csv")
    return [f"T{i:02d}" for i in range(max_tp)]


def _single_tp_csv_map(dendrite_dir: Path) -> Dict[str, Path]:
    """tp_name ("T00", ...) -> its detected_spines csv, for single-TP-mode output."""
    tables = dendrite_dir / "Tables"
    out: Dict[str, Path] = {}
    if not tables.is_dir():
        return out
    for f in tables.iterdir():
        m = _SINGLE_TP_CSV_RE.match(f.name)
        if m:
            out[f"T{int(m.group('tp')):02d}"] = f
    return out


def _discover_timepoint_names_single_tp(dendrite_dirs: List[Path]) -> List[str]:
    names: set = set()
    for d in dendrite_dirs:
        names.update(_single_tp_csv_map(d).keys())
    if not names:
        raise ValueError("Could not find any *_T##_detected_spines.csv under the given dendrite folders")
    return sorted(names)


def _write_dendrite_links(annot_dir: Path, animal_id: str, fov: int, timepoint_names: List[str]) -> None:
    payload = {
        "animal_id": animal_id,
        "fov": str(fov),
        "timepoint_names": timepoint_names,
        "files": {},
        "links": [
            {
                "link_id": "link_1",
                "members": {tp: [DENDRITE_ID] for tp in timepoint_names},
                "notes": "ReSpAn: single pre-linked dendrite spanning the whole folder.",
            }
        ],
    }
    (annot_dir / "dendrite_links.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = ["animal_id", "fov", "link_id"] + [f"dendrite_id_{n}" for n in timepoint_names]
    row = [animal_id, str(fov), "link_1"] + [DENDRITE_ID for _ in timepoint_names]
    lines = [",".join(header), ",".join(row)]
    (annot_dir / "dendrite_links_wide.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_dendrite(
    dendrite_dir: Path, dest_root: Path, animal_id: str, timepoint_names: List[str]
) -> int:
    fov = _fov_from_dendrite_name(dendrite_dir.name)
    tables = dendrite_dir / "Tables"
    detected_csv = tables / "Detected_spines_4D.csv"
    volume_csv = tables / "Volume_4D.csv"
    registered_tif = dendrite_dir / "Registered" / "Registered_images_4d.tif"

    if not detected_csv.is_file():
        raise FileNotFoundError(f"Missing {detected_csv}")

    df = pd.read_csv(detected_csv)
    df["spine_id"] = df["spine_id"].astype(str)
    df["timepoint"] = df["timepoint"].astype(int)

    unique_ids = list(dict.fromkeys(df["spine_id"].tolist()))  # first-appearance order
    if volume_csv.is_file():
        with volume_csv.open(encoding="utf-8") as fh:
            vol_rows = sum(1 for _ in fh) - 1
        if vol_rows != len(unique_ids):
            raise ValueError(
                f"{dendrite_dir.name}: unique spine_id count ({len(unique_ids)}) "
                f"!= Volume_4D.csv row count ({vol_rows}) -- layout assumption broken, stop and check."
            )

    global_id_by_local = {sid: _format_global_id(i + 1) for i, sid in enumerate(unique_ids)}

    respan_root = dest_root / "respan"
    annot_dir = respan_root / "_annotator" / f"fov{fov}"
    annot_dir.mkdir(parents=True, exist_ok=True)

    stack = None
    if registered_tif.is_file():
        stack = tifffile.imread(str(registered_tif))  # (T, Z, Y, X)
        if stack.shape[0] != len(timepoint_names):
            raise ValueError(
                f"{dendrite_dir.name}: registered stack has {stack.shape[0]} frames, "
                f"expected {len(timepoint_names)}"
            )
        # T01+ frames carry a registration/warp padding artifact: real signal
        # tops out ~300, then a dense cluster sits at ~65485-65535 with a
        # total gap in between (confirmed per-frame on Dendrite_01: real max
        # 276-302, artifact cluster starts 65485-65505, nothing between).
        # Zero it out so percentile autocontrast isn't wrecked by it.
        stack = stack.copy()
        stack[stack >= ARTIFACT_CLIP_THRESHOLD] = 0

    catalog_rows: List[dict] = []
    presence: Dict[str, List[str]] = {sid: [] for sid in unique_ids}
    positions: Dict[str, Dict[str, Tuple[float, float, float]]] = {sid: {} for sid in unique_ids}

    for t_idx, tp_name in enumerate(timepoint_names):
        tp_df = df[df["timepoint"] == t_idx + 1].copy()
        tp_dir = respan_root / tp_name
        tables_dir = tp_dir / "Tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        out_cols = ["spine_id", "x", "y", "z", "dendrite_id"] + [
            c for c in tp_df.columns if c not in {"timepoint", "spine_id", "x", "y", "z"}
        ]
        tp_df["dendrite_id"] = DENDRITE_ID
        csv_path = tables_dir / f"fov{fov}_detected_spines.csv"
        tp_df[out_cols].to_csv(csv_path, index=False)

        if stack is not None:
            tif_path = tp_dir / f"fov{fov}.tif"
            tifffile.imwrite(str(tif_path), stack[t_idx])

        for _, row in tp_df.iterrows():
            sid = str(row["spine_id"])
            gid = global_id_by_local[sid]
            catalog_rows.append(
                {
                    "global_spine_id": gid,
                    "animal_id": animal_id,
                    "fov": str(fov),
                    "timepoint": tp_name,
                    "local_spine_id": sid,
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "dendrite_id": DENDRITE_ID,
                    "source_csv": str(csv_path),
                }
            )
            presence[sid].append(tp_name)
            positions[sid][tp_name] = (row["x"], row["y"], row["z"])

    _write_dendrite_links(annot_dir, animal_id, fov, timepoint_names)

    # spine_catalog.csv -- written after every per-TP CSV so its mtime is newest
    # (ensure_catalog treats an existing, non-stale catalog as authoritative).
    with (annot_dir / "spine_catalog.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in catalog_rows:
            writer.writerow(row)

    # spine_registry_wide.csv: pre-fill offer, and a gap-flag companion for manual QC.
    registry_header = ["fov", "lineage_id"]
    for tp in timepoint_names:
        registry_header += [f"id_{tp}", f"status_{tp}", f"{tp}_x", f"{tp}_y", f"{tp}_z"]
    registry_lines = [",".join(registry_header)]
    gap_rows: List[dict] = []
    for sid in unique_ids:
        gid = global_id_by_local[sid]
        present = presence[sid]
        present_idx = sorted(timepoint_names.index(tp) for tp in present)
        has_gap = (present_idx[-1] - present_idx[0] + 1) != len(present_idx)
        cells = [str(fov), gid]
        for tp in timepoint_names:
            if tp in positions[sid]:
                x, y, z = positions[sid][tp]
                cells += [gid, "", f"{x}", f"{y}", f"{z}"]
            else:
                cells += ["", "", "", "", ""]
        registry_lines.append(",".join(cells))
        if has_gap:
            gap_tps = [
                tp for tp in timepoint_names[present_idx[0] : present_idx[-1] + 1] if tp not in present
            ]
            gap_rows.append(
                {
                    "lineage_id": gid,
                    "spine_id": sid,
                    "present_tps": ";".join(present),
                    "gap_tps": ";".join(gap_tps),
                }
            )
    (annot_dir / "spine_registry_wide.csv").write_text("\n".join(registry_lines) + "\n", encoding="utf-8")

    with (annot_dir / "respan_ready_gap_flags.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["lineage_id", "spine_id", "present_tps", "gap_tps"])
        writer.writeheader()
        for r in gap_rows:
            writer.writerow(r)

    print(
        f"fov{fov} ({dendrite_dir.name}): {len(unique_ids)} lineages, "
        f"{len(catalog_rows)} detections, {len(gap_rows)} with gaps"
    )
    return fov


_TRACKED_MODE_OWNED_FILES = [
    "spine_catalog.csv",
    "dendrite_links.json",
    "dendrite_links_wide.csv",
    "spine_registry_wide.csv",
    "respan_ready_gap_flags.csv",
]


def materialize_dendrite_single_tp(
    dendrite_dir: Path, dest_root: Path, animal_id: str, timepoint_names: List[str]
) -> int:
    """No cross-TP link, no registration -- just reshape Dendrite_NN-per-folder
    into this app's TP-per-folder layout. Real per-spine dendrite_id passes
    through untouched, so Phase 0 dendrite linking is meaningful (not
    trivial) for this data, same as a normal fresh dataset. No catalog/
    registry/dendrite_links pre-seed -- nothing here is pre-decided."""
    fov = _fov_from_dendrite_name(dendrite_dir.name)
    csv_by_tp = _single_tp_csv_map(dendrite_dir)
    respan_root = dest_root / "respan"

    # Drop any tracked-mode pre-fill files left over in this fov's _annotator
    # dir from a previous run against different (tracked-mode) source data --
    # they'd otherwise sit there stale and get read by the app as if valid.
    annot_dir = respan_root / "_annotator" / f"fov{fov}"
    stale = [annot_dir / name for name in _TRACKED_MODE_OWNED_FILES if (annot_dir / name).is_file()]
    if stale:
        for p in stale:
            p.unlink()
        print(f"fov{fov}: removed {len(stale)} stale tracked-mode file(s) from a prior run: "
              f"{', '.join(p.name for p in stale)}")
    review_state = [
        annot_dir / name
        for name in ("lineage_decisions.json", "spine_review_progress.json", "spine_disposition.csv")
        if (annot_dir / name).is_file()
    ]
    if review_state:
        print(
            f"fov{fov}: WARNING -- {', '.join(p.name for p in review_state)} still present from a "
            f"prior run against different source data; those decisions reference spine ids that no "
            f"longer exist in the current CSVs. Not touched automatically -- review/clear manually if stale."
        )

    n_spines_total = 0
    for tp_name in timepoint_names:
        tp_dir = respan_root / tp_name
        tables_dir = tp_dir / "Tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        src_csv = csv_by_tp.get(tp_name)
        if src_csv is not None:
            dst_csv = tables_dir / f"fov{fov}_detected_spines.csv"
            shutil.copyfile(src_csv, dst_csv)
            with dst_csv.open(encoding="utf-8") as fh:
                n_spines_total += max(sum(1 for _ in fh) - 1, 0)

        src_tif = dendrite_dir / f"{dendrite_dir.name}_{tp_name}.tif"
        if src_tif.is_file():
            tp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_tif, tp_dir / f"fov{fov}.tif")

    print(f"fov{fov} ({dendrite_dir.name}, single-TP mode): {n_spines_total} detections, no cross-TP link")
    return fov


def materialize_all(
    source_root: Path, dest_root: Path, animal_id: str, *, only: Optional[List[str]] = None
) -> Tuple[List[int], List[str]]:
    dendrite_dirs = _dendrite_folders(source_root)
    if only:
        wanted = {n.lower() for n in only}
        dendrite_dirs = [d for d in dendrite_dirs if d.name.lower() in wanted]
    if not dendrite_dirs:
        raise SystemExit(f"No Dendrite_NN folders found under {source_root}")

    modes = {_dendrite_mode(d) for d in dendrite_dirs}
    modes.discard(None)
    if not modes:
        raise SystemExit(f"No recognizable dendrite output (tracked or single-TP) under {source_root}")
    if len(modes) > 1:
        raise SystemExit(f"Mixed dendrite output modes under {source_root}: {sorted(modes)} -- stop and check")
    mode = modes.pop()

    if mode == "tracked":
        timepoint_names = _discover_timepoint_names(dendrite_dirs)
        fovs = [materialize_dendrite(d, dest_root, animal_id, timepoint_names) for d in dendrite_dirs]
    else:
        timepoint_names = _discover_timepoint_names_single_tp(dendrite_dirs)
        fovs = [materialize_dendrite_single_tp(d, dest_root, animal_id, timepoint_names) for d in dendrite_dirs]
    return sorted(fovs), timepoint_names


def auto_prepare_from_config(config_path: Path) -> bool:
    """If config's workspace looks like a pre-matched source, materialize it
    and rewrite the config to point at the result. No-op (returns False)
    otherwise -- covers an already-respan-shaped workspace unchanged."""
    if not config_path.is_file():
        return False
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    workspace_raw = str(raw.get("workspace") or "").strip()
    if not workspace_raw:
        return False

    project_root = Path(__file__).resolve().parents[2]
    workspace = Path(workspace_raw).expanduser()
    if not workspace.is_absolute():
        workspace = project_root / workspace
    workspace = workspace.resolve()

    mode = _detect_source_mode(workspace)
    if mode is None:
        return False

    animal_id = str(raw.get("animal_id") or "").strip() or workspace.name
    dest = workspace.parent / "respan_ready_bootstrap"
    print(f"[respan_ready_import] Source detected at {workspace} (mode: {mode})")
    print(f"[respan_ready_import] Materializing into {dest} ...")
    fovs, timepoint_names = materialize_all(workspace, dest, animal_id)

    raw["workspace"] = str(dest)
    raw["animal_id"] = animal_id
    raw["fovs"] = fovs
    raw["timepoint_order"] = timepoint_names
    raw["active_timepoints"] = []
    config_path.write_text(json.dumps(raw, indent=4), encoding="utf-8")
    print(f"[respan_ready_import] {config_path} updated: workspace -> {dest}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="ReSpAn_ready root (contains Dendrite_NN folders)")
    ap.add_argument("--dest", help="Destination workspace root (respan/ is created inside it)")
    ap.add_argument("--animal-id", default="GP04")
    ap.add_argument("--only", nargs="*", help="Optional subset of dendrite folder names, e.g. Dendrite_01")
    ap.add_argument(
        "--from-config",
        help="Path to annotator.json: auto-detect+materialize if its workspace looks like "
        "a pre-matched source, rewriting the config to point at the result; no-op otherwise. "
        "Ignores --source/--dest/--animal-id/--only when given.",
    )
    args = ap.parse_args()

    if args.from_config:
        did = auto_prepare_from_config(Path(args.from_config))
        if not did:
            print("[respan_ready_import] Nothing to do (workspace is not a pre-matched source).")
        return

    if not args.source or not args.dest:
        raise SystemExit("--source and --dest are required unless --from-config is given")

    materialize_all(Path(args.source), Path(args.dest), args.animal_id, only=args.only)


if __name__ == "__main__":
    main()
