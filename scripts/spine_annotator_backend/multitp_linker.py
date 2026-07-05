"""Phase 0: link dendrites across all 5 timepoints, up front.

Self-contained, additive module. Does NOT touch the pairwise (t1/t2) matching
engine. It loads up to 5 timepoint detection CSVs (and optional TIFFs for FOV
preview), lets the user group dendrite IDs into cross-timepoint "dendrite
lineages", and persists them in the same wide schema used by
``spine_summary/dendrite_links_wide.csv`` (plus a round-trippable JSON).

The output unlocks dendrite-relative location prediction for the live
multi-timepoint spine viewer (later phases).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import baseline_adapter, io_service
from . import animal_config, dendrite_link_store

router = APIRouter(prefix="/mtp", tags=["multi-timepoint-dendrite-linker"])

DEFAULT_TIMEPOINTS = [
    "pre-droplet",
    "mid-droplet",
    "end-droplet",
    "end-lever",
    "return to droplet",
]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class _MtpState:
    def __init__(self) -> None:
        self.animal_id: str = ""
        self.fov: str = ""
        self.timepoint_names: List[str] = []
        self.files: Dict[str, Dict[str, str]] = {}          # name -> {csv, tiff}
        self.dendrite_ids: Dict[str, List[str]] = {}         # name -> [ids]
        self.spines: Dict[str, List[dict]] = {}              # name -> [{id,x,y,z,dendrite_id}]
        self._mip: Dict[str, np.ndarray] = {}                # name -> full-res MIP (Y,X)
        self.links: List[dict] = []                          # [{link_id, members, notes}]
        self.link_counter: int = 0
        self.catalog_message: str = ""

    def reset_data(self) -> None:
        self.timepoint_names = []
        self.files = {}
        self.dendrite_ids = {}
        self.spines = {}
        self._mip = {}
        self.catalog_message = ""


_STATE = _MtpState()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sort_ids(ids) -> List[str]:
    def key(s: str):
        try:
            return (0, float(s), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(s))

    return sorted({str(i) for i in ids if str(i).strip() not in {"", "nan", "None"}}, key=key)


def _next_link_id() -> str:
    _STATE.link_counter += 1
    n = _STATE.link_counter
    a = _STATE.animal_id.strip()
    f = _STATE.fov.strip()
    if a and f:
        return f"{a}_fov{f}_D{n:03d}"
    if a:
        return f"{a}_D{n:03d}"
    return f"D{n:03d}"


def _mip_for(name: str) -> np.ndarray:
    """Lazily compute and cache a full-resolution Z-MIP for a timepoint's TIFF."""
    if name in _STATE._mip:
        return _STATE._mip[name]
    tiff = (_STATE.files.get(name) or {}).get("tiff", "")
    if not tiff:
        raise HTTPException(status_code=400, detail=f"No TIFF loaded for timepoint '{name}'.")
    stack = baseline_adapter.load_stack(Path(tiff))
    arr = np.asarray(stack)
    if arr.ndim == 2:
        mip = arr.astype(np.float32)
    else:
        mip = arr.max(axis=0).astype(np.float32)
    _STATE._mip[name] = mip
    return mip


def _respan_and_fov() -> tuple[Path, int]:
    cfg = animal_config.load_config()
    respan = animal_config.require_respan(cfg)
    fov = int(_STATE.fov.strip() or cfg.default_fov or 1)
    return respan, fov


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class PickFileRequest(BaseModel):
    kind: str = "csv"           # "csv" | "tiff"
    title: str = ""


class PickFileResponse(BaseModel):
    path: str


class ConfigureTimepoint(BaseModel):
    name: str
    csv_path: str
    tiff_path: Optional[str] = None


class ConfigureRequest(BaseModel):
    animal_id: str = ""
    fov: str = ""
    timepoints: List[ConfigureTimepoint]


class TimepointInfo(BaseModel):
    name: str
    csv_path: str
    tiff_path: Optional[str] = None
    spine_count: int
    dendrite_ids: List[str]
    has_tiff: bool


class StateResponse(BaseModel):
    animal_id: str
    fov: str
    default_timepoints: List[str] = Field(default_factory=lambda: list(DEFAULT_TIMEPOINTS))
    timepoints: List[TimepointInfo]
    links: List[dict]
    visited_link_ids: List[str] = Field(default_factory=list)
    catalog_message: str = ""


class MarkVisitedRequest(BaseModel):
    link_id: str


class FovPreviewRequest(BaseModel):
    timepoint_name: str
    dendrite_ids: List[str] = Field(default_factory=list)
    max_dim: int = 480


class FovPreviewResponse(BaseModel):
    timepoint_name: str
    height: int
    width: int
    scale: float
    intensity_min: float
    intensity_max: float
    pixels: List[int]           # flat uint8, length height*width
    points: List[dict]          # [{dendrite_id, x, y}] in downsampled coords


class LinkRequest(BaseModel):
    members: Dict[str, List[str]]   # timepoint_name -> [dendrite_ids]
    notes: str = ""


class SaveResponse(BaseModel):
    ok: bool
    json_path: str
    wide_csv_path: str
    link_count: int


class LoadSavedRequest(BaseModel):
    animal_id: str = ""
    fov: str = ""
    json_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# State assembly
# --------------------------------------------------------------------------- #
def _state_response(*, fov: Optional[int] = None) -> StateResponse:
    tps: List[TimepointInfo] = []
    for name in _STATE.timepoint_names:
        files = _STATE.files.get(name, {})
        tps.append(
            TimepointInfo(
                name=name,
                csv_path=files.get("csv", ""),
                tiff_path=files.get("tiff") or None,
                spine_count=len(_STATE.spines.get(name, [])),
                dendrite_ids=_STATE.dendrite_ids.get(name, []),
                has_tiff=bool(files.get("tiff")),
            )
        )
    visited: List[str] = []
    try:
        respan, fov_i = _respan_and_fov()
        if fov is not None:
            fov_i = int(fov)
        visited = list(dendrite_link_store.load_progress(respan, fov_i).get("visited_link_ids", []))
    except Exception:
        pass
    return StateResponse(
        animal_id=_STATE.animal_id,
        fov=_STATE.fov,
        timepoints=tps,
        links=_STATE.links,
        visited_link_ids=visited,
        catalog_message=_STATE.catalog_message,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post("/pick-file", response_model=PickFileResponse)
def pick_file(req: PickFileRequest) -> PickFileResponse:
    try:
        if req.kind == "tiff":
            title = req.title or "Select TIFF stack"
            path = io_service._pick_file(title, (("TIFF files", "*.tif *.tiff"), ("All files", "*.*")))
        else:
            title = req.title or "Select detection CSV"
            path = io_service._pick_file(title, (("CSV files", "*.csv"), ("All files", "*.*")))
        return PickFileResponse(path=path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _sync_spine_catalog(respan: Path, fov: int, *, animal_id: str) -> str:
    """Ensure spine_catalog.csv exists for this FOV (called automatically on load)."""
    from . import spine_catalog_store

    csv_by_tp = {
        name: Path(files["csv"])
        for name, files in _STATE.files.items()
        if files.get("csv") and Path(files["csv"]).is_file()
    }
    if not csv_by_tp:
        return ""
    info = spine_catalog_store.sync_catalog_for_fov(
        respan,
        fov,
        animal_id=animal_id,
        timepoint_names=list(_STATE.timepoint_names),
        csv_by_tp=csv_by_tp,
    )
    _STATE.catalog_message = str(info.get("message") or "")
    return _STATE.catalog_message


@router.post("/configure", response_model=StateResponse)
def configure(req: ConfigureRequest) -> StateResponse:
    try:
        if not req.timepoints:
            raise ValueError("Provide at least one timepoint with a CSV.")
        _STATE.animal_id = req.animal_id.strip()
        _STATE.fov = req.fov.strip()
        _STATE.reset_data()
        for tp in req.timepoints:
            name = tp.name.strip()
            if not name:
                raise ValueError("Timepoint name cannot be empty.")
            csv_path = io_service.validate_existing_path(tp.csv_path, f"{name} CSV")
            df = baseline_adapter.load_spines(csv_path)
            if "dendrite_id" not in df.columns:
                raise ValueError(f"{name} CSV has no 'dendrite_id' column; cannot link dendrites.")
            df = df.copy()
            df["dendrite_id"] = df["dendrite_id"].astype(str)
            tiff_path = ""
            if tp.tiff_path:
                tiff_path = str(io_service.validate_existing_path(tp.tiff_path, f"{name} TIFF"))
            _STATE.timepoint_names.append(name)
            _STATE.files[name] = {"csv": str(csv_path), "tiff": tiff_path}
            _STATE.dendrite_ids[name] = _sort_ids(df["dendrite_id"].tolist())
            _STATE.spines[name] = [
                {
                    "id": str(r["id"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                    "dendrite_id": str(r["dendrite_id"]),
                }
                for _, r in df.iterrows()
            ]
        try:
            cfg = animal_config.load_config()
            respan = animal_config.require_respan(cfg)
            _sync_spine_catalog(respan, int(req.fov or 1), animal_id=_STATE.animal_id)
        except Exception:
            pass
        return _state_response(fov=int(req.fov) if req.fov else None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    return _state_response()


@router.post("/fov-preview", response_model=FovPreviewResponse)
def fov_preview(req: FovPreviewRequest) -> FovPreviewResponse:
    try:
        name = req.timepoint_name
        if name not in _STATE.timepoint_names:
            raise ValueError(f"Timepoint '{name}' is not loaded.")
        mip = _mip_for(name)
        h0, w0 = int(mip.shape[0]), int(mip.shape[1])
        max_dim = max(64, min(int(req.max_dim), 1200))
        scale = 1.0
        big = max(h0, w0)
        if big > max_dim:
            scale = max_dim / float(big)
        out_h = max(1, int(round(h0 * scale)))
        out_w = max(1, int(round(w0 * scale)))
        ys = np.linspace(0, h0 - 1, out_h).astype(int)
        xs = np.linspace(0, w0 - 1, out_w).astype(int)
        small = mip[np.ix_(ys, xs)]
        lo = float(np.percentile(small, 1.0))
        hi = float(np.percentile(small, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((small - lo) / (hi - lo), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        sel = {str(d) for d in req.dendrite_ids}
        points: List[dict] = []
        if sel:
            for sp in _STATE.spines.get(name, []):
                if sp["dendrite_id"] in sel:
                    points.append(
                        {
                            "dendrite_id": sp["dendrite_id"],
                            "x": float(sp["x"] * scale),
                            "y": float(sp["y"] * scale),
                        }
                    )
        return FovPreviewResponse(
            timepoint_name=name,
            height=out_h,
            width=out_w,
            scale=scale,
            intensity_min=lo,
            intensity_max=hi,
            pixels=u8.flatten().tolist(),
            points=points,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/link", response_model=StateResponse)
def create_link(req: LinkRequest) -> StateResponse:
    try:
        members: Dict[str, List[str]] = {}
        total = 0
        for name, ids in req.members.items():
            if name not in _STATE.timepoint_names:
                continue
            clean = _sort_ids(ids)
            if clean:
                members[name] = clean
                total += len(clean)
        if total == 0:
            raise ValueError("Select at least one dendrite in at least one timepoint to create a link.")
        link = {"link_id": _next_link_id(), "members": members, "notes": req.notes.strip()}
        _STATE.links.append(link)
        return _state_response()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/link/{link_id}", response_model=StateResponse)
def delete_link(link_id: str) -> StateResponse:
    _STATE.links = [l for l in _STATE.links if str(l.get("link_id")) != str(link_id)]
    return _state_response()


@router.delete("/links", response_model=StateResponse)
def clear_links() -> StateResponse:
    _STATE.links = []
    _STATE.link_counter = 0
    return _state_response()


@router.post("/auto-link-shared", response_model=StateResponse)
def auto_link_shared() -> StateResponse:
    """Bootstrap: one link per dendrite ID that appears (identically) in >= 2 timepoints."""
    try:
        all_ids: Dict[str, Dict[str, bool]] = {}
        for name in _STATE.timepoint_names:
            for did in _STATE.dendrite_ids.get(name, []):
                all_ids.setdefault(did, {})[name] = True
        existing = {
            tuple(sorted((tp, did) for tp, ids in l["members"].items() for did in ids))
            for l in _STATE.links
        }
        for did in _sort_ids(all_ids.keys()):
            present = [n for n in _STATE.timepoint_names if all_ids[did].get(n)]
            if len(present) < 2:
                continue
            members = {n: [did] for n in present}
            sig = tuple(sorted((tp, d) for tp, ids in members.items() for d in ids))
            if sig in existing:
                continue
            _STATE.links.append({"link_id": _next_link_id(), "members": members, "notes": "auto-shared-id"})
        return _state_response()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/save", response_model=SaveResponse)
def save_links() -> SaveResponse:
    try:
        if not _STATE.links:
            raise ValueError("No dendrite links to save.")
        respan, fov = _respan_and_fov()
        paths = dendrite_link_store.save_links(
            respan,
            fov,
            animal_id=_STATE.animal_id or animal_config.load_config().animal_id,
            timepoint_names=_STATE.timepoint_names,
            links=_STATE.links,
            files=_STATE.files,
        )
        return SaveResponse(
            ok=True,
            json_path=paths["json"],
            wide_csv_path=paths["wide"],
            link_count=len(_STATE.links),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load-saved", response_model=StateResponse)
def load_saved(req: LoadSavedRequest) -> StateResponse:
    try:
        data: Optional[dict] = None
        if req.json_path:
            data = json.loads(Path(req.json_path).read_text(encoding="utf-8"))
            _STATE.links = list(data.get("links", []))
        else:
            respan, fov = _respan_and_fov()
            if req.fov:
                fov = int(req.fov)
            paths = dendrite_link_store.link_paths(respan, fov)
            if paths["json"].is_file():
                data = json.loads(paths["json"].read_text(encoding="utf-8"))
                _STATE.links = list(data.get("links", []))
            elif paths["wide"].is_file():
                _STATE.links = dendrite_link_store.wide_csv_to_links(paths["wide"])
            else:
                raise FileNotFoundError(
                    f"No dendrite links at {paths['json']} or {paths['wide']}"
                )
        _STATE.link_counter = max(len(_STATE.links), _STATE.link_counter)
        if data is not None:
            if data.get("animal_id"):
                _STATE.animal_id = str(data["animal_id"])
            if data.get("fov"):
                _STATE.fov = str(data["fov"])
            tps = data.get("timepoint_names")
            if tps:
                _STATE.timepoint_names = list(tps)
            files = data.get("files")
            if files:
                _STATE.files = dict(files)
        elif _STATE.links and not _STATE.timepoint_names:
            keys = sorted({k for l in _STATE.links for k in (l.get("members") or {})})
            _STATE.timepoint_names = keys
        return _state_response(fov=int(req.fov) if req.fov else None)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load-from-animal", response_model=StateResponse)
def load_from_animal(fov: int = 1, timepoints: str = "") -> StateResponse:
    """Auto-configure timepoints from respan; optional comma-separated subset."""
    try:
        from . import animal_config, animal_layout, timepoint_selection

        cfg = animal_config.load_config()
        respan = animal_config.require_respan_root(cfg)
        requested = timepoint_selection.parse_timepoint_query(timepoints)
        saved = timepoint_selection.load(respan, fov)
        available = animal_layout.discover_available_timepoints(respan, fov)
        selected = animal_config.resolve_active_timepoints(
            cfg,
            available,
            requested=requested or None,
            saved=saved or None,
        )
        if requested:
            timepoint_selection.save(respan, fov, selected, animal_id=cfg.animal_id)
        inv = animal_layout.build_fov_inventory(
            respan, fov, animal_id=cfg.animal_id, selected_timepoints=selected
        )
        tps: List[ConfigureTimepoint] = []
        for row in inv.timepoints:
            if not row.csv_path:
                continue
            tps.append(
                ConfigureTimepoint(
                    name=row.name,
                    csv_path=row.csv_path,
                    tiff_path=row.tiff_path,
                )
            )
        if not tps:
            raise ValueError(f"No detection CSVs found for FOV {fov} under {respan}")
        configure(
            ConfigureRequest(animal_id=cfg.animal_id, fov=str(fov), timepoints=tps)
        )
        paths = dendrite_link_store.link_paths(respan, fov)
        if paths["json"].is_file():
            data = json.loads(paths["json"].read_text(encoding="utf-8"))
            _STATE.links = list(data.get("links", []))
            _STATE.link_counter = max(len(_STATE.links), _STATE.link_counter)
        elif paths["wide"].is_file():
            _STATE.links = dendrite_link_store.wide_csv_to_links(paths["wide"])
            _STATE.link_counter = max(len(_STATE.links), _STATE.link_counter)
        _sync_spine_catalog(respan, fov, animal_id=cfg.animal_id)
        return _state_response(fov=fov)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/progress")
def get_progress(fov: int = 1) -> dict:
    try:
        cfg = animal_config.load_config()
        respan = animal_config.require_respan(cfg)
        return dendrite_link_store.load_progress(respan, fov)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/mark-visited")
def mark_visited(req: MarkVisitedRequest) -> dict:
    try:
        respan, fov = _respan_and_fov()
        prog = dendrite_link_store.mark_link_visited(respan, fov, req.link_id)
        return {"ok": True, **prog}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/", response_class=HTMLResponse)
def linker_page() -> str:
    return _PAGE_HTML


_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Phase 1 · Dendrite Links</title>
<style>
  :root {
    --bg:#f7f8fa; --panel:#ffffff; --ink:#1b1f24; --muted:#697077;
    --line:#e3e7ec; --blue:#2563eb; --blue-soft:#e8f0fe; --red:#dc2626;
    --accent:#0f766e;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); font-size:13px; }
  header { padding:12px 18px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:14px; position:sticky; top:0; z-index:5; }
  header h1 { font-size:15px; font-weight:600; margin:0; letter-spacing:.2px; }
  header .sub { color:var(--muted); font-size:12px; }
  .wrap { padding:16px 18px 60px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:14px 16px; margin-bottom:16px; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted);
              margin:0 0 12px; font-weight:600; }
  label { display:block; font-size:11px; color:var(--muted); margin-bottom:3px; }
  input[type=text] { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px;
                     font-size:12px; background:#fff; color:var(--ink); }
  button { font:inherit; font-size:12px; padding:7px 12px; border-radius:7px; cursor:pointer;
           border:1px solid var(--line); background:#fff; color:var(--ink); transition:.12s; }
  button:hover { border-color:#c4ccd6; }
  button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
  button.primary:hover { background:#1d4ed8; }
  button.ghost { background:transparent; }
  button.danger { color:var(--red); border-color:#f1c4c4; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
  .idrow { display:flex; gap:12px; margin-bottom:12px; }
  .idrow > div { flex:1; }
  table.setup { width:100%; border-collapse:collapse; }
  table.setup th { text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:.5px;
                   color:var(--muted); padding:4px 6px; font-weight:600; }
  table.setup td { padding:4px 6px; vertical-align:middle; }
  table.setup .pathcell { display:flex; gap:6px; }
  table.setup .pathcell input { flex:1; }
  .toolbar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; align-items:center; }
  .columns { display:flex; gap:12px; overflow-x:auto; padding-bottom:8px; }
  .col { flex:1 1 0; min-width:240px; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; display:flex; flex-direction:column; }
  .col .chead { padding:9px 11px; border-bottom:1px solid var(--line); }
  .col .chead .name { font-weight:600; font-size:12.5px; }
  .col .chead .meta { color:var(--muted); font-size:11px; }
  .col canvas { width:100%; display:block; background:#000; border-bottom:1px solid var(--line); }
  .col .idlist { max-height:260px; overflow-y:auto; padding:6px 4px; }
  .idchip { display:flex; align-items:center; gap:7px; padding:3px 8px; border-radius:6px;
            cursor:pointer; font-size:12px; transition:background .12s; }
  .idchip:hover { background:var(--blue-soft); }
  .idchip.selected { background:#d1d5db; }
  .idchip.selected:hover { background:#c4cbd4; }
  .idchip input { margin:0; pointer-events:none; }
  .swatch { width:10px; height:10px; border-radius:50%; flex:0 0 auto; }
  .idchip .lk { margin-left:auto; font-size:10px; color:var(--accent); }
  .links { display:flex; flex-direction:column; gap:8px; }
  .linkitem { display:flex; align-items:center; gap:10px; padding:8px 10px; border:1px solid var(--line);
              border-radius:8px; background:#fcfdff; }
  .linkitem .lid { font-weight:600; font-size:12px; min-width:120px; }
  .linkitem .mem { display:flex; gap:8px; flex-wrap:wrap; flex:1; }
  .memtag { font-size:11px; background:#f1f4f8; border:1px solid var(--line); border-radius:5px;
            padding:2px 7px; }
  .memtag b { color:var(--blue); }
  .status { position:fixed; bottom:0; left:0; right:0; padding:7px 18px; background:#11161d;
            color:#d7dde6; font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .hidden { display:none; }
  .hint { color:var(--muted); font-size:11px; margin-top:6px; }
  table.linkhub { width:100%; border-collapse:collapse; margin-top:8px; }
  table.linkhub th { text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:.5px;
                     color:var(--muted); padding:8px 10px; border-bottom:2px solid var(--line); font-weight:600; }
  table.linkhub td { padding:10px; border-bottom:1px solid var(--line); font-size:12px; vertical-align:middle; }
  tr.linkrow { cursor:pointer; transition:background .12s; }
  tr.linkrow:hover { background:var(--blue-soft); }
  tr.linkrow.visited { background:#e5e7eb; color:#6b7280; }
  tr.linkrow.visited:hover { background:#d1d5db; }
  tr.linkrow.visited td { color:#6b7280; }
  .badge { font-size:10px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
  .badge.pending { background:#fff; color:var(--ink); }
  .badge.done { background:#9ca3af; color:#fff; border-color:#9ca3af; }
  .hub-toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:8px; }
  .progress-hint { font-size:12px; color:var(--muted); margin:0 0 8px; }
</style>
</head>
<body>
<header>
  <h1>Phase 1 · Dendrite Links</h1>
  <span class="sub">entry point — click a link to analyze spines · visited links stay gray</span>
  <a href="/mtp/viewer/" style="margin-left:auto;font-size:12px;">Spine tracker</a>
  <a href="/" style="font-size:12px;">Matcher</a>
</header>

<div class="wrap">
  <div class="panel" id="hubPanel">
    <h2>Dendrite link list</h2>
    <div id="animalBanner" class="hint" style="margin-bottom:10px; padding:8px 10px; background:#eef4ff; border-radius:6px;">Checking animal config...</div>
    <div class="hub-toolbar">
      <div><label>Animal</label><input type="text" id="animalId" placeholder="GP04" style="width:72px"/></div>
      <div><label>FOV</label><input type="text" id="fovId" placeholder="1" style="width:48px"/></div>
      <button class="primary" onclick="bootstrapHub()">Reload FOV</button>
      <button class="primary" onclick="startSpineMatching()" style="background:#0f766e;border-color:#0f766e">Start spine matching</button>
      <button onclick="autoLink()">Auto-link identical IDs</button>
      <button onclick="saveLinks()">Save links</button>
      <span style="flex:1"></span>
      <span id="linkCountLabel" class="hint" style="margin:0">0 links</span>
    </div>
    <div id="tpPick" class="hint" style="margin:8px 0;padding:8px 10px;background:#f8fafc;border-radius:6px;border:1px solid var(--line);"></div>
    <p class="progress-hint">Use <b>Start spine matching</b> to review all pre spines (dendrite links guide matching). Or click a row to focus one dendrite link.</p>
    <table class="linkhub">
      <thead id="linkHubHead"><tr><th>Link ID</th><th>Status</th></tr></thead>
      <tbody id="linkHubBody"></tbody>
    </table>
    <div id="emptyHub" class="hint hidden" style="padding:16px;text-align:center;">No dendrite links yet. Expand “Create links” below, or run Auto-link.</div>
  </div>

  <details class="panel" id="setupPanel" style="padding:14px 16px;">
    <summary style="cursor:pointer;font-weight:600;font-size:12px;color:var(--muted);">Create / edit dendrite links</summary>
    <details style="margin-top:14px">
      <summary style="cursor:pointer; color:#555; font-size:12px;">Manual file paths (fallback)</summary>
    <table class="setup" style="margin-top:10px">
      <thead><tr><th style="width:150px">Timepoint</th><th>Detection CSV (required)</th><th>TIFF stack (optional, for preview)</th></tr></thead>
      <tbody id="setupRows"></tbody>
    </table>
    <div class="hint">Dendrite IDs come from the CSV's <code>dendrite_id</code> column. A TIFF lets you see the FOV to identify dendrites.</div>
    <div class="row" style="margin-top:12px">
      <button onclick="loadTimepoints()">Load from manual paths</button>
      <button class="ghost" onclick="loadFromAnimal()">Reload from animal folder</button>
    </div>
  </details>

  <div class="panel hidden" id="workPanel">
    <h2>Group dendrites into cross-timepoint links</h2>
    <div class="toolbar">
      <button class="primary" onclick="createLink()">Create link from selection</button>
      <button onclick="autoLink()">Auto-link identical IDs</button>
      <button class="ghost" onclick="clearSelection()">Clear selection</button>
      <span style="flex:1"></span>
      <button onclick="saveLinks()">Save links</button>
      <button class="danger ghost" onclick="clearLinks()">Clear all links</button>
    </div>
    <div class="columns" id="columns"></div>
  </div>
  </details>
</div>

<div class="status" id="status">Loading dendrite links…</div>

<script>
const PALETTE = ["#2563eb","#dc2626","#0f766e","#d97706","#7c3aed","#0891b2","#be123c",
                 "#4d7c0f","#9333ea","#c2410c","#0369a1","#15803d","#b91c1c","#6d28d9"];
let DEFAULT_TPS = ["pre-droplet","mid-droplet","end-droplet","end-lever","return to droplet"];
let STATE = null;
const SEL = {};                 // timepoint -> Set(dendrite_id)
const previewTimers = {};

function setStatus(msg){ document.getElementById('status').textContent = msg; }
function colorForId(id){
  let h=0; const s=String(id); for(let i=0;i<s.length;i++){ h=(h*31+s.charCodeAt(i))>>>0; }
  return PALETTE[h % PALETTE.length];
}
async function api(url, opts){
  const r = await fetch(url, opts);
  if(!r.ok){ const t = await r.text(); throw new Error(t || (r.status+' '+r.statusText)); }
  return await r.json();
}

function renderSetup(){
  const tb = document.getElementById('setupRows');
  tb.innerHTML = '';
  DEFAULT_TPS.forEach((name, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input type="text" class="tpName" value="${name}"/></td>
      <td><div class="pathcell"><input type="text" class="tpCsv" placeholder="...detected_spines.csv"/>
          <button onclick="browse(${i},'csv')">Browse</button></div></td>
      <td><div class="pathcell"><input type="text" class="tpTiff" placeholder="...stack.tif"/>
          <button onclick="browse(${i},'tiff')">Browse</button></div></td>`;
    tb.appendChild(tr);
  });
}
async function browse(i, kind){
  try{
    setStatus('Opening file dialog...');
    const res = await api('/mtp/pick-file', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({kind, title: kind==='tiff'?'Select TIFF stack':'Select detection CSV'})});
    const sel = kind==='tiff' ? '.tpTiff' : '.tpCsv';
    document.querySelectorAll('#setupRows tr')[i].querySelector(sel).value = res.path;
    setStatus('Selected: '+res.path);
  }catch(e){ setStatus('Dialog cancelled or failed: '+e.message); }
}

async function loadTimepoints(){
  const rows = [...document.querySelectorAll('#setupRows tr')];
  const tps = [];
  rows.forEach(r=>{
    const name = r.querySelector('.tpName').value.trim();
    const csv = r.querySelector('.tpCsv').value.trim();
    const tiff = r.querySelector('.tpTiff').value.trim();
    if(name && csv) tps.push({name, csv_path:csv, tiff_path: tiff||null});
  });
  if(!tps.length){ setStatus('Add at least one timepoint with a CSV.'); return; }
  try{
    setStatus('Loading '+tps.length+' timepoint(s)...');
    STATE = await api('/mtp/configure', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({animal_id: document.getElementById('animalId').value,
                            fov: document.getElementById('fovId').value, timepoints: tps})});
    STATE.timepoints.forEach(tp=>{ if(!SEL[tp.name]) SEL[tp.name]=new Set(); });
    renderColumns(); renderLinkHub();
    document.getElementById('workPanel').classList.remove('hidden');
    setStatus('Loaded. Select dendrites in each timepoint, then "Create link".');
    STATE.timepoints.forEach(tp=>{ if(tp.has_tiff) schedulePreview(tp.name); });
  }catch(e){ setStatus('Load failed: '+e.message); }
}

function renderColumns(){
  const c = document.getElementById('columns'); c.innerHTML='';
  STATE.timepoints.forEach(tp=>{
    const col = document.createElement('div'); col.className='col';
    const ids = tp.dendrite_ids.map(id=>{
      const on = SEL[tp.name] && SEL[tp.name].has(id);
      const checked = on ? 'checked':'';
      const selCls = on ? ' selected':'';
      return `<label class="idchip${selCls}" data-tp="${tp.name.replace(/"/g,'&quot;')}" data-id="${id}"
        onclick="rowPick(event,'${tp.name.replace(/'/g,"\\\\'")}','${id}')">
        <input type="checkbox" ${checked}/>
        <span class="swatch" style="background:${colorForId(id)}"></span>
        <span>dendrite ${id}</span></label>`;
    }).join('');
    col.innerHTML = `
      <div class="chead"><div class="name">${tp.name}</div>
        <div class="meta">${tp.spine_count} spines &middot; ${tp.dendrite_ids.length} dendrites${tp.has_tiff?'':' &middot; no TIFF'}</div></div>
      ${tp.has_tiff ? `<canvas id="cv_${cssId(tp.name)}" width="10" height="10"></canvas>`:''}
      <div class="idlist">${ids||'<div class="hint" style="padding:8px">no dendrites</div>'}</div>`;
    c.appendChild(col);
  });
}
function cssId(name){ return name.replace(/[^a-zA-Z0-9]/g,'_'); }

function togglePick(tp, id, on){
  if(!SEL[tp]) SEL[tp]=new Set();
  if(on) SEL[tp].add(id); else SEL[tp].delete(id);
  syncChipHighlight(tp, id, on);
  schedulePreview(tp);
}
function rowPick(ev, tp, id){
  ev.preventDefault();
  const on = !(SEL[tp] && SEL[tp].has(id));
  togglePick(tp, id, on);
}
function syncChipHighlight(tp, id, on){
  document.querySelectorAll('.idchip').forEach(el=>{
    if(el.dataset.tp===tp && el.dataset.id===String(id)){
      el.classList.toggle('selected', !!on);
      const cb = el.querySelector('input[type=checkbox]');
      if(cb) cb.checked = !!on;
    }
  });
}
function clearSelection(){ Object.keys(SEL).forEach(k=>SEL[k].clear()); renderColumns();
  STATE.timepoints.forEach(tp=>{ if(tp.has_tiff) schedulePreview(tp.name); }); }

function schedulePreview(tp){
  const info = STATE.timepoints.find(t=>t.name===tp);
  if(!info || !info.has_tiff) return;
  clearTimeout(previewTimers[tp]);
  previewTimers[tp] = setTimeout(()=>refreshPreview(tp), 180);
}
async function refreshPreview(tp){
  try{
    const data = await api('/mtp/fov-preview', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({timepoint_name:tp, dendrite_ids:[...(SEL[tp]||[])], max_dim:480})});
    drawFov(data);
  }catch(e){ /* preview is best-effort */ }
}
function drawFov(data){
  const cv = document.getElementById('cv_'+cssId(data.timepoint_name));
  if(!cv) return;
  const off = document.createElement('canvas'); off.width=data.width; off.height=data.height;
  const octx = off.getContext('2d');
  const img = octx.createImageData(data.width, data.height);
  const px = data.pixels;
  for(let i=0;i<px.length;i++){ const v=px[i]; const j=i*4; img.data[j]=v; img.data[j+1]=v; img.data[j+2]=v; img.data[j+3]=255; }
  octx.putImageData(img,0,0);
  const dispW = cv.clientWidth || 240;
  const dispH = Math.round(dispW * data.height / data.width);
  cv.width = dispW; cv.height = dispH;
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0,0, data.width, data.height, 0,0, dispW, dispH);
  const sx = dispW / data.width, sy = dispH / data.height;
  data.points.forEach(p=>{
    ctx.beginPath(); ctx.arc(p.x*sx, p.y*sy, 3.5, 0, 6.283);
    ctx.fillStyle = colorForId(p.dendrite_id); ctx.globalAlpha=.9; ctx.fill();
    ctx.globalAlpha=1; ctx.lineWidth=1; ctx.strokeStyle='#fff'; ctx.stroke();
  });
}

async function createLink(){
  const members = {};
  Object.keys(SEL).forEach(tp=>{ if(SEL[tp].size) members[tp]=[...SEL[tp]]; });
  if(!Object.keys(members).length){ setStatus('Select dendrites first.'); return; }
  try{
    STATE = await api('/mtp/link', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({members, notes:''})});
    Object.keys(SEL).forEach(k=>SEL[k].clear());
    renderColumns(); renderLinkHub();
    STATE.timepoints.forEach(tp=>{ if(tp.has_tiff) schedulePreview(tp.name); });
    setStatus('Link created. '+STATE.links.length+' total.');
  }catch(e){ setStatus('Create link failed: '+e.message); }
}
async function autoLink(){
  try{ STATE = await api('/mtp/auto-link-shared',{method:'POST'}); renderLinkHub();
    setStatus('Auto-linked identical IDs. '+STATE.links.length+' total.'); }
  catch(e){ setStatus('Auto-link failed: '+e.message); }
}
async function deleteLink(id){
  try{ STATE = await api('/mtp/link/'+encodeURIComponent(id),{method:'DELETE'}); renderLinkHub(); }
  catch(e){ setStatus('Delete failed: '+e.message); }
}
async function clearLinks(){
  if(!confirm('Remove ALL dendrite links?')) return;
  try{ STATE = await api('/mtp/links',{method:'DELETE'}); renderLinkHub(); setStatus('All links cleared.'); }
  catch(e){ setStatus('Clear failed: '+e.message); }
}
async function saveLinks(){
  try{ const res = await api('/mtp/save',{method:'POST'});
    setStatus('Saved '+res.link_count+' links -> '+res.wide_csv_path);
    await bootstrapHub(false); }
  catch(e){ setStatus('Save failed: '+e.message); }
}
async function loadSaved(){
  try{
    const s = await api('/mtp/load-saved', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({animal_id: document.getElementById('animalId').value,
                            fov: document.getElementById('fovId').value})});
    STATE = s; renderLinkHub();
    setStatus('Loaded '+s.links.length+' saved links.');
  }catch(e){ setStatus('Load saved failed: '+e.message); }
}

async function loadFromAnimal(){ await bootstrapHub(true); }

function renderLinkHub(){
  const links = (STATE && STATE.links) || [];
  const visited = new Set((STATE && STATE.visited_link_ids) || []);
  const names = (STATE && STATE.timepoints && STATE.timepoints.length)
    ? STATE.timepoints.map(t=>t.name) : DEFAULT_TPS;
  document.getElementById('linkCountLabel').textContent = links.length+' link(s) · '+visited.size+' visited';
  const head = document.getElementById('linkHubHead');
  head.innerHTML = '<tr><th>Link ID</th>'+names.map(n=>'<th>'+n+'</th>').join('')+'<th>Status</th></tr>';
  const tbody = document.getElementById('linkHubBody');
  tbody.innerHTML = '';
  document.getElementById('emptyHub').classList.toggle('hidden', links.length > 0);
  links.forEach(link=>{
    const tr = document.createElement('tr');
    tr.className = 'linkrow' + (visited.has(link.link_id) ? ' visited' : '');
    tr.title = 'Click to analyze spines for this dendrite link';
    tr.onclick = ()=> openLinkAnalysis(link.link_id);
    const memCells = names.map(n=>{
      const ids = (link.members||{})[n] || [];
      if(!ids.length) return '<td style="opacity:.45">—</td>';
      return '<td>'+ids.map(id=>'<span class="swatch" style="background:'+colorForId(id)+';display:inline-block;vertical-align:middle;margin-right:4px"></span><b>'+id+'</b>').join(' ')+'</td>';
    }).join('');
    const badge = visited.has(link.link_id)
      ? '<span class="badge done">Visited</span>' : '<span class="badge pending">Pending</span>';
    tr.innerHTML = '<td><b>'+link.link_id+'</b></td>'+memCells+'<td>'+badge+'</td>';
    tbody.appendChild(tr);
  });
}

async function openLinkAnalysis(linkId){
  const fov = document.getElementById('fovId').value || '1';
  try{
    setStatus('Opening analysis for '+linkId+'…');
    const res = await api('/mtp/mark-visited', {method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({link_id: linkId})});
    if(!STATE.visited_link_ids) STATE.visited_link_ids = [];
    if(res.visited_link_ids) STATE.visited_link_ids = res.visited_link_ids;
    renderLinkHub();
    window.location.href = '/mtp/viewer/?link_id='+encodeURIComponent(linkId)+'&fov='+encodeURIComponent(fov)+(selectedTimepointsQuery() ? '&timepoints='+encodeURIComponent(selectedTimepointsQuery()) : '');
  }catch(e){ setStatus('Failed to open link: '+e.message); }
}

function startSpineMatching(){
  const fov = document.getElementById('fovId').value || '1';
  const tps = selectedTimepointsQuery();
  window.location.href = '/mtp/viewer/?fov='+encodeURIComponent(fov)+(tps ? '&timepoints='+encodeURIComponent(tps) : '');
}

let TP_CATALOG = [];

async function loadTpCatalog(fov) {
  const inv = await api('/animal/inventory?fov=' + encodeURIComponent(fov));
  TP_CATALOG = inv.timepoint_catalog || [];
  renderTpPick();
}

function renderTpPick() {
  const el = document.getElementById('tpPick');
  if (!el) return;
  if (!TP_CATALOG.length) { el.innerHTML = ''; return; }
  const boxes = TP_CATALOG.map(tp => {
    const dis = tp.has_csv ? '' : ' disabled';
    const chk = tp.selected && tp.has_csv ? ' checked' : '';
    const tag = tp.has_csv ? ' ('+tp.spine_count+' spines)' : ' (no CSV)';
    return '<label style="margin-right:12px;white-space:nowrap"><input type="checkbox" class="tp-chk" value="'+tp.name.replace(/"/g,'')+'"'+dis+chk+'> '+tp.name+tag+'</label>';
  }).join('');
  el.innerHTML = '<b>Active timepoints</b> (uncheck to exclude) — '+boxes;
}

function selectedTimepointsQuery() {
  const names = [...document.querySelectorAll('.tp-chk:checked')].map(el => el.value);
  return names.join(',');
}

async function bootstrapHub(reloadAnimal=true){
  const fov = parseInt(document.getElementById('fovId').value || '1', 10);
  try{
    setStatus('Loading FOV '+fov+'…');
    await loadTpCatalog(fov);
    const tps = selectedTimepointsQuery();
    const tpArg = tps ? '&timepoints='+encodeURIComponent(tps) : '';
    if(reloadAnimal){
      STATE = await api('/mtp/load-from-animal?fov='+fov+tpArg, {method:'POST'});
    } else {
      STATE = await api('/mtp/state');
    }
    if(document.getElementById('animalId') && STATE.animal_id) document.getElementById('animalId').value = STATE.animal_id;
    document.getElementById('fovId').value = String(fov);
    if(STATE.timepoints){
      STATE.timepoints.forEach(tp=>{ if(!SEL[tp.name]) SEL[tp.name]=new Set(); });
      renderColumns();
      document.getElementById('workPanel').classList.remove('hidden');
      STATE.timepoints.forEach(tp=>{ if(tp.has_tiff) schedulePreview(tp.name); });
    }
    renderLinkHub();
    const n = (STATE.links||[]).length;
    const v = (STATE.visited_link_ids||[]).length;
    const cat = STATE.catalog_message ? (' · '+STATE.catalog_message) : '';
    setStatus((n ? ('Ready: '+n+' link(s), '+v+' visited. Click a row to analyze.') : 'No links yet — use Auto-link or Create links below.') + cat);
  }catch(e){ setStatus('Load failed: '+e.message); }
}

async function refreshAnimalBanner() {
  const el = document.getElementById('animalBanner');
  try {
    const cfg = await api('/animal/config');
    if (!cfg.configured) {
      el.textContent = 'Edit config/annotator.json — set workspace. App builds workspace/respan/ automatically.';
      return;
    }
    el.innerHTML = '<b>'+cfg.animal_id+'</b> &rarr; <code style="font-size:11px">'+cfg.respan_root+'</code>';
    if (cfg.default_fov) document.getElementById('fovId').value = String(cfg.default_fov);
    if (cfg.animal_id) document.getElementById('animalId').value = cfg.animal_id;
  } catch(e) {
    el.textContent = 'Could not read animal config: '+e.message;
  }
}

(async function init(){
  try{ const s = await api('/mtp/state'); if(s.default_timepoints && s.default_timepoints.length) DEFAULT_TPS=s.default_timepoints;
       if(s.animal_id) document.getElementById('animalId').value=s.animal_id;
       if(s.fov) document.getElementById('fovId').value=s.fov; }catch(e){}
  renderSetup();
  await refreshAnimalBanner();
  await bootstrapHub(true);
})();
</script>
</body>
</html>"""
