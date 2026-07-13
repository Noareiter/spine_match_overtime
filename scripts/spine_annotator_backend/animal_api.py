"""HTTP API: animal workspace config + respan bootstrap."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import animal_config, animal_layout, models

router = APIRouter(prefix="/animal", tags=["animal-layout"])


class AnimalConfigResponse(BaseModel):
    configured: bool
    animal_id: str = ""
    workspace: str = ""
    respan_root: str = ""
    results_root: str = ""
    results_final_dir: str = ""
    default_fov: int = 1
    folder_open_depth: int = 2
    config_path: str = ""
    timepoint_order: List[str] = Field(default_factory=list)
    active_timepoints: List[str] = Field(default_factory=list)
    fovs: List[int] = Field(default_factory=list)
    bootstrap_created: int = 0
    bootstrap_summary: str = ""
    error: str = ""


class BootstrapResponse(BaseModel):
    ok: bool
    workspace: str
    respan_root: str
    created: List[str]
    existing_count: int
    warnings: List[str]
    summary: str


class FovInventoryResponse(BaseModel):
    animal_id: str
    fov: int
    respan_root: str
    timepoints: List[dict]
    workflow_pairs: List[dict] = Field(default_factory=list)
    selected_timepoints: List[str] = Field(default_factory=list)
    timepoint_catalog: List[dict] = Field(default_factory=list)


class LoadPairRequest(BaseModel):
    fov: int = 1
    t1_timepoint: str
    t2_timepoint: str


class LoadPairResponse(BaseModel):
    selected_files: models.SelectFilesResponse
    session: models.SessionStats
    t1_timepoint: str
    t2_timepoint: str
    fov: int
    dendrite_links_loaded: int = 0
    dendrite_links_path: str = ""


class TimepointFilesResponse(BaseModel):
    name: str
    folder: str
    csv_path: str | None = None
    tiff_path: str | None = None
    spine_count: int = 0
    dendrite_ids: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)


@router.get("/config", response_model=AnimalConfigResponse)
def get_animal_config() -> AnimalConfigResponse:
    cfg = animal_config.load_config()
    paths = animal_config.resolve_paths(cfg)
    fovs: List[int] = []
    respan = paths.get("respan_root", "")
    if respan:
        try:
            fovs = animal_layout.discover_fovs(animal_config.require_respan(cfg))
        except Exception:
            fovs = cfg.fovs
    return AnimalConfigResponse(
        configured=cfg.is_configured() and bool(respan) and not paths.get("error"),
        animal_id=cfg.animal_id,
        workspace=paths.get("workspace", ""),
        respan_root=respan,
        results_root=paths.get("results_root", ""),
        results_final_dir=paths.get("results_final_dir", ""),
        default_fov=cfg.default_fov,
        folder_open_depth=cfg.folder_open_depth,
        config_path=paths.get("config_path", ""),
        timepoint_order=list(cfg.timepoint_order),
        active_timepoints=list(cfg.active_timepoints),
        fovs=fovs or cfg.fovs,
        bootstrap_created=int(paths.get("bootstrap_created", 0) or 0),
        bootstrap_summary=str(paths.get("bootstrap_summary", "")),
        error=str(paths.get("error", "")),
    )


@router.post("/bootstrap", response_model=BootstrapResponse)
def bootstrap_respan() -> BootstrapResponse:
    try:
        cfg = animal_config.load_config()
        report = animal_config.bootstrap(cfg)
        return BootstrapResponse(
            ok=True,
            workspace=str(report.workspace),
            respan_root=str(report.respan),
            created=report.created,
            existing_count=len(report.existing),
            warnings=report.warnings,
            summary=report.summary(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/inventory", response_model=FovInventoryResponse)
def get_fov_inventory(fov: int = 1, timepoints: str = "") -> FovInventoryResponse:
    try:
        from . import timepoint_selection

        cfg = animal_config.load_config()
        respan = animal_config.require_respan(cfg)
        requested = timepoint_selection.parse_timepoint_query(timepoints)
        saved = timepoint_selection.load(respan, fov)
        selected = animal_config.resolve_active_timepoints(
            cfg,
            animal_layout.discover_available_timepoints(respan, fov),
            requested=requested or None,
            saved=saved or None,
        )
        inv = animal_layout.build_fov_inventory(
            respan, fov, animal_id=cfg.animal_id, selected_timepoints=selected
        )
        catalog = animal_layout.list_timepoint_catalog(respan, fov, saved_selection=selected)
        return FovInventoryResponse(
            animal_id=inv.animal_id,
            fov=inv.fov,
            respan_root=inv.respan_root,
            timepoints=[TimepointFilesResponse(**tp.__dict__).model_dump() for tp in inv.timepoints],
            workflow_pairs=inv.workflow_pairs,
            selected_timepoints=selected,
            timepoint_catalog=catalog,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ResetFovResponse(BaseModel):
    ok: bool
    fov: int
    removed: List[str] = Field(default_factory=list)
    kept: List[str] = Field(default_factory=list)
    backup_dir: str = ""
    message: str = ""


class RebuildRegistryResponse(BaseModel):
    ok: bool
    fov: int
    registry_path: str
    lineage_count: int = 0
    timepoint_names: List[str] = Field(default_factory=list)
    message: str = ""


@router.post("/rebuild-registry", response_model=RebuildRegistryResponse)
def rebuild_registry(fov: int = 1) -> RebuildRegistryResponse:
    """Rebuild spine_registry_wide.csv from lineage_decisions.json."""
    try:
        from . import spine_lineage_store

        cfg = animal_config.load_config()
        respan = animal_config.require_respan(cfg)
        data = spine_lineage_store.load_decisions(respan, fov)
        path = spine_lineage_store.rebuild_registry_wide(
            respan, fov, animal_id=str(data.get("animal_id") or cfg.animal_id)
        )
        tps = spine_lineage_store._ordered_union_timepoints(list(data.get("lineages") or []))
        n = len(data.get("lineages") or [])
        return RebuildRegistryResponse(
            ok=True,
            fov=fov,
            registry_path=str(path),
            lineage_count=n,
            timepoint_names=tps,
            message=f"Rebuilt registry with {n} lineage(s), {len(tps)} timepoint column(s).",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reset-fov", response_model=ResetFovResponse)
def reset_fov_workspace(
    fov: int = 1,
    backup: bool = True,
    keep_dendrite_links: bool = False,
) -> ResetFovResponse:
    """Clear annotator metadata for one FOV. Set keep_dendrite_links=true to preserve dendrite links."""
    try:
        from . import fov_reset

        cfg = animal_config.load_config()
        respan = animal_config.require_respan(cfg)
        result = fov_reset.reset_fov_annotator(
            respan,
            int(fov),
            backup=bool(backup),
            keep_dendrite_links=bool(keep_dendrite_links),
        )
        return ResetFovResponse(ok=True, **result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
