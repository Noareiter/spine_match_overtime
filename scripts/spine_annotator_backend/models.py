from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Timepoint = Literal["t1", "t2"]


class SelectFilesRequest(BaseModel):
    t1_tiff_path: Optional[str] = None
    t2_tiff_path: Optional[str] = None
    t1_csv_path: Optional[str] = None
    t2_csv_path: Optional[str] = None
    use_dialog: bool = True


class SelectFilesResponse(BaseModel):
    t1_tiff_path: str
    t2_tiff_path: str
    t1_csv_path: str
    t2_csv_path: str


class LoadSessionRequest(BaseModel):
    t1_tiff_path: str
    t2_tiff_path: str
    t1_csv_path: str
    t2_csv_path: str


class SessionStats(BaseModel):
    t1_spine_count: int
    t2_spine_count: int
    t1_dendrite_ids: List[str]
    t2_dendrite_ids: List[str]
    t1_z_range: List[int]
    t2_z_range: List[int]


class SelectAndLoadResponse(BaseModel):
    selected_files: SelectFilesResponse
    session: SessionStats


class SpineRecord(BaseModel):
    spine_id: str
    dendrite_id: Optional[str] = None
    x: float
    y: float
    z: float
    features: Dict[str, Any] = Field(default_factory=dict)


class DendriteGroup(BaseModel):
    dendrite_id: str
    spine_ids: List[str]


class CropTarget(BaseModel):
    spine_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    timepoint: Timepoint


class LocalCropRequest(BaseModel):
    targets: List[CropTarget]
    width: int = 96
    height: int = 96
    depth: int = 13


class CropMeta(BaseModel):
    source_bounds: Dict[str, int]
    clamped: Dict[str, bool]
    center_index_source: Dict[str, int]
    center_index_local: Dict[str, int]


class CropResult(BaseModel):
    timepoint: Timepoint
    spine_id: Optional[str] = None
    shape: List[int]
    crop: List[List[List[float]]]
    meta: CropMeta


class LocalCropResponse(BaseModel):
    count: int
    crops: List[CropResult]


class CropPreviewRequest(BaseModel):
    timepoint: Timepoint
    spine_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    width: int = 96
    height: int = 96
    depth: int = 13
    projection: Literal["mid", "mip", "slice"] = "mid"
    slice_index: Optional[int] = None
    # crop_local: slice_index is within the small centered 3D crop; stack_global: slice_index is Z in the full TIFF
    slice_z_mode: Literal["crop_local", "stack_global"] = "crop_local"


class CropPreviewResponse(BaseModel):
    timepoint: Timepoint
    spine_id: Optional[str] = None
    shape: List[int]
    projection: Literal["mid", "mip", "slice"]
    slice_index: Optional[int] = None
    image_2d: List[List[float]]
    intensity_min: float
    intensity_max: float
    meta: CropMeta


class ReviewSuggestion(BaseModel):
    t2_spine_id: str
    suggested_t1_spine_id: Optional[str] = None
    distance_xy: Optional[float] = None
    distance_z: Optional[float] = None


class ReviewDecisionRequest(BaseModel):
    t2_spine_id: str
    action: Literal["match", "manual_match", "no_match", "remove_t1", "remove_t2", "new", "lost", "skip", "not_in_t1", "ignore_t1", "ignore_t2"]
    t1_spine_id: Optional[str] = None
    notes: str = ""


class ReviewDecision(BaseModel):
    t2_spine_id: str
    action: Literal["match", "manual_match", "no_match", "remove_t1", "remove_t2", "new", "lost", "skip", "not_in_t1", "ignore_t1", "ignore_t2"]
    t1_spine_id: Optional[str] = None
    notes: str = ""


class ReviewDecisionsResponse(BaseModel):
    count: int
    decisions: List[ReviewDecision]


class ReviewUndoResponse(BaseModel):
    ok: bool
    message: str
    removed_decision: Optional[ReviewDecision] = None


class DendriteIdsResponse(BaseModel):
    t1_dendrite_ids: List[str]
    t2_dendrite_ids: List[str]


class DendriteLinkRequest(BaseModel):
    t1_dendrite_ids: List[str]
    t2_dendrite_ids: List[str]
    notes: str = ""


class DendriteLink(BaseModel):
    link_id: str
    t1_dendrite_ids: List[str]
    t2_dendrite_ids: List[str]
    notes: str = ""


class DendriteLinksResponse(BaseModel):
    count: int
    links: List[DendriteLink]


class SessionPersistenceResponse(BaseModel):
    ok: bool
    message: str
    has_saved_session: bool = False
    saved_path: Optional[str] = None
    selected_files: Optional[SelectFilesResponse] = None
    session: Optional[SessionStats] = None


class ViewerState(BaseModel):
    modal_slice_t1: int = 10
    modal_slice_t2: int = 10


class MatchSettings(BaseModel):
    max_match_z_gap: float = 7.0


class StackBoundsResponse(BaseModel):
    """Full-stack Z extent for large-viewer slice sliders (independent per timepoint)."""

    t1_shape_z: int
    t2_shape_z: int
    t1_slice_max: int
    t2_slice_max: int


class ReviewQueueItem(BaseModel):
    t2_spine_id: str
    t2_dendrite_id: Optional[str] = None
    suggested_t1_spine_id: Optional[str] = None
    suggested_t1_dendrite_id: Optional[str] = None
    distance_xy: Optional[float] = None
    distance_z: Optional[float] = None
    score_feature_weighted: Optional[float] = None
    score_toolb_model: Optional[float] = None
    stability_score: Optional[float] = None
    final_score: Optional[float] = None
    margin: Optional[float] = None
    local_shift_xyz: Optional[Dict[str, float]] = None
    registration_applied: bool = False
    nearby_t1_candidates: List[Dict[str, float | str]] = Field(default_factory=list)
    nearby_t2_candidates: List[Dict[str, float | str]] = Field(default_factory=list)


class ReviewQueueResponse(BaseModel):
    offset: int
    limit: int
    total_candidates: int
    items: List[ReviewQueueItem]


class Top5Candidate(BaseModel):
    t2_spine_id: str
    t2_dendrite_id: Optional[str] = None
    distance_xy: Optional[float] = None
    distance_z: Optional[float] = None
    final_score: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class Top5NextResponse(BaseModel):
    has_item: bool
    t1_spine_id: Optional[str] = None
    t1_dendrite_id: Optional[str] = None
    t1_xyz: Optional[Dict[str, float]] = None
    candidates: List[Top5Candidate] = Field(default_factory=list)


class AlgoMatchRemapRequest(BaseModel):
    t2_spine_id: str
    t1_spine_id: str


class AlgoMatchUpdateResponse(BaseModel):
    ok: bool
    message: str
    t2_spine_id: str
    t1_spine_id: Optional[str] = None


class UndoLastChoiceResponse(BaseModel):
    ok: bool
    message: str
    action_label: Optional[str] = None


class MatchFinalizeResponse(BaseModel):
    matched_count: int
    no_match_count: int
    inferred_new_count: int
    pending_new_validation_count: int
    inferred_lost_count: int
    removed_t1_count: int
    removed_t2_count: int
    ignored_t1_count: int
    ignored_t2_count: int
    inferred_new_t2_ids: List[str]
    inferred_lost_t1_ids: List[str]


class ManualT2ClickAddRequest(BaseModel):
    x: float
    y: float
    z: float
    notes: str = ""


class ManualT2ClickItem(BaseModel):
    manual_id: str
    timepoint: Literal["t2"] = "t2"
    x: float
    y: float
    z: float
    notes: str = ""
    created_at: str


class ManualT2ClickAddResponse(BaseModel):
    ok: bool
    message: str
    item: ManualT2ClickItem


class ManualT2ClickListResponse(BaseModel):
    count: int
    items: List[ManualT2ClickItem]


class ManualT1ClickAddRequest(BaseModel):
    x: float
    y: float
    z: float
    notes: str = ""


class ManualT1ClickItem(BaseModel):
    manual_id: str
    timepoint: Literal["t1"] = "t1"
    x: float
    y: float
    z: float
    notes: str = ""
    created_at: str


class ManualT1ClickAddResponse(BaseModel):
    ok: bool
    message: str
    item: ManualT1ClickItem


class ManualT1ClickListResponse(BaseModel):
    count: int
    items: List[ManualT1ClickItem]


class ManualClickMatchRequest(BaseModel):
    t2_spine_id: Optional[str] = None
    manual_t2_id: Optional[str] = None
    t1_spine_id: Optional[str] = None
    manual_t1_id: Optional[str] = None
    notes: str = ""


class ManualClickMatchItem(BaseModel):
    match_id: str
    t2_kind: Literal["existing", "manual"]
    t2_id: str
    t2_x: Optional[float] = None
    t2_y: Optional[float] = None
    t2_z: Optional[float] = None
    t1_kind: Literal["existing", "manual"]
    t1_id: str
    t1_x: Optional[float] = None
    t1_y: Optional[float] = None
    t1_z: Optional[float] = None
    notes: str = ""
    created_at: str


class ManualClickMatchResponse(BaseModel):
    ok: bool
    message: str
    item: ManualClickMatchItem


class NewValidationItem(BaseModel):
    t2_spine_id: str
    t2_dendrite_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class NewValidationNextResponse(BaseModel):
    has_item: bool
    pending_count: int
    item: Optional[NewValidationItem] = None
    t1_slice_max: int = 0
    t2_slice_max: int = 0


class ReviewCountersResponse(BaseModel):
    manual_matched: int
    algo_matched: int
    to_review: int
    t1_lost: int
    t2_new: int
    t1_slice_max: int = 20
    t2_slice_max: int = 20
    max_match_z_gap: float = 7.0


class CleanupQueueItem(BaseModel):
    timepoint: Timepoint
    spine_id: str
    x: float
    y: float
    z: float
    nearest_other_timepoint_spine_id: Optional[str] = None
    is_unlinked_dendrite: bool = False
    dendrite_id: Optional[str] = None


class CleanupQueueResponse(BaseModel):
    total_unclassified: int
    unreviewed_unlinked_count: int = 0
    queueable_count: int = 0
    orphan_ids: List[str] = Field(default_factory=list)
    items: List[CleanupQueueItem]


class CleanupClassifyRequest(BaseModel):
    timepoint: Timepoint
    spine_id: str
    classification: Literal["new", "lost", "artifact", "ignore", "reviewed"]


class CleanupClassifyResponse(BaseModel):
    ok: bool
    message: str


class ExportResultsResponse(BaseModel):
    ok: bool
    output_dir: str
    files: List[str]


class ClientLogRequest(BaseModel):
    message: str = ""


class ClientLogResponse(BaseModel):
    ok: bool


class ExportResultsRequest(BaseModel):
    use_dialog: bool = True
    output_parent_dir: Optional[str] = None
    output_name: Optional[str] = None


class NearestSpinesRequest(BaseModel):
    timepoint: Timepoint
    x: float
    y: float
    z: float
    limit: int = 5


class NearestSpineItem(BaseModel):
    spine_id: str
    distance_xy: float
    distance_z: float


class NearestSpinesResponse(BaseModel):
    timepoint: Timepoint
    query: Dict[str, float]
    items: List[NearestSpineItem]


class DendritePreviewRequest(BaseModel):
    timepoint: Timepoint
    dendrite_ids: List[str]
    width: int = 160
    height: int = 160
    depth: int = 13
    projection: Literal["mid", "mip"] = "mip"


class DendritePreviewItem(BaseModel):
    dendrite_id: str
    spine_count: int
    center_xyz: Dict[str, float]
    shape: List[int]
    projection: Literal["mid", "mip"]
    image_2d: List[List[float]]
    intensity_min: float
    intensity_max: float
    meta: CropMeta


class DendritePreviewResponse(BaseModel):
    timepoint: Timepoint
    selected_count: int
    previews: List[DendritePreviewItem]


class FovPreviewRequest(BaseModel):
    timepoint: Timepoint
    dendrite_ids: List[str] = Field(default_factory=list)
    projection: Literal["mid", "mip"] = "mip"


class FovPreviewPoint(BaseModel):
    spine_id: str
    dendrite_id: Optional[str] = None
    x: float
    y: float
    z: float


class FovPreviewResponse(BaseModel):
    timepoint: Timepoint
    shape_zyx: List[int]
    projection: Literal["mid", "mip"]
    image_2d: List[List[float]]
    intensity_min: float
    intensity_max: float
    highlighted_points: List[FovPreviewPoint]

