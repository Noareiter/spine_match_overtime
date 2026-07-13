# Spine Match Overtime — Codebase Guide

**Dendritic spine matching and lineage review UI for longitudinal microscopy analysis.**

## Quick Overview

This project is a **web-based annotation tool** for reviewing and matching neuronal dendritic spines across multiple timepoints in microscopy imaging data. Scientists load imaging stacks (TIFF) and spine detection CSVs, link dendrites across timepoints (Phase 0), and then review/confirm spine matches and track their lineage over time (Phase 1–2).

- **Entry point**: `scripts/run_annotator.py` — launches FastAPI server + browser UI
- **Configuration**: `config/annotator.json` — animal ID, workspace path, FOVs, timepoint order
- **Frontend**: Browser-based UI (served from FastAPI)
- **Server port**: Default `http://127.0.0.1:8010/mtp/`

---

## Folder Structure

```
spine_match_overtime/
├── config/
│   └── annotator.json           # Animal ID, workspace, timepoints, FOVs
├── scripts/
│   ├── run_annotator.py         # CLI entry point (launches server + browser)
│   ├── spine_annotator_backend/  # Backend package
│   │   ├── app.py               # FastAPI main app (t1/t2 matching engine)
│   │   ├── animal_config.py      # Config loading + path resolution
│   │   ├── models.py             # Pydantic request/response schemas
│   │   ├── session_store.py      # In-memory session state
│   │   ├── baseline_adapter.py   # Spine/stack I/O + matching scoring
│   │   ├── io_service.py         # File path validation + dialogs
│   │   ├── crop_service.py       # TIFF cropping utilities
│   │   ├── multitp_linker.py     # Phase 0: dendrite linking (multi-TP)
│   │   ├── mtp_spine_viewer.py   # Phase 1: spine viewer UI routes + HTML
│   │   ├── animal_api.py         # Animal/FOV discovery API
│   │   ├── dendrite_link_store.py # Persistent dendrite links
│   │   ├── spine_catalog_store.py # Global spine catalog (S_XXXXX)
│   │   ├── spine_lineage_store.py # Lineage decisions (fate, matched pairs)
│   │   ├── manual_spine_store.py  # Manual spine annotations
│   │   ├── timepoint_selection.py # TP selection per FOV
│   │   ├── spine_qc_store.py      # QC metadata
│   │   ├── respan_bootstrap.py    # Workspace layout creation
│   │   ├── project_paths.py       # Path constants (PROJECT_ROOT, RESULTS_DIR)
│   │   ├── __main__.py            # Module launcher
│   │   └── __init__.py
│   ├── create_spine_logic_demo.py # Demo data generator
│   ├── build_spine_catalog.py     # Spine catalog builder
│   ├── open_animal_folders.py     # File explorer opener
│   └── [assume_t1_t2_onPre/, ...] # Utility scripts (not main flow)
├── results/
│   └── demo_spine_logic_lab/      # Demo workspace (created by demo script)
├── docs/
│   └── spine_annotator_output_fields.csv # Output field reference
├── README.md                      # User-facing overview
├── HOW_TO_WORK.txt               # Hebrew workflow guide
├── HOW_TO_WORK_EN.txt            # English workflow guide
└── QUICK_REFERENCE.html          # Quick reference card
```

---

## Data Flow & Architecture

### 1. **Configuration Loading** (`animal_config.py`)

```
annotator.json
    ↓
load_config()  → AnimalConfig(animal_id, workspace, fovs, timepoint_order)
    ↓
resolve_paths()  → bootstrap respan/ structure
    ↓
Path: workspace/respan/ (or workspace itself if named "respan")
```

- **workspace**: Can be absolute (`E:/...`) or relative (`results/...`)
- **timepoint_order**: Defines ordering (pre-droplet, mid-droplet, end-droplet, end-lever, return to droplet)
- **fovs**: List of field-of-view numbers to load
- **active_timepoints**: Subset of TPs to work on (optional, defaults to all)

### 2. **Bootstrap & Directory Layout** (`respan_bootstrap.py`)

On startup, the app creates this structure under `workspace/respan/`:

```
respan/
├── pre-droplet/
│   ├── Tables/
│   │   └── fov1_detected_spines.csv
│   └── fov1/
│       └── fov1.tif
├── mid-droplet/, end-droplet/, end-lever/, ...  # Same structure per TP
└── _annotator/
    ├── fov1/
    │   ├── spine_catalog.csv         # Global spine IDs (S_00001, S_00002, ...)
    │   ├── dendrite_links.json       # Phase 0: dendrite linking decisions
    │   ├── dendrite_link_progress.json
    │   ├── timepoint_selection.json  # Which TPs are active
    │   ├── lineage_decisions.json    # Phase 1: spine matches + fate
    │   ├── spine_registry_wide.csv   # Output: denormalized per-lineage registry
    │   └── spine_review_progress.json
    └── Results final/               # Export output (created on export)
```

- Each FOV gets its own `_annotator/fovN/` directory
- All decision files are JSON or CSV within `_annotator/`
- Bootstrap is **automatic** on FOV load—no manual setup needed

### 3. **Data Loading & Session State** (`session_store.py`, `baseline_adapter.py`)

```
Load → Select TIFF + CSV for t1 & t2 (two timepoints)
        ↓
app._load_session_internal()
        ↓
LoadedSession in memory:
  - t1_stack, t2_stack        (NumPy 3D arrays from TIFF)
  - t1_df, t2_df              (Pandas DataFrames from CSVs)
  - t1_lookup, t2_lookup      (Dict: spine_id → row data)
  - dendrite_links            (List of linked dendrite pairs)
  - algo_matches              (Auto-computed matches by scoring)
  - confirmed_matches         (User-reviewed matches)
  - matched_t1_ids, matched_t2_ids (Sets of IDs with decisions)
  - rejected_pairs, lost_t1_ids, new_t2_ids (Fate decisions)
```

- **LoadedSession** is the single in-memory session; only one active at a time
- Dendrite links → come from `_annotator/fov/dendrite_links.json`
- Spine matches → computed via `baseline_adapter.compute_matches()` (distance + feature scoring)

### 4. **Multi-Timepoint Dendrite Linking** (Phase 0: `multitp_linker.py`)

**Purpose**: Link the same dendrite across many timepoints (e.g., pre, mid, end).

- Separate from t1/t2 matching; doesn't touch session state
- Creates/updates `dendrite_links.json` per FOV
- Returns dendriteN ↔ dendriteM links across consecutive TPs

**API Routes** (in `multitp_linker.py`):
- `GET /mtp/dendrite-links/<fov>` — List saved dendrite links
- `POST /mtp/dendrite-links/<fov>` — Save new dendrite link
- `DELETE /mtp/dendrite-links/<fov>/<link_id>` — Remove link

### 5. **Spine Viewer & Matching** (Phase 1: `mtp_spine_viewer.py`)

**Purpose**: Review and confirm spine matches across two consecutive timepoints (t1 → t2).

**Workflow**:
1. Load two TPs (usually from the same FOV)
2. Auto-match spines by distance + feature scoring
3. User reviews each candidate, confirms or overrides
4. Saves lineage decisions → `lineage_decisions.json`
5. Builds output registry → `spine_registry_wide.csv`

**Key Data Structures** (in `models.py`):
- `ReviewQueueItem` — one candidate match (t1 spine ↔ t2 candidate)
- `ReviewDecision` — user's action (match, no_match, new, lost, artifact, ignore, etc.)
- `CropPreviewRequest/Response` — 3D crop + 2D projection for visual review

**Key Routes**:
- `GET /mtp/viewer/` — Serve spine viewer HTML
- `POST /mtp/select-files/` — Choose TIFF + CSV files
- `POST /mtp/load-session/` — Load selected files into session
- `GET /mtp/review-queue` — List candidates for review (paginated)
- `POST /mtp/review-decision` — Record user's match decision
- `POST /mtp/finalize-matches` — Compute derived fates (LOST, NEW, etc.)
- `POST /mtp/export-results/` — Save all outputs to disk

### 6. **Fate Logic** (Derived in `app.py`)

After user confirms all matches, the app infers **fate** for each spine:

| Fate | Meaning |
|------|---------|
| `matched` | Spine observed at this TP, matches to baseline (anchor) |
| `NEW` | Spine appears for first time at this TP |
| `ARTIFACT` | False positive (algorithm error); no spine exists |
| `IGNORE` | Spine exists but out-of-frame (OOF) at this TP |
| `LOST` | Spine was present in baseline, missing at this TP without other fate |
| `CENSORED` | Data quality issue; not counted as lost |

---

## Core Modules Deep Dive

### **animal_config.py** — Configuration & Paths

**Key Types**:
- `AnimalConfig` — dataclass: animal_id, workspace, fovs, timepoint_order, active_timepoints
- Functions:
  - `load_config(config_path?)` — Load from JSON, override from env vars
  - `resolve_workspace_path(cfg)` — Expand relative paths under PROJECT_ROOT
  - `bootstrap(cfg)` → `BootstrapReport` — Create respan layout if needed
  - `resolve_paths(cfg)` → dict — Return all path info for UI/logging

**Environment Variables** (override config file):
- `SPINE_CONFIG` — Path to annotator.json
- `SPINE_ANIMAL_ID` — Animal ID
- `SPINE_WORKSPACE` or `SPINE_DATA_ROOT` — Workspace path
- `SPINE_DEFAULT_FOV` — Default FOV number

### **session_store.py** — Session Management

**Key Types**:
- `LoadedSession` — In-memory session:
  - `t1_tiff_path`, `t2_tiff_path`, `t1_csv_path`, `t2_csv_path` — File paths
  - `t1_stack`, `t2_stack` — NumPy 3D arrays (Z×Y×X)
  - `t1_df`, `t2_df` — Pandas DataFrames with spine detections
  - `t1_lookup`, `t2_lookup` — Dicts for fast spine lookup
  - `dendrite_links` — List of linked dendrite pairs
  - `algo_matches` — Auto-computed matches (distance + scoring)
  - `confirmed_matches` — User-confirmed matches
  - `matched_t1_ids`, `matched_t2_ids` — Sets of matched spine IDs
  - `rejected_pairs`, `lost_t1_ids`, `new_t2_ids` — Fate decisions
  - `max_match_z_gap` — Z-distance threshold for matches

**Functions**:
- `set_active_session(session)` — Store session globally
- `require_active_session()` → LoadedSession — Get current session
- `reset_active_session()` — Clear session

### **baseline_adapter.py** — Spine Data & Matching

**Key Functions**:
- `load_stack(tiff_path)` → NumPy array (Z×Y×X, int16 or similar)
- `load_spines(csv_path)` → Pandas DataFrame (columns: spine_id, x, y, z, dendrite_id, ...)
- `to_lookup(df)` → Dict[spine_id → row_data] for fast access
- `dendrite_groups(df)` → List[tuple(dendrite_id, [spine_ids])]
- `compute_matches(t1_df, t2_df, distance_threshold=?, ...)` → List[Match]
  - Scoring: hybrid distance (XY + Z gap) + optional feature-based weighting
  - Returns candidates ranked by score

### **crop_service.py** — Image Cropping

**Purpose**: Extract 3D crops around spines for visual review.

**Key Functions**:
- `make_local_crop(stack, x, y, z, width=96, height=96, depth=13)` → 3D array
- `make_2d_projection(crop, projection_type='mid'|'mip'|'slice', slice_z=None)` → 2D array for browser display

### **multitp_linker.py** — Phase 0: Dendrite Linking

**Purpose**: Link dendrites across multiple consecutive timepoints (e.g., pre → mid → end).

**Routes**:
- `GET /mtp/dendrite-links/<fov>` — List saved links
- `POST /mtp/dendrite-links/<fov>` — Create/update link
- `DELETE /mtp/dendrite-links/<fov>/<link_id>` — Remove link

**Data Stored**: `_annotator/fovN/dendrite_links.json`
```json
{
  "links": [
    {
      "link_id": "d_pre_1__d_mid_1",
      "pre-droplet": ["dendrite_1"],
      "mid-droplet": ["dendrite_1"],
      ...
    }
  ]
}
```

### **mtp_spine_viewer.py** — Phase 1: Spine Viewer UI

**HTML Page**: Served at `/mtp/viewer/?fov=1`

**Routes**:
- `GET /mtp/viewer/` — Serve HTML
- `POST /mtp/select-files` — Choose files (TIFF + CSV)
- `POST /mtp/load-session` — Load selected files
- `GET /mtp/review-queue` — Get next candidates (paginated, offset+limit)
- `POST /mtp/review-decision` — Record match/no-match/new/lost/artifact decision
- `POST /mtp/finalize-matches` — Compute fates, save to JSON/CSV
- `POST /mtp/export-results` — Export to external directory
- `GET /mtp/crop-preview` — Get 2D crop image for spine
- `GET /mtp/nearest-spines` — Find nearby spines at a coordinate
- Various undo, cleanup, and manual-click routes

### **animal_api.py** — Animal & FOV Discovery

**Routes**:
- `GET /mtp/animals` — List configured animals
- `GET /mtp/animal/{animal_id}` → fovs, workspaces, detected TPs
- `GET /mtp/animal/{animal_id}/fov/{fov}` → TP list, session state

### **Data Store Modules**

Each handles persistence of one type of decision/metadata:

- **dendrite_link_store.py** — `dendrite_links.json` (multi-TP dendrite associations)
- **spine_catalog_store.py** — `spine_catalog.csv` (global spine ID assignment)
- **spine_lineage_store.py** — `lineage_decisions.json` (per-spine per-TP decisions)
- **manual_spine_store.py** — Manual spine annotations (clicks)
- **timepoint_selection.py** — Which TPs are active per FOV
- **spine_qc_store.py** — QC metadata
- **ignored_spine_store.py**, **oof_segment_store.py** — Out-of-frame & ignore flags

---

## Running the Application

### **For Demo**

```bash
# Create demo data first (one-time)
python scripts/create_spine_logic_demo.py

# Launch demo
./start_demo_annotator.bat
# or: python scripts/run_annotator.py --config config/annotator_demo.json
```

Browser opens to `http://127.0.0.1:8010/mtp/` (Phase 0: dendrite linker) and `http://127.0.0.1:8010/mtp/viewer/` (Phase 1: spine viewer).

### **For Real Animal Data**

1. **Edit** `config/annotator.json`:
   ```json
   {
     "animal_id": "GP04",
     "workspace": "E:/experiments/GP04_2024",
     "fovs": [1, 2, 3],
     "timepoint_order": ["pre-droplet", "mid-droplet", "end-droplet", "end-lever"],
     "active_timepoints": ["pre-droplet", "end-droplet"]
   }
   ```

2. **Run**:
   ```bash
   ./start_annotator.bat
   # or: python scripts/run_annotator.py
   ```

3. **App will**:
   - Verify workspace exists
   - Bootstrap respan layout (create folders if needed)
   - Scan for timepoint folders + detected spine CSVs
   - Start server at `http://127.0.0.1:8010/mtp/`

---

## Output Files

After completing spine review + finalization:

### **Automatic (in `_annotator/fovN/`)**:
- `spine_catalog.csv` — Global spine IDs (S_XXXXX) for every detection
- `dendrite_links.json` — Multi-TP dendrite associations (Phase 0)
- `lineage_decisions.json` — Per-spine per-TP decision details (fate, coordinates, source)
- `spine_registry_wide.csv` — **Main output**: one row per lineage, columns per TP + derived fields

### **User-Exported (in results/)**:
- Called via `/mtp/export-results` → saved to `_annotator/Results final/`

### **spine_registry_wide.csv Schema** (Key Columns):
- `lineage_id` — Unique lineage identifier
- `pre_spine_id`, `mid_spine_id`, ... — Spine IDs at each TP (or empty)
- `anchor_timepoint` — First TP where spine was confirmed
- `first_seen_tp`, `last_seen_tp` — Temporal extent of observation
- `status_<tp>` — matched | new | artifact | ignore | censored
- `fate_<tp>` — The semantic fate (LOST, NEW, etc.)
- `event_<tp>` — Derived events (appeared_at_X, lost_at_Y)
- `lifecycle` — Derived summary (stable, transient, persistent_engram, right_censored, ...)

See `docs/spine_annotator_output_fields.csv` for full field reference.

---

## Key Design Decisions

1. **Single FOV + Pair of TPs at a Time**: The session state handles exactly two timepoints. Multi-TP analysis (dendrite linking) is separate (Phase 0).

2. **Global Spine Catalog**: Every detection gets a unique ID (`S_XXXXX`) to avoid cross-TP collisions. CSV numbers are kept as `local_spine_id`.

3. **Automatic Bootstrap**: Respan layout is created automatically on first load; no manual directory setup needed.

4. **JSON + CSV Outputs**: Decisions are saved as JSON for full fidelity; registry is CSV for Excel/analysis tools.

5. **In-Memory Matching**: Scoring and matching is done in memory; decisions are persisted to disk.

6. **Stateless Routes**: Most API routes fetch fresh data from disk or in-memory session to ensure consistency.

---

## External Dependencies

### **Python Packages**:
- `fastapi`, `uvicorn` — Web framework
- `pandas`, `numpy` — Data processing
- `pydantic` — Request/response validation
- `tifffile` — TIFF image loading (demo only)

### **External Scripts**:
- `D:\learning_project_spines\scripts\hybrid_tracking\track_hybrid.py`
- `D:\learning_project_spines\scripts\step2-spine tracking\spine_matching_tool\utils.py`
- Set via `SPINE_TRACKING_SCRIPTS` env var if path changes

---

## Development Notes

- **Code Style**: No special linting configured; follows standard Python conventions
- **Testing**: No test suite present; verify via manual demo
- **Logging**: Uses standard Python logging; also `matching_activity.log` in session state
- **Browser Compatibility**: Tested on modern Chrome/Edge; uses WebGL for 3D preview
- **Performance**: Session state is fully in-memory; suitable for one FOV + 2 TPs at a time

---

## File Locations (Summary)

| What | Where |
|------|-------|
| User config | `config/annotator.json` |
| Main server | `scripts/spine_annotator_backend/app.py` |
| CLI launcher | `scripts/run_annotator.py` |
| Demo data | `results/demo_spine_logic_lab/` (auto-created) |
| Session state | `workspace/respan/_annotator/fovN/*.json` |
| Output files | `workspace/respan/_annotator/fovN/*.csv` |
| Quick help | `QUICK_REFERENCE.html` (opened on launch) |
