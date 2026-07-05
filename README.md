# Spine Match Overtime

Longitudinal spine matching and lineage review UI.

## Layout

| Folder | Purpose |
|--------|---------|
| `config/` | `annotator.json` — animal_id, workspace, timepoints |
| `scripts/` | Python app (`run_annotator.py`, `spine_annotator_backend/`) |
| `results/` | Demo data + app session state |
| `docs/` | Output field reference (CSV) |

## Step 0 — Global spine catalog (automatic)

On FOV load, the app builds or loads:

`respan/_annotator/fovN/spine_catalog.csv`

Every detection at every timepoint gets a unique ID (`S_00001`, `S_00002`, …) with the original CSV number kept as `local_spine_id`.

Built **automatically** when you load a FOV (dendrite linker or spine tracker).

UI labels show `local (S_*)`. Lineage saves use `lineage_key` = global ID (no cross-timepoint collisions).

See **HOW_TO_WORK.txt** (Hebrew) or **HOW_TO_WORK_EN.txt** (English) for workflow, fate logic, and output semantics.

## Run

- **Real animal:** `start_annotator.bat` (edit `config/annotator.json` first)
- **Demo:** `start_demo_annotator.bat`

Browser: `http://127.0.0.1:8010/mtp/` (dendrite links) · `/mtp/viewer/` (spine tracker)
