# Outputs review — October 2026

Decisions taken one by one with the user (2026-10-20). **Implemented** on branch `cleanup/single-version`.
Scope: what the current code writes for **new** data. Legacy files on disk are out of scope.

## Decisions

| # | Output | Decision |
|---|--------|----------|
| 2 | `spine_registry_wide.csv`: `fate_<tp>`, `artifact_mode_<tp>` | Remove both. `status_<tp>` (matched / new / lost / absent) is the single per-timepoint column. |
| 3 | Registry x/y/z at absent / lost timepoints | Leave empty. Coordinates only where a spine was observed (matched / new). |
| 5 | `spine_disposition.csv` | Keep animal_id, fov, global_spine_id, timepoint, local_spine_id, disposition, lineage_key, conflict, reviewed. Drop event (always empty), status, fate. Remove unreachable values lineage_lost / lineage_ignore / lineage_artifact from code and docs. |
| 6 | Excluded detections | One `excluded_spines.csv` (replaces ignored_spines.csv + artifact_spines.csv) with a `reason` column: artifact / ignore / oof / dendrite_ignore. |
| 7 | Always-empty fields | Remove `censored_from_tp` / `right_censored` from lineage_decisions.json entries, and `t1_timepoint` / `t2_timepoint` / `pair_label` + workflow_pair rows from fov_inventory.csv (timepoint rows only). |
| 8 | `postprocess_registry.py` / `spine_registry_derived.csv` (events, lifecycle) | Remove from the project, with `_derive_events_and_lifecycle` and the derived-fields docs. Classification belongs to the analysis, from the registry / long file. |
| 4 | Morphology | New `spine_observations_long.csv` (per FOV + merged in Results final): one row per lineage × timepoint with status and every RESPAN column (38 measures: volumes, lengths, head/neck, intensities, distance to soma), empty at absent / lost. The wide registry stays ids + status + coords. |

## Dropped

- 1: registry not rebuilt on load — affects only data saved under older rules; new data rebuilds on every save.
