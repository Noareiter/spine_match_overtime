# Rules review — October 2026

Decisions taken one by one with the user (2026-10-20). **Implemented** on branch `cleanup/single-version`: dcf7590 (1, 3), f9dc783 (6, 7), 8a3a0ed (19, 20, 9, 10, 11, 14, 22, 23), 6fb3126 (12, 13, 24), ed44c3e (29, 29b, 26-28, 30-32, 35), 15f72d1 (4). Output vocabulary (censored, lineage_ignore, right_censored, artifact_mode, events/lifecycle) is left for the outputs review.

## Decisions

| # | Rule | Decision |
|---|------|----------|
| 1 | Which `respan/` subfolders are timepoints | Only folders named in `config.timepoint_order`. Ignore everything else (also leave it out of `fov_inventory.csv`). |
| 3 | First timepoint / per-FOV timepoint choice | Timepoints come only from config (`timepoint_order`, optional `active_timepoints`) and are identical for every FOV. Remove the per-FOV "Active timepoints" checkboxes (viewer + linker) and stop using `timepoint_selection.json`. If **any** configured FOV lacks the CSV or TIFF for any selected timepoint, the **whole animal** is blocked at startup with an error listing every missing file. Spine-level: no match / OOF at TP1 is the user's call (add manual spine, or leave empty → NEW at TP2) — already works. |
| 12 | What an OOF shape removes | Every spine inside the shape (any dendrite), **only at the timepoint it was drawn on**. No projection to other timepoints: remove the "Select timepoints to apply" boxes and the registration-adjusted copy. To mark another timepoint, draw there. (Makes bugs 13/24 mostly moot for OOF.) |
| 16 | XY distance cap on algorithm suggestions | None (keep). GP04 confirmed matches: XY median 31 px, p99 98, 73/1230 > 80 px (raw coords, inter-TP drift). Z: p99 7.3, max 11.1 — the z gate (7) only limits suggestions, manual picks may exceed it. One confirmed pair at 666 px looks suspicious — flag for the user. |
| 17 | Scoring weights (0.45 XY / 0.20 Z / 0.35 features; final 0.85 feature + 0.15 stability; w_app=0) | Keep as is. |
| 19 | Spine on a dendrite with no Phase-0 link (when links exist) | **No cross-dendrite suggestions at all** (clarified 2026-10-20, replaces "goes to cross queue"). Remove the cross queue / `allow_cross_dendrite`. Spines on unlinked dendrites are not suggested; they reach the orphan / unreviewed queues and are matched manually. |
| 20a | FOV with no Phase-0 links at all | Keep as today: the algorithm searches the whole FOV with no dendrite restriction. |
| 20 | Links exist but give zero candidates | Do not search outside the links — no suggestion (remove the unrestricted fallback that labels results non-cross). |
| 21 | Same mid spine suggested to several pres | Keep as today (confirmed 2026-10-20): one suggestion per timepoint. Availability = saved claims only. A spine not chosen (or removed) returns to the pool, stays selectable and suggestable for other lineages; if a lineage's best candidate is claimed, the next-best is offered even at a lower score. |
| 29 | How a lineage may end | Only two endings: MATCHED through the last timepoint, or LOST (first missing TP) then ABSENT. No lineage-level CENSORED / right-censored, no lineage-level IGNORE. ARTIFACT / IGNORE are point events on one detection at one timepoint, set only in Timepoint mode (QC tags). Remove lineage-level censoring code, IGNORE propagation and the `lineage_ignore` fallback (covers bugs 26, 28). |
| 29c | A/I tag in Timepoint mode on a spine a saved lineage claims | Keep as today: the **whole lineage is deleted**, all its spines return to the pool, rebuild it from scratch. |
| 29b | Tab→I (ignore every TP from Lineage mode) | Remove. IGNORE only in Timepoint mode, one detection at one timepoint (makes bug 34 moot). Supersedes the 2026-08-20 decision to keep it. |
| 6 | Phase 0 auto-link by identical dendrite id | Remove the auto-link button/endpoint. RESPAN dendrite numbering is not consistent across timepoints; linking is manual only. |

## Bugs to fix (no decision needed)

- 4: Catalog rebuild reassigns all `S_` ids when a CSV changes — saved decisions then point at wrong spines.
- 7: Dendrite link ids can collide after deletions (`link_counter = max(len(links), counter)`).
- 9: `set_spine` guard (`_is_spine_ignored`) does not check QC tags (A/I).
- 10: Phase-0 anchor list does not drop QC-tagged pre spines.
- 11: `_build_queues()` runs before OOF / ignored / ignored-dendrite state is loaded.
- 13: `coords_in_oof` compares the pre-TP dendrite id against other-TP dendrite ids.
- 14: `/focus` nearest-snap checks only OOF (not claimed / QC / ignored dendrite / z).
- 22: Registry-sourced suggestion is not checked against claimed / ignored.
- 23: Local-registration nearest search has no `MAX_MATCH_Z_GAP` gate and no distance cap.
- 24: OOF shape shift sign is opposite to the matcher's.
- 26: Nothing validated before the first match (LOST / IGNORE / censored pass) — fix per decision 29: only ABSENT before the first match.
- 27: Sticky NEW drops an earlier-TP match silently.
- 28: Live IGNORE propagation + `lineage_ignore` disposition fallback — remove per decision 29.
- 30: `fate_locked` only covers last+1; `commit-timepoint-fate` ignores it.
- 31: Conflict resolution skips `validate_lineage_shape`.
- 32: False-positive deletes the lineage (decision not persisted) but marks reviewed.
- 34: (moot — Tab→I removed, decision 29b)
- 35: Tab in Lineage mode clears all TPs with no confirm — add a confirm dialog (Claude's default; user may override).
