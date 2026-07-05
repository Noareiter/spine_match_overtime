# assume_t1_t2_onPre — pairwise inference from manual exports

Generate **any missing chronological** pairwise export (e.g. `pre-end droplet` from `pre-mid` + `mid-end`, or `mid-droplet` vs `end-lever`) from **all manual** `*_spine_annotator_export` folders (inferred/synthetic folders are **ignored** for input).

## Iterative workflow

1. Annotate baseline pairs (`pre-mid droplet`, …) and/or validate inferred pairs in the review UI (creates a **manual** export).
2. Run `infer_baseline_bridged_pairs.py` → new `*_inferred_spine_annotator_export/` for pairs still missing a manual export.
3. Review inferred → export manual → `archive_inferred_gp04.bat` moves inferred to `old_inffered/`.
4. **Run the script again** — it reads the new manual exports too, so more spines match on remaining pairs.

## Logic (union-find lineages)

Every manual export contributes `matched.csv` links and statuses. Lineages are merged across comparisons (pre–mid, pre–end, **mid–end**, …). For target pair **T_A** vs **T_B**:

| Outcome | Rule |
|--------|------|
| **matched.csv** | Same lineage, biological status at both timepoints |
| **lost.csv** | Present at T_A, absent/lost at T_B |
| **new.csv** | Absent/lost at T_A, present at T_B |
| **unresolved_manual_review.csv** | ID clash in lineage, matched T1/T2 clash, ignored/removed, or ambiguous |

**Chronological routing (reduces false unresolved):**

| Rule | Behaviour |
|------|-----------|
| **Birth** | NEW at T_B → absent before T_B; inferred pairs with T_A < T_B < T_C → `new.csv` at T_C |
| **Death** | LOST at T_B → pruned from all later timepoints; crossing pairs → `lost.csv` at T_A |
| **Artifact** | Forward from first artifact TP → `removed_t1.csv` / `removed_t2.csv` (not unresolved) |
| **Unresolved** | Only `lineage_id_clash` or `ignore_breaks_chain` (ignore strictly between T_A and T_B) |

**Forward artifact rule:** also applied in the transitive review UI via `transitive_artifact_blacklist.py`.

TIFF stacks for review are copied into `input_files/` from `matching_activity_log.txt` (or `metadata.json`) on each baseline comparison export.

## Input layout

```
<IMAGING_ROOT>/<ANIMAL_ID>/respan/results/fovN/<comparison>/<timestamp>_spine_annotator_export/
  matched.csv, new.csv, lost.csv, removed_*.csv, ignored_*.csv
  matching_activity_log.txt
```

Baseline comparison folder names (Step 3 convention):

- `pre-mid droplet`, `pre-end droplet`, `pre droplet-end lever`, `pre droplet - return to droplet`

## Usage

```cmd
python "D:\learning_project_spines\code final\assume_t1_t2_onPre\infer_baseline_bridged_pairs.py" ^
  --animal-id GP08 ^
  --imaging-root "E:\Noa\Pons - layer 5\Imaging"
```

Or point directly at results:

```cmd
python infer_baseline_bridged_pairs.py ^
  --results-root "E:\Noa\Pons - layer 5\Imaging\GP04\respan\results"
```

### Options

| Flag | Description |
|------|-------------|
| `--animal-id` | Animal folder (default: `GP04`) |
| `--imaging-root` | Parent of `GP04`, `GP08`, … |
| `--results-root` | Override `.../respan/results` path |
| `--fovs 1 2` | Subset of FOVs |
| `--pairs "mid-droplet - end-lever"` | Only named pairs |
| `--dry-run` | Print plan only |
| `--force` | Overwrite inferred exports |
| `--no-tiffs` | Skip `input_files/` |
| `--validate` | Optional ID sanity check vs Tables (off by default) |
| `--symlink-tiffs` | Symlink instead of copy |

## Output layout

```
results/fov1/mid-droplet - end-lever/latest_inferred_spine_annotator_export/
  input_files/
    t1_mid-droplet_fov1.tif
    t2_end-lever_fov1.tif
    README.txt
  matched.csv
  new.csv
  lost.csv
  unresolved_manual_review.csv
  metadata.json
  session_inputs/          # canonical t1/t2_detected_spines.csv (after validation)
  validation_failed.txt    # only if provenance check failed
```

After writing CSVs and `input_files/`, the script copies canonical `detected_spines` Tables into `session_inputs/` (no fraction sanity check by default). Use `--validate` to enable the legacy ≥60% ID match gate. Coordinates at review time always come from those Tables CSVs.

Default: all **10** chronological timepoint pairs per FOV that do not already have a **manual** export (including `pre-droplet` ↔ follow-up when inferable from the lineage graph). Comparison folders with **only** inferred exports are not used as sources. Skips pairs that already have a manual export unless `--force`. Requires **≥2** manual exports per FOV.

When several manual `*_spine_annotator_export` folders exist under one comparison, only the **most recently updated** is used (metadata timestamp, folder name date, or file mtime).

Manifest: `results/baseline_bridge_inference_manifest.json`

## Related

- Step 3 pipeline: `code final/step 3 - dendrite information/`
- Spine annotator: `code final/spine annotator/`
