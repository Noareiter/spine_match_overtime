# Cross-FOV duplicate spines

One physical spine that lies in the overlap of two FOVs is in the registry twice (once per FOV). This tool shows every
candidate pair and lets you decide by eye; the result is a registry with each spine counted once.

## Start

```
start_duplicates.bat
```
(or `python duplicates\duplicates_review.py`). The animal is taken from `config\annotator.json` like the annotator
itself; the page opens at http://127.0.0.1:8020/ . Options: `--port`, `--rebuild` (new candidate list + crops, decisions
kept), `--max-um 5`, `--export` / `--no-serve` (write the registry from the saved decisions without opening the page),
`--respan <folder>` (ignore the config).

## Needs

- `_annotator\Results final\spine_registry_wide.csv` (annotator, Results final built)
- `tree_mapping\spine_tree_assignment*.csv` (tree mapper export, newest file is used) - the branch of every spine and its
  position in the structural frame. Candidates = same branch, different FOV, closer than 5 um.
- the stacks `<timepoint>\fovN.tif` for the crops.

## Review

Top row = FOV A, bottom = FOV B, one crop per timepoint from that FOV's own stack (18 x 18 um, MIP z +- 2).
Red cross = the spine at that timepoint; yellow = position taken from the nearest timepoint where it had coordinates;
green circles = other registry spines of that FOV present there. Keys: `1` keep top, `2` keep bottom, `3` different
spines (both stay), `U` undo, `N` skip. Pairs whose fate differs between the FOVs are queued first; tick
"show concordant clusters too" for the rest. A cluster of 3+ FOVs is resolved pair by pair. Every click is saved.

## Output (only under `_annotator\Analysis\twin_review\`)

- `decisions.json` - every click
- `twin_exclusions.csv` - the dropped copies (fov, lineage_key, kept copy); the MATLAB tree scripts read this
- `registry_resolved.csv` - always current
- **`spine_registry_wide_deduplicated.csv`** - written by "Finish / export" (or `--export`): the final registry minus the
  dropped copies, same columns. The tree mapper ("Open respan folder") loads this file instead of
  `spine_registry_wide.csv` when it exists; manual points and exclusions survive (keyed by lineage_key).

`Results final\` is never written (the annotator rebuilds that folder from scratch), nor `_annotator\fovN\` or
`tree_mapping\`.

## Order of work

1. annotator -> Results final
2. tree mapper on the full registry -> `tree_mapping\spine_tree_assignment*.csv`
3. this tool -> review -> Finish / export
4. tree mapper again: it now loads the deduplicated registry.

The candidate list is built from the full registry and kept (`data.json`); it is rebuilt only with `--rebuild` or when
`data.json` is missing, so re-mapping on the deduplicated registry does not throw the decisions away.
