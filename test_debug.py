#!/usr/bin/env python3
import json
from pathlib import Path

fov1 = Path(r"E:\Noa\Pons - layer 5\Imaging\GP04\try\respan\_annotator\fov1")
dec_path = fov1 / "lineage_decisions.json"

print(f"File exists: {dec_path.is_file()}")

data = json.loads(dec_path.read_text(encoding="utf-8-sig"))
lineages = data.get("lineages") or []
tps = data.get("timepoint_names") or []

print(f"Total lineages: {len(lineages)}")
print(f"Timepoint names: {tps}")

trailing_blank_count = 0
for lin in lineages[:10]:  # Just check first 10
    per_tp = lin.get("per_tp") or {}
    lineage_key = str(lin.get("lineage_key") or lin.get("pre_spine_id") or "").strip()

    # Find true last match
    last_match_tp = None
    for tp in tps:
        td = per_tp.get(tp) or {}
        if str(td.get("spine_id") or "").strip():
            last_match_tp = tp

    if last_match_tp is None:
        continue

    # Check trailing TPs
    last_match_idx = tps.index(last_match_tp)
    trailing_blanks = []
    for tp in tps[last_match_idx + 1:]:
        td = per_tp.get(tp) or {}
        sid = str(td.get("spine_id") or "").strip()
        fate = str(td.get("fate") or "").strip()
        if not sid and not fate:
            trailing_blanks.append(tp)

    print(f"{lineage_key}: last_match={last_match_tp}, trailing_blanks={trailing_blanks}")
