import os
from pathlib import Path

os.environ["SPINE_CONFIG"] = str(Path(r"D:\spine_match_overtime\config\annotator.json").resolve())
from spine_annotator_backend import baseline_adapter, mtp_spine_viewer as mv
from spine_annotator_backend.animal_api import load_animal_inventory
from spine_annotator_backend import spine_lineage_store as ls

inv = load_animal_inventory(
    animal_id="GP04", workspace=Path(r"E:/Noa/Pons - layer 5/Imaging/GP04/try")
)
respan = Path(inv.respan_root)
fov = 1
mv._STATE.respan_root = str(respan)
mv._STATE.fov = fov
mv._STATE.animal_id = "GP04"
mv._STATE.timepoint_names = [tp.name for tp in inv.timepoints if tp.csv_path]
mv._STATE.t1_timepoint = mv._STATE.timepoint_names[0]
mv._STATE.spine_lookup = {}
mv._STATE.spine_dfs = {}
mv._STATE.files = {}
for tp in inv.timepoints:
    if not tp.csv_path:
        continue
    df = baseline_adapter.load_spines(Path(tp.csv_path))
    mv._STATE.spine_dfs[tp.name] = df
    mv._STATE.spine_lookup[tp.name] = baseline_adapter.to_lookup(df)
    mv._STATE.files[tp.name] = {"csv": tp.csv_path, "tiff": tp.tiff_path or ""}
mv._apply_catalog(respan)
mv._STATE.review_progress = ls.load_progress(respan, fov)
mv._STATE.queue_phase_index = int(mv._STATE.review_progress.get("phase_index", 0))
mv._build_queues()
mv._build_phase_queues(respan)
lc = mv._lineage_phase_count()
print("phase_queues lens", [len(q) for q in mv._STATE.phase_queues])
print("lc", lc, "unrev slot", len(mv._STATE.phase_queues[lc + 1]))
print("idx before", mv._STATE.queue_phase_index)
mv._rebuild_spine_id_list()
print("t1 after rebuild", len(mv._STATE.t1_spine_ids))
mv._ensure_nonempty_phase(respan)
print("idx after", mv._STATE.queue_phase_index)
mv._rebuild_spine_id_list()
print("t1 final", len(mv._STATE.t1_spine_ids), mv._STATE.t1_spine_ids[:3])
