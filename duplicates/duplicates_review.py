"""Cross-FOV duplicate review: one physical spine imaged in two overlapping FOVs appears twice in the registry.

usage:  python duplicates/duplicates_review.py [--config config/annotator.json] [--port 8020] [--rebuild]
                                               [--max-um 5] [--export] [--no-serve]
        (or start_duplicates.bat from the project root)

The animal comes from config/annotator.json (workspace -> respan folder), exactly like the annotator itself.

Candidates: registry spines on the SAME branch of the tree mapper export (tree_mapping/spine_tree_assignment*.csv),
in DIFFERENT FOVs, closer than --max-um in the mapper's structural frame, matched one-to-one (closest first).
Every candidate spine is shown at every timepoint as a crop from its OWN FOV's stack; decisions per pair:
keep FOV A / keep FOV B / different spines.  A cluster of more than two FOVs is resolved in sequence
(A vs B, then the winner vs C ...).  Every spine ends up counted once.

Reads (never modifies):
  <respan>/_annotator/Results final/spine_registry_wide.csv      the annotator's final registry
  <respan>/tree_mapping/spine_tree_assignment*.csv               newest tree mapper export (branch + structural x/y)
  <respan>/<timepoint>/fovN*.tif                                 stacks, for the crops
Writes ONLY under  <respan>/_annotator/Analysis/twin_review/ :
  data.json, crops/*.png, decisions.json (saved after every click), twin_exclusions.csv (dropped copies),
  registry_resolved.csv (always current) and, on "Finish / export" (or --export):
  spine_registry_wide_deduplicated.csv  = the registry with the dropped copies removed, one row per physical spine,
                                          same columns as spine_registry_wide.csv.  The tree mapper's "Open respan folder"
                                          prefers this file when it exists.
Never touches Results final/ (the annotator rebuilds that folder from scratch), _annotator/fovN/ or tree_mapping/.

Workflow: map once on the full registry -> review duplicates here -> Finish / export -> re-open the respan folder in the
tree mapper (it loads the deduplicated registry; manual points / exclusions survive, they are keyed by lineage_key).
The candidate list is built from the FULL registry and is rebuilt only with --rebuild or when data.json is missing,
so a later re-mapping on the deduplicated registry does not discard the decisions already made.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

PRESENT = {"matched", "manual", "new"}
WIN = 160          # crop window in functional px (0.11 um/px -> ~18 um)
ZHALF = 2          # MIP over z +- ZHALF
SCALE = 2          # upscale factor for the PNG
OUT_SUBDIR = ("_annotator", "Analysis", "twin_review")
EXPORT_NAME = "spine_registry_wide_deduplicated.csv"


# ----------------------------------------------------------------------------------------------- config / paths
def resolve_respan(config_path: str | None) -> tuple[Path, str]:
    """respan folder from annotator.json (same rules as the annotator), without creating anything."""
    from spine_annotator_backend.animal_config import load_config, resolve_workspace_path

    cfg = load_config(config_path)
    ws = resolve_workspace_path(cfg)
    if not ws.is_dir():
        raise SystemExit(f"ERROR: workspace folder not found: {ws}\nConnect the drive or edit workspace in {cfg.config_path}")
    respan = ws if ws.name.lower() == "respan" else ws / "respan"
    if not respan.is_dir():
        raise SystemExit(f"ERROR: respan folder not found: {respan}")
    return respan, cfg.animal_id


def find_stack(root: Path, tp: str, fov: str) -> Path | None:
    try:
        from spine_annotator_backend.animal_layout import find_tiff_for_fov
        p = find_tiff_for_fov(root / tp, int(fov))
        if p is not None:
            return p
    except Exception:
        pass
    p = root / tp / f"fov{fov}.tif"
    return p if p.exists() else None


# ----------------------------------------------------------------------------------------------- tables
def load_tables(root: Path):
    tms = sorted(glob.glob(str(root / "tree_mapping" / "spine_tree_assignment*.csv")), key=os.path.getmtime)
    if not tms:
        raise SystemExit(f"ERROR: no tree_mapping/spine_tree_assignment*.csv under {root} - run the tree mapper first")
    tm = tms[-1]
    tree = {(r["orig_fov"], r["orig_lineage_key"]): r for r in csv.DictReader(open(tm, encoding="utf-8-sig"))
            if r["branch_id"] and r["status"] != "unassigned"}
    regp = root / "_annotator" / "Results final" / "spine_registry_wide.csv"
    if not regp.exists():
        raise SystemExit(f"ERROR: {regp} not found - build Results final in the annotator first")
    reg_rows = list(csv.DictReader(open(regp, encoding="utf-8-sig")))
    cols = list(reg_rows[0].keys())
    tps = [c[len("status_"):] for c in cols if c.startswith("status_")]
    reg = {(r["fov"], r["lineage_key"]): r for r in reg_rows}
    return tree, reg, reg_rows, cols, tps, tm


def candidates(tree, reg, tps, max_um):
    px = float(next(iter(tree.values()))["pixel_size_um"])
    keys = [k for k in tree if k in reg]
    by_branch = {}
    for k in keys:
        by_branch.setdefault(int(float(tree[k]["branch_id"])), []).append(k)
    sxy = {k: (float(tree[k]["struct_x_px"]) * px, float(tree[k]["struct_y_px"]) * px, float(tree[k]["struct_plane"] or 0)) for k in keys}
    pairs = []
    for b, ks in by_branch.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, c = ks[i], ks[j]
                if a[0] == c[0]:
                    continue
                d = float(np.hypot(sxy[a][0] - sxy[c][0], sxy[a][1] - sxy[c][1]))
                if d <= max_um:
                    pairs.append((d, a, c, b))
    pairs.sort()
    # one-to-one per FOV pair, closest first; remember the alternatives for display
    used = {}
    chosen, alt = [], {}
    for d, a, c, b in pairs:
        if used.get((a, c[0])) or used.get((c, a[0])):
            alt.setdefault(a, []).append((c, d)); alt.setdefault(c, []).append((a, d))
            continue
        used[(a, c[0])] = True; used[(c, a[0])] = True
        chosen.append((d, a, c, b))
    # clusters (union-find)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for d, a, c, b in chosen:
        parent[find(a)] = find(c)
    clusters = {}
    for d, a, c, b in chosen:
        clusters.setdefault(find(a), {"members": set(), "branch": b, "pairs": []})
        clusters[find(a)]["members"].update([a, c]); clusters[find(a)]["pairs"].append((a, c, d))
    vec = lambda k: [reg[k][f"status_{t}"] in PRESENT for t in tps]
    out = []
    for cid, cl in clusters.items():
        members = sorted(cl["members"], key=lambda k: (int(k[0]), k[1]))
        pd = {}
        for a, c, d in cl["pairs"]:
            pd[f"{a[0]}|{a[1]}__{c[0]}|{c[1]}"] = d
        concordant = all(vec(m) == vec(members[0]) for m in members)
        out.append({
            "id": "c%04d" % len(out), "branch": cl["branch"], "concordant": concordant,
            "members": [{"fov": m[0], "lineage_key": m[1], "key": f"{m[0]}|{m[1]}", "plane": sxy[m][2],
                         "status": {t: reg[m][f"status_{t}"] for t in tps},
                         "alt": [{"key": f"{o[0]}|{o[1]}", "d": round(dd, 2)} for o, dd in alt.get(m, [])]} for m in members],
            "pair_dist": {k: round(v, 2) for k, v in pd.items()},
        })
    out.sort(key=lambda c: (c["concordant"], c["branch"]))
    return out, px


def coords_for(reg_row, tps, t_index):
    """x, y, z at timepoint t_index; if missing, the nearest timepoint with coordinates (same FOV, registered sessions)."""
    order = sorted(range(len(tps)), key=lambda j: (abs(j - t_index), j))
    for j in order:
        t = tps[j]
        if reg_row.get(f"{t}_x", "") not in ("", None):
            return float(reg_row[f"{t}_x"]), float(reg_row[f"{t}_y"]), float(reg_row[f"{t}_z"] or 0), (j == t_index)
    return None


# ----------------------------------------------------------------------------------------------- crops
def render_crops(root: Path, out: Path, clusters, reg, reg_rows, tps):
    from PIL import Image, ImageDraw
    import tifffile
    try:
        from spine_annotator_backend.crop_service import centered_crop
    except Exception:
        def centered_crop(stack, x, y, z, width, height, depth):
            zc, yc, xc = int(round(z)), int(round(y)), int(round(x))
            z0, z1 = max(0, zc - depth // 2), min(stack.shape[0], zc - depth // 2 + depth)
            y0, y1 = max(0, yc - height // 2), min(stack.shape[1], yc - height // 2 + height)
            x0, x1 = max(0, xc - width // 2), min(stack.shape[2], xc - width // 2 + width)
            return stack[z0:z1, y0:y1, x0:x1], {"source_bounds": {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1}}
    crops_dir = out / "crops"; crops_dir.mkdir(parents=True, exist_ok=True)
    needed = {}   # (tp index, fov) -> list of member keys
    for cl in clusters:
        for m in cl["members"]:
            for ti, t in enumerate(tps):
                needed.setdefault((ti, m["fov"]), []).append(m["key"])
    by_fov_tp = {}
    for r in reg_rows:
        for ti, t in enumerate(tps):
            if r[f"status_{t}"] in PRESENT and r.get(f"{t}_x", "") != "":
                by_fov_tp.setdefault((ti, r["fov"]), []).append((float(r[f"{t}_x"]), float(r[f"{t}_y"]), r["lineage_key"]))
    n_done = 0
    for (ti, fov), keys in sorted(needed.items()):
        missing = [k for k in keys if not (crops_dir / f"{k.replace('|', '_')}__{ti}.png").exists()]
        if not missing:
            continue
        tif = find_stack(root, tps[ti], fov)
        if tif is None:
            print("missing stack:", root / tps[ti], f"fov{fov}"); continue
        stack = tifffile.imread(str(tif))
        others = by_fov_tp.get((ti, fov), [])
        for key in missing:
            png = crops_dir / f"{key.replace('|', '_')}__{ti}.png"
            row = reg[tuple(key.split("|", 1))]
            c = coords_for(row, tps, ti)
            if c is None:
                continue
            x, y, z, exact = c
            crop, meta = centered_crop(stack, x, y, z, WIN, WIN, 2 * ZHALF + 1)
            mip = crop.max(axis=0).astype(np.float32)
            lo, hi = np.percentile(mip, [1, 99.7]); img = np.clip((mip - lo) / max(hi - lo, 1), 0, 1)
            im = Image.fromarray((img * 255).astype(np.uint8)).resize((mip.shape[1] * SCALE, mip.shape[0] * SCALE), Image.BILINEAR).convert("RGB")
            d = ImageDraw.Draw(im)
            b = meta["source_bounds"]
            for ox, oy, lk in others:
                if b["x0"] <= ox < b["x1"] and b["y0"] <= oy < b["y1"] and lk != row["lineage_key"]:
                    px_, py_ = (ox - b["x0"]) * SCALE, (oy - b["y0"]) * SCALE
                    d.ellipse([px_ - 4, py_ - 4, px_ + 4, py_ + 4], outline=(80, 220, 80), width=2)
            cx, cy = (x - b["x0"]) * SCALE, (y - b["y0"]) * SCALE
            col = (255, 60, 60) if exact else (255, 200, 0)
            d.line([cx - 14, cy, cx - 5, cy], fill=col, width=2); d.line([cx + 5, cy, cx + 14, cy], fill=col, width=2)
            d.line([cx, cy - 14, cx, cy - 5], fill=col, width=2); d.line([cx, cy + 5, cx, cy + 14], fill=col, width=2)
            im.save(png); n_done += 1
        del stack
        print(f"  {tps[ti]} fov{fov}: {len(missing)} crops")
    return n_done


# ----------------------------------------------------------------------------------------------- page
INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Duplicate spines review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#15181d;color:#e8e8e8}
header{padding:10px 16px;background:#20242b;display:flex;gap:24px;align-items:center;flex-wrap:wrap}
header b{color:#9cc8ff} .row{display:flex;gap:8px;padding:8px 16px;align-items:flex-start}
.lab{width:150px;font-size:13px;color:#bbb;padding-top:8px} .lab b{display:block;font-size:15px;color:#fff}
.cell{text-align:center;font-size:12px;color:#ccc} .cell img{display:block;border:1px solid #333;width:300px;height:300px;object-fit:contain;background:#000}
.cell .st{padding:3px;margin-top:2px;border-radius:3px} .st.on{background:#245a2f} .st.off{background:#5a2424}
.btns{padding:10px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center} button{font-size:15px;padding:10px 18px;border:0;border-radius:6px;cursor:pointer}
#bA{background:#2f6fd6;color:#fff}#bB{background:#8a4fd6;color:#fff}#bD{background:#555;color:#fff}#bU{background:#333;color:#ddd}
#bF{background:#1f7a4a;color:#fff;margin-left:auto} #msg{font-size:13px;color:#9cc8ff;white-space:pre-line}
.small{font-size:12px;color:#999} .tag{padding:2px 8px;border-radius:10px;background:#3a3f48;font-size:12px}
</style></head><body>
<header><span><b id="prog"></b></span><span id="info"></span><span class="small">keys: 1 = keep top · 2 = keep bottom · 3 = different spines · U = undo · N = skip</span>
<label class="small"><input type="checkbox" id="showConc"> show concordant clusters too</label><span id="animal" class="small"></span></header>
<div id="rows"></div>
<div class="btns"><button id="bA">1 · keep TOP (FOV <span id="fa"></span>)</button><button id="bB">2 · keep BOTTOM (FOV <span id="fb"></span>)</button><button id="bD">3 · different spines</button><button id="bU">U · undo</button><button id="bN">N · skip</button>
<button id="bF">Finish / export deduplicated registry</button></div>
<div id="msg" style="padding:0 16px 16px"></div>
<script>
let D=null, dec={}, hist=[], queue=[], qi=0;
async function load(){ D=await (await fetch('data.json')).json(); try{dec=await (await fetch('decisions.json?'+Date.now())).json();}catch(e){dec={};}
  document.getElementById('animal').textContent=(D.animal_id?D.animal_id+' · ':'')+(D.tree_file||'');
  document.getElementById('showConc').onchange=buildQueue; buildQueue(); }
function pairKey(a,b){return a.key+'__'+b.key;}
// resolve a cluster from decisions: returns {next:[a,b]|null, kept:[...], dropped:[...]}
function resolve(cl){ let m=cl.members; let kept=[m[0]]; let cur=m[0]; let dropped=[];
  for(let i=1;i<m.length;i++){ let k=pairKey(cur,m[i]); let d=dec[k];
    if(!d) return {next:[cur,m[i]],kept,dropped};
    if(d==='A'){dropped.push(m[i]);} else if(d==='B'){dropped.push(cur); kept=kept.filter(x=>x!==cur); kept.push(m[i]); cur=m[i];} else {kept.push(m[i]);}
  } return {next:null,kept,dropped}; }
function buildQueue(){ let conc=document.getElementById('showConc').checked; queue=D.clusters.filter(c=>conc||!c.concordant); qi=0;
  while(qi<queue.length && !resolve(queue[qi]).next) qi++; show(); }
function show(){ let done=D.clusters.filter(c=>!resolve(c).next).length; document.getElementById('prog').textContent=`cluster ${Math.min(qi+1,queue.length)} / ${queue.length} in queue · resolved ${done} / ${D.clusters.length}`;
  let rows=document.getElementById('rows'); rows.innerHTML='';
  if(qi>=queue.length){document.getElementById('info').textContent='queue finished'; return;}
  let cl=queue[qi]; let r=resolve(cl); let [a,b]=r.next; let pk=pairKey(a,b); let dist=cl.pair_dist[pk] ?? cl.pair_dist[pairKey(b,a)] ?? '?';
  document.getElementById('info').innerHTML=`branch <b>${cl.branch}</b> · struct distance <b>${dist} µm</b> · Δplane ${Math.abs(a.plane-b.plane)} · cluster of ${cl.members.length} FOVs ${cl.concordant?'<span class=tag>concordant</span>':'<span class=tag style="background:#6b3b2a">different fate</span>'}`;
  document.getElementById('fa').textContent=a.fov; document.getElementById('fb').textContent=b.fov;
  for(const [m,label] of [[a,'TOP'],[b,'BOTTOM']]){ let div=document.createElement('div'); div.className='row';
    let alt=m.alt.length?`<span class=small>also near: ${m.alt.map(x=>x.key+' ('+x.d+' µm)').join(', ')}</span>`:'';
    div.innerHTML=`<div class=lab><b>${label} · FOV ${m.fov}</b>${m.lineage_key}<br>plane ${m.plane}<br>${alt}</div>`;
    D.tps.forEach((t,ti)=>{ let c=document.createElement('div'); c.className='cell'; let st=m.status[t]; let on=['matched','manual','new'].includes(st);
      c.innerHTML=`<img src="crops/${m.key.replace('|','_')}__${ti}.png" onerror="this.style.opacity=.2"><div>${t}</div><div class="st ${on?'on':'off'}">${st}</div>`; div.appendChild(c); });
    rows.appendChild(div); }
}
async function decide(choice){ if(qi>=queue.length) return; let cl=queue[qi]; let [a,b]=resolve(cl).next; let k=pairKey(a,b); dec[k]=choice; hist.push(k);
  await fetch('decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,choice:choice,cluster:cl.id})});
  if(!resolve(cl).next){ qi++; while(qi<queue.length && !resolve(queue[qi]).next) qi++; } show(); }
function undo(){ let k=hist.pop(); if(!k) return; delete dec[k]; fetch('decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,choice:null})}); buildQueue(); }
function skip(){ qi++; while(qi<queue.length && !resolve(queue[qi]).next) qi++; show(); }
async function finish(){ let unresolved=D.clusters.filter(c=>resolve(c).next).length;
  if(unresolved && !confirm(`${unresolved} cluster(s) not reviewed yet - their copies all stay in the registry. Export anyway?`)) return;
  let r=await (await fetch('export',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).json();
  document.getElementById('msg').textContent=`written: ${r.path}\n${r.rows_in} rows in the registry -> ${r.rows_out} rows (${r.dropped} duplicate copies removed, ${r.unresolved} cluster(s) unreviewed)\nTree mapper: "Open respan folder" now loads this file.`; }
document.getElementById('bA').onclick=()=>decide('A'); document.getElementById('bB').onclick=()=>decide('B'); document.getElementById('bD').onclick=()=>decide('different');
document.getElementById('bU').onclick=undo; document.getElementById('bN').onclick=skip; document.getElementById('bF').onclick=finish;
document.addEventListener('keydown',e=>{ if(e.target.tagName==='INPUT') return; if(e.key==='1')decide('A'); else if(e.key==='2')decide('B'); else if(e.key==='3')decide('different'); else if(e.key==='u'||e.key==='U')undo(); else if(e.key==='n'||e.key==='N')skip(); });
load();
</script></body></html>"""


# ----------------------------------------------------------------------------------------------- outputs
def dropped_copies(clusters, decisions):
    """{dropped key: kept key} from the decisions; returns also the number of clusters not fully reviewed."""
    dropped, unresolved = {}, 0
    for cl in clusters:
        m = cl["members"]; cur = m[0]
        for i in range(1, len(m)):
            d = decisions.get(f"{cur['key']}__{m[i]['key']}")
            if d is None:
                unresolved += 1; break
            if d == "A":
                dropped[m[i]["key"]] = cur["key"]
            elif d == "B":
                dropped[cur["key"]] = m[i]["key"]; cur = m[i]
    return dropped, unresolved


def write_outputs(out: Path, clusters, reg_rows, cols, decisions):
    """twin_exclusions.csv + registry_resolved.csv from the current decisions (unreviewed clusters: nothing dropped)."""
    dropped, _ = dropped_copies(clusters, decisions)
    with open(out / "twin_exclusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fov", "lineage_key", "kept_fov", "kept_lineage_key"])
        for k, kk in sorted(dropped.items()):
            w.writerow(k.split("|", 1) + kk.split("|", 1))
    with open(out / "registry_resolved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in reg_rows:
            if f"{r['fov']}|{r['lineage_key']}" not in dropped:
                w.writerow(r)
    return len(dropped)


def export_registry(out: Path, clusters, reg_rows, cols, decisions) -> dict:
    """spine_registry_wide_deduplicated.csv: the final registry minus the dropped copies (same columns, same row order)."""
    dropped, unresolved = dropped_copies(clusters, decisions)
    path = out / EXPORT_NAME
    n_out = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in reg_rows:
            if f"{r['fov']}|{r['lineage_key']}" not in dropped:
                w.writerow(r); n_out += 1
    info = {"path": str(path), "rows_in": len(reg_rows), "rows_out": n_out, "dropped": len(dropped), "unresolved": unresolved}
    print(f"exported {path}: {info['rows_in']} -> {info['rows_out']} rows ({info['dropped']} duplicate copies removed, "
          f"{unresolved} cluster(s) unreviewed)")
    return info


# ----------------------------------------------------------------------------------------------- server
def serve(out: Path, clusters, reg_rows, cols, port, open_browser):
    dec_path = out / "decisions.json"
    decisions = json.load(open(dec_path)) if dec_path.exists() else {}
    lock = threading.Lock()

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(out), **k)

        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            super().do_GET()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); msg = json.loads(self.rfile.read(n) or b"{}")
            with lock:
                if self.path.rstrip("/") == "/export":
                    self._json(export_registry(out, clusters, reg_rows, cols, decisions)); return
                if msg.get("choice") is None:
                    decisions.pop(msg["key"], None)
                else:
                    decisions[msg["key"]] = msg["choice"]
                json.dump(decisions, open(dec_path, "w"), indent=1)
                nd = write_outputs(out, clusters, reg_rows, cols, decisions)
            self._json({"ok": True, "decisions": len(decisions), "dropped": nd})

    url = f"http://127.0.0.1:{port}/"
    print(f"duplicate review at {url}   (decisions -> {dec_path})", flush=True)
    if open_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Cross-FOV duplicate spine review (animal from config/annotator.json)")
    ap.add_argument("--config", default=os.environ.get("SPINE_CONFIG", str(PROJECT_ROOT / "config" / "annotator.json")))
    ap.add_argument("--respan", default=None, help="respan folder (overrides the config)")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the candidate list and crops (decisions are kept)")
    ap.add_argument("--max-um", type=float, default=5.0)
    ap.add_argument("--export", action="store_true", help=f"write {EXPORT_NAME} from the saved decisions")
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if a.respan:
        root, animal = Path(a.respan), ""
    else:
        root, animal = resolve_respan(a.config)
    print(f"Animal: {animal or '(from --respan)'}\nRespan: {root}", flush=True)
    out = root.joinpath(*OUT_SUBDIR); out.mkdir(parents=True, exist_ok=True)
    tree, reg, reg_rows, cols, tps, tm = load_tables(root)
    data_path = out / "data.json"
    if a.rebuild or not data_path.exists():
        clusters, px = candidates(tree, reg, tps, a.max_um)
        print(f"{len(clusters)} clusters ({sum(not c['concordant'] for c in clusters)} with different fate), "
              f"{sum(len(c['members']) for c in clusters)} spines, max {a.max_um} um in the structural frame ({px} um/px); tree file {Path(tm).name}")
        json.dump({"tps": tps, "clusters": clusters, "max_um": a.max_um, "struct_px_um": px, "tree_file": Path(tm).name,
                   "animal_id": animal}, open(data_path, "w"), indent=1)
        print("rendering crops ...", flush=True); n = render_crops(root, out, clusters, reg, reg_rows, tps); print(f"{n} crops written")
    data = json.load(open(data_path)); clusters = data["clusters"]
    dec_path = out / "decisions.json"
    decisions = json.load(open(dec_path)) if dec_path.exists() else {}
    nd = write_outputs(out, clusters, reg_rows, cols, decisions)
    _, unresolved = dropped_copies(clusters, decisions)
    print(f"{len(clusters)} clusters, {len(clusters) - unresolved} reviewed, {nd} duplicate copies dropped so far", flush=True)
    if a.export:
        export_registry(out, clusters, reg_rows, cols, decisions)
    if not a.no_serve:
        serve(out, clusters, reg_rows, cols, a.port, not a.no_browser)


if __name__ == "__main__":
    main()
