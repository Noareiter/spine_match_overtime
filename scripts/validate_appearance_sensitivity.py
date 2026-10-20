"""
Appearance-weight (w_app) turnover-sensitivity report.

NOT a LOFO (leave-one-FOV-out) validation. LOFO is model-side validation of
the Siamese encoder itself and lives in D:\\DeepNetSpineMatching's own
evaluate.py; nothing in this repo can perform it, since this repo has no
labeled ground truth for spine identity. This script instead answers a
narrower, repo-side question: for a real FOV's real spine matching
pipeline, how much does turning the appearance score on (w_app=1.0) change
the matching decisions actually produced by score_candidates_hybrid(),
relative to appearance being off (w_app=0.0)?

This is the council's "turnover ON/OFF sensitivity run" step. It measures
an effect; it does not certify one. If appearance systematically reduces
turnover (fewer LOST/NEW, more matched), that is a flag to bring to the
neuroscientists for review before trusting it in real annotation -- per the
council's guardrail, appearance can only ever argue FOR a match, so it
structurally biases toward "matched" if not scrutinized. A low delta is
reassuring; a high delta is not, by itself, damning -- it just means the
signal needs a human look.

Matching rule used here (independent of and simpler than the live UI's
review-queue/lineage-decision workflow, which requires a human in the
loop): greedy globally-best-first 1:1 assignment over all candidate pairs
above a score threshold. This is NOT what the annotator persists to disk;
it is a deterministic proxy good enough to compare "did the ranking change
enough to flip a decision" between two weight settings.

turnover_rate for a t1->t2 timepoint pair is defined as:
    (unmatched_t1_count + unmatched_t2_count) / (n_t1 + n_t2)

Usage:
    python scripts/validate_appearance_sensitivity.py
    python scripts/validate_appearance_sensitivity.py --checkpoint PATH --threshold 0.5

Requires D:\\DeepNetSpineMatching to be present on this machine (source of
the demo workspace and the appearance-model checkpoint) and torch/monai
installed in this environment (already required for the appearance
feature itself).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEEPNET_ROOT = Path(r"D:\DeepNetSpineMatching")
DEMO_RESPAN = DEEPNET_ROOT / "results" / "demo_spine_logic_lab" / "respan"
DEFAULT_CHECKPOINT = DEEPNET_ROOT / "data" / "checkpoints" / "DEMO10" / "fold_fov1_best.pt"

DEMO_TIMEPOINTS = ["pre-droplet", "end-droplet", "end-lever"]
DEMO_FOV = 1


def _tp_paths(tp: str, fov: int) -> Tuple[Path, Path]:
    tp_dir = DEMO_RESPAN / tp
    tiff_path = tp_dir / f"fov{fov}.tif"
    csv_path = tp_dir / "Tables" / f"fov{fov}_detected_spines_demo.csv"
    return tiff_path, csv_path


def _greedy_assignment(
    scored: pd.DataFrame, threshold: float
) -> Tuple[Dict[str, str], set, set]:
    """Global greedy 1:1 assignment: highest final_score first, skip if either
    side is already claimed or the score is below threshold."""
    assigned_t1: Dict[str, str] = {}
    claimed_t2: set = set()
    if scored.empty:
        return assigned_t1, claimed_t2, claimed_t2
    ranked = scored.sort_values("final_score", ascending=False)
    for _, row in ranked.iterrows():
        t1_id = str(row["t1_spine_id"])
        t2_id = str(row["t2_spine_id"])
        score = float(row["final_score"])
        if score < threshold:
            continue
        if t1_id in assigned_t1:
            continue
        if t2_id in claimed_t2:
            continue
        assigned_t1[t1_id] = t2_id
        claimed_t2.add(t2_id)
    return assigned_t1, claimed_t2, claimed_t2


def _run_one_pair(
    baseline_adapter,
    mtp_spine_matching,
    tp1: str,
    tp2: str,
    fov: int,
    threshold: float,
) -> Optional[dict]:
    tiff1, csv1 = _tp_paths(tp1, fov)
    tiff2, csv2 = _tp_paths(tp2, fov)
    for p in (tiff1, csv1, tiff2, csv2):
        if not p.is_file():
            print(f"  [skip] {tp1} -> {tp2}: missing {p}")
            return None

    t1_df = baseline_adapter.load_spines(csv1)
    t2_df = baseline_adapter.load_spines(csv2)

    linked, cross = mtp_spine_matching._candidate_rows(
        t1_df, t2_df, allowed_t1_by_t2={}, require_link=False
    )
    rows = linked + cross
    if not rows:
        print(f"  [skip] {tp1} -> {tp2}: no candidate pairs within z-gate")
        return None
    candidates = pd.DataFrame(rows)

    scored_off = baseline_adapter.score_candidates_hybrid(
        candidates,
        t1_df,
        t2_df,
        gating_z=mtp_spine_matching.MAX_Z,
        w_app=0.0,
        t1_tiff_path=str(tiff1),
        t2_tiff_path=str(tiff2),
    )
    scored_on = baseline_adapter.score_candidates_hybrid(
        candidates,
        t1_df,
        t2_df,
        gating_z=mtp_spine_matching.MAX_Z,
        w_app=1.0,
        t1_tiff_path=str(tiff1),
        t2_tiff_path=str(tiff2),
    )

    n_t1 = len(t1_df)
    n_t2 = len(t2_df)

    assigned_off, claimed_t2_off, _ = _greedy_assignment(scored_off, threshold)
    assigned_on, claimed_t2_on, _ = _greedy_assignment(scored_on, threshold)

    unmatched_t1_off = n_t1 - len(assigned_off)
    unmatched_t2_off = n_t2 - len(claimed_t2_off)
    unmatched_t1_on = n_t1 - len(assigned_on)
    unmatched_t2_on = n_t2 - len(claimed_t2_on)

    turnover_off = (unmatched_t1_off + unmatched_t2_off) / (n_t1 + n_t2)
    turnover_on = (unmatched_t1_on + unmatched_t2_on) / (n_t1 + n_t2)

    changed = []
    all_t1_ids = set(assigned_off) | set(assigned_on)
    for t1_id in sorted(all_t1_ids):
        t2_off = assigned_off.get(t1_id)
        t2_on = assigned_on.get(t1_id)
        if t2_off != t2_on:
            changed.append((t1_id, t2_off, t2_on))

    return {
        "pair": f"{tp1} -> {tp2}",
        "n_t1": n_t1,
        "n_t2": n_t2,
        "n_candidates": len(candidates),
        "turnover_off": turnover_off,
        "turnover_on": turnover_on,
        "turnover_delta": turnover_on - turnover_off,
        "n_changed_assignments": len(changed),
        "changed_assignments": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Appearance-model checkpoint (.pt). Default: DEMO10/fold_fov1_best.pt, "
        "trained/held-out on the demo FOV1 data this script runs against.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum final_score to accept a greedy assignment as 'matched' (default 0.5).",
    )
    parser.add_argument("--fov", type=int, default=DEMO_FOV)
    args = parser.parse_args()

    if not DEEPNET_ROOT.is_dir():
        print(f"Reference repo not found at {DEEPNET_ROOT} on this machine. Nothing to run.")
        return 1
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}. Nothing to run.")
        return 1
    if not DEMO_RESPAN.is_dir():
        print(f"Demo workspace not found at {DEMO_RESPAN}. Nothing to run.")
        return 1

    os.environ["SPINE_MATCHER_CHECKPOINT"] = str(args.checkpoint)

    from spine_annotator_backend import baseline_adapter, mtp_spine_matching  # noqa: E402

    print("=" * 72)
    print("Appearance-weight (w_app) turnover-sensitivity report")
    print(f"  workspace : {DEMO_RESPAN}")
    print(f"  fov       : {args.fov}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  threshold : {args.threshold}")
    print("  NOTE: this is a sensitivity measurement, not a LOFO validation")
    print("  and not a pass/fail gate. See module docstring.")
    print("=" * 72)

    results: List[dict] = []
    for tp1, tp2 in zip(DEMO_TIMEPOINTS[:-1], DEMO_TIMEPOINTS[1:]):
        print(f"\n[{tp1} -> {tp2}]")
        result = _run_one_pair(
            baseline_adapter, mtp_spine_matching, tp1, tp2, args.fov, args.threshold
        )
        if result is None:
            continue
        results.append(result)
        print(f"  n_t1={result['n_t1']} n_t2={result['n_t2']} n_candidates={result['n_candidates']}")
        print(f"  turnover  w_app=0.0: {result['turnover_off']:.3f}")
        print(f"  turnover  w_app=1.0: {result['turnover_on']:.3f}")
        print(f"  turnover  delta    : {result['turnover_delta']:+.3f}")
        print(f"  changed assignments: {result['n_changed_assignments']} / {result['n_t1']}")
        for t1_id, t2_off, t2_on in result["changed_assignments"]:
            print(f"    t1={t1_id}: w_app=0.0 -> t2={t2_off}   w_app=1.0 -> t2={t2_on}")

    if not results:
        print("\nNo timepoint pairs could be evaluated.")
        return 1

    print("\n" + "=" * 72)
    print("Summary across all pairs")
    total_changed = sum(r["n_changed_assignments"] for r in results)
    total_t1 = sum(r["n_t1"] for r in results)
    mean_delta = sum(r["turnover_delta"] for r in results) / len(results)
    print(f"  pairs evaluated          : {len(results)}")
    print(f"  total changed assignments: {total_changed} / {total_t1}")
    print(f"  mean turnover delta      : {mean_delta:+.3f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
