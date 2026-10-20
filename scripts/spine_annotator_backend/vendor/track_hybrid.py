from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


FEATURE_KEYS = [
    "spine_length_euclidean",
    "spine_bbox_vol",
    "neck_bbox_vol",
    "spine_convex_vol",
    "neck_area",
    "neck_surf_area",
]
FEATURE_R_WEIGHTS = np.array([0.76, 0.71, 0.60, 0.59, 0.54, 0.52], dtype=np.float32)
FEATURE_R_WEIGHTS = FEATURE_R_WEIGHTS / FEATURE_R_WEIGHTS.sum()


@dataclass
class RobustScaler:
    median: float
    iqr: float

    def transform(self, values: pd.Series) -> pd.Series:
        if not np.isfinite(self.iqr) or abs(self.iqr) < 1e-9:
            return values * 0.0
        return (values - self.median) / self.iqr


def canonical_id(value: object) -> str:
    raw = str(value).strip()
    if raw == "" or raw.lower() == "nan":
        return ""
    try:
        as_float = float(raw)
        if np.isfinite(as_float) and abs(as_float - round(as_float)) < 1e-9:
            return str(int(round(as_float)))
    except Exception:
        pass
    return raw


def load_spines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "spine_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"spine_id": "id"})
    required = {"id", "x", "y", "z"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.copy()
    df["id"] = df["id"].apply(canonical_id)
    for col in ("x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_KEYS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df = df.dropna(subset=["x", "y", "z"])
    return df.reset_index(drop=True)


def load_manual_matches(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path).copy()
    for col in ("t1_spine_id", "t2_spine_id"):
        truth[col] = truth[col].apply(canonical_id)
    truth["status"] = truth["status"].astype(str).str.strip().str.lower()
    return truth


def robust_scalers(t1_df: pd.DataFrame, t2_df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, RobustScaler]:
    scalers: Dict[str, RobustScaler] = {}
    both = pd.concat([t1_df, t2_df], ignore_index=True)
    for col in cols:
        vals = pd.to_numeric(both[col], errors="coerce").dropna()
        if vals.empty:
            scalers[col] = RobustScaler(0.0, 1.0)
            continue
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        med = float(np.percentile(vals, 50))
        iqr = max(q3 - q1, 1e-6)
        scalers[col] = RobustScaler(med, iqr)
    return scalers


def add_scaled_features(df: pd.DataFrame, scalers: Dict[str, RobustScaler], cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_scaled"] = scalers[col].transform(pd.to_numeric(out[col], errors="coerce"))
    return out


def build_candidates(t1_df: pd.DataFrame, t2_df: pd.DataFrame, gating_xy: float, gating_z: float) -> pd.DataFrame:
    t1_xyz = t1_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    t2_xyz = t2_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    rows: List[Dict[str, float]] = []
    for j, p2 in enumerate(t2_xyz):
        dx = t1_xyz[:, 0] - p2[0]
        dy = t1_xyz[:, 1] - p2[1]
        dz = t1_xyz[:, 2] - p2[2]
        dxy = np.hypot(dx, dy)
        valid = (dxy <= gating_xy) & (np.abs(dz) <= gating_z)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        ordered = idx[np.argsort(dxy[idx])]
        for rank, i in enumerate(ordered, 1):
            rows.append(
                {
                    "t1_spine_id": str(t1_df.iloc[i]["id"]),
                    "t2_spine_id": str(t2_df.iloc[j]["id"]),
                    "distance_xy": float(dxy[i]),
                    "distance_z": float(abs(dz[i])),
                    "candidate_rank": float(rank),
                }
            )
    return pd.DataFrame(rows)


def pair_feature_distance(row: pd.Series, t1_by: pd.DataFrame, t2_by: pd.DataFrame) -> float:
    a = t1_by.loc[row["t1_spine_id"]]
    b = t2_by.loc[row["t2_spine_id"]]
    diffs = []
    for col in FEATURE_KEYS:
        av = float(a.get(f"{col}_scaled", np.nan))
        bv = float(b.get(f"{col}_scaled", np.nan))
        d = av - bv if np.isfinite(av) and np.isfinite(bv) else 2.5
        diffs.append(d)
    vec = np.asarray(diffs, dtype=np.float32)
    return float(np.sqrt(np.sum(FEATURE_R_WEIGHTS * np.square(vec))))


def build_feature_weighted_scores(
    candidates: pd.DataFrame,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    gating_xy: float,
    gating_z: float,
    w_xy: float,
    w_z: float,
    w_feat: float,
) -> pd.DataFrame:
    t1_by = t1_df.set_index("id")
    t2_by = t2_df.set_index("id")
    out = candidates.copy()
    out["feat_dist"] = out.apply(pair_feature_distance, axis=1, t1_by=t1_by, t2_by=t2_by)
    xy_n = np.clip(out["distance_xy"] / max(gating_xy, 1e-6), 0, 1.5)
    z_n = np.clip(out["distance_z"] / max(gating_z, 1e-6), 0, 1.5)
    f_n = np.clip(out["feat_dist"] / 3.0, 0, 1.5)
    total_cost = w_xy * xy_n + w_z * z_n + w_feat * f_n
    out["score_feature_weighted"] = 1.0 / (1.0 + total_cost)
    return out


def maybe_load_toolb_model(model_path: Optional[Path]):
    from importlib.util import module_from_spec, spec_from_file_location

    if model_path is None:
        return None, None, None
    toolb_dir = model_path.parent
    matching_fp = toolb_dir / "matching_model.py"
    utils_fp = toolb_dir / "utils.py"
    if not matching_fp.exists() or not utils_fp.exists():
        return None, None, None

    import sys

    if str(toolb_dir) not in sys.path:
        sys.path.insert(0, str(toolb_dir))
    spec = spec_from_file_location("toolb_matching_model", matching_fp)
    if spec is None or spec.loader is None:
        return None, None, None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    model = module.MatchLearningModel(str(toolb_dir / "classifier_model.pkl"))
    return module, model, toolb_dir


def add_toolb_scores(
    candidates: pd.DataFrame,
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    toolb_module,
    toolb_model,
) -> pd.DataFrame:
    out = candidates.copy()
    if toolb_module is None or toolb_model is None:
        out["score_toolb_model"] = np.nan
        return out
    feat_df = toolb_module.build_features(candidates, t1_df, t2_df)
    probs = toolb_model.predict_proba(feat_df)
    prob_map = {
        (str(r["t1_spine_id"]), str(r["t2_spine_id"])): float(p)
        for r, p in zip(feat_df.to_dict(orient="records"), probs)
    }
    out["score_toolb_model"] = out.apply(
        lambda r: prob_map.get((str(r["t1_spine_id"]), str(r["t2_spine_id"])), np.nan), axis=1
    )
    return out


def compute_stability_proxy(candidates: pd.DataFrame, t1_df: pd.DataFrame, t2_df: pd.DataFrame) -> pd.Series:
    t1_missing = t1_df[FEATURE_KEYS].isna().mean(axis=1)
    t2_missing = t2_df[FEATURE_KEYS].isna().mean(axis=1)
    t1_map = dict(zip(t1_df["id"].astype(str), t1_missing.to_numpy(dtype=np.float32)))
    t2_map = dict(zip(t2_df["id"].astype(str), t2_missing.to_numpy(dtype=np.float32)))
    vals = []
    for _, r in candidates.iterrows():
        m1 = float(t1_map.get(str(r["t1_spine_id"]), 1.0))
        m2 = float(t2_map.get(str(r["t2_spine_id"]), 1.0))
        vals.append(max(0.0, 1.0 - 0.5 * (m1 + m2)))
    return pd.Series(vals, index=candidates.index, dtype=np.float32)


def finalize_scores(candidates: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = candidates.copy()
    if mode == "feature_weighted":
        out["final_score"] = out["score_feature_weighted"]
    elif mode == "toolb_model":
        out["final_score"] = out["score_toolb_model"].fillna(out["score_feature_weighted"])
    else:
        s_feat = out["score_feature_weighted"].fillna(0.0)
        s_model = out["score_toolb_model"].fillna(s_feat)
        stability = out["stability_score"].fillna(0.5)
        out["final_score"] = 0.45 * s_feat + 0.40 * s_model + 0.15 * stability
    out["final_score"] = out["final_score"].clip(0.0, 1.0)
    return out


def hungarian_match(scored_pairs: pd.DataFrame, t1_ids: List[str], t2_ids: List[str]) -> Dict[str, str]:
    if scored_pairs.empty:
        return {}
    big_cost = 10.0
    t1_idx = {sid: i for i, sid in enumerate(t1_ids)}
    t2_idx = {sid: j for j, sid in enumerate(t2_ids)}
    mat = np.full((len(t1_ids), len(t2_ids)), big_cost, dtype=np.float32)
    for _, r in scored_pairs.iterrows():
        i = t1_idx.get(str(r["t1_spine_id"]))
        j = t2_idx.get(str(r["t2_spine_id"]))
        if i is None or j is None:
            continue
        mat[i, j] = min(mat[i, j], float(1.0 - r["final_score"]))
    row_ind, col_ind = linear_sum_assignment(mat)
    matches: Dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        if mat[i, j] < 1.0:
            matches[t2_ids[j]] = t1_ids[i]
    return matches


def build_outputs(
    scored_pairs: pd.DataFrame,
    matches: Dict[str, str],
    t1_df: pd.DataFrame,
    t2_df: pd.DataFrame,
    tau_high: float,
    tau_low: float,
    candidate_counts: Optional[Dict[str, int]] = None,
    isolate_autoaccept: bool = False,
    isolate_max_candidates: int = 1,
    isolate_xy_max: float = 25.0,
    isolate_z_max: float = 3.0,
    isolate_pmatch_min: float = 0.60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    best = (
        scored_pairs.sort_values(["t2_spine_id", "final_score"], ascending=[True, False])
        .groupby("t2_spine_id")
        .head(2)
        .copy()
    )
    confidence_rows = []
    for t2_id, grp in best.groupby("t2_spine_id"):
        top = grp.iloc[0]
        second = grp.iloc[1] if len(grp) > 1 else None
        margin = float(top["final_score"] - (float(second["final_score"]) if second is not None else 0.0))
        final_conf = float(np.clip(0.8 * float(top["final_score"]) + 0.2 * margin, 0.0, 1.0))
        match_t1 = matches.get(str(t2_id), "")
        cand_count = int(candidate_counts.get(str(t2_id), 0)) if candidate_counts else 0
        auto_override = False
        if isolate_autoaccept and match_t1:
            auto_override = (
                cand_count <= isolate_max_candidates
                and float(top.get("distance_xy", np.inf)) <= isolate_xy_max
                and float(top.get("distance_z", np.inf)) <= isolate_z_max
                and float(top.get("final_score", 0.0)) >= isolate_pmatch_min
            )
        if match_t1 == "":
            status = "new"
        elif final_conf >= tau_high or auto_override:
            status = "matched_auto"
        elif final_conf >= tau_low:
            status = "review"
        else:
            status = "new"
        confidence_rows.append(
            {
                "t2_spine_id": str(t2_id),
                "t1_spine_id": match_t1,
                "status": status,
                "p_match": float(top["final_score"]),
                "margin": margin,
                "stability_score": float(top.get("stability_score", 0.5)),
                "confidence": final_conf,
                "candidate_count": cand_count,
                "auto_accept_override": bool(auto_override),
                "distance_xy": float(top.get("distance_xy", np.nan)),
                "distance_z": float(top.get("distance_z", np.nan)),
            }
        )
    all_t2 = {str(x) for x in t2_df["id"].astype(str)}
    present_t2 = {str(r["t2_spine_id"]) for r in confidence_rows}
    for t2_id in sorted(all_t2 - present_t2):
        confidence_rows.append(
            {
                "t2_spine_id": t2_id,
                "t1_spine_id": "",
                "status": "new",
                "p_match": 0.0,
                "margin": 0.0,
                "stability_score": 0.0,
                "confidence": 0.0,
                "candidate_count": int(candidate_counts.get(str(t2_id), 0)) if candidate_counts else 0,
                "auto_accept_override": False,
                "distance_xy": np.nan,
                "distance_z": np.nan,
            }
        )

    match_df = pd.DataFrame(confidence_rows).sort_values("t2_spine_id").reset_index(drop=True)
    lost_t1 = set(t1_df["id"].astype(str)) - set(x for x in match_df["t1_spine_id"] if x)
    lost_rows = [{"t1_spine_id": sid, "t2_spine_id": "", "status": "lost"} for sid in sorted(lost_t1)]
    if lost_rows:
        match_df = pd.concat([match_df, pd.DataFrame(lost_rows)], ignore_index=True)
    review_df = match_df[match_df["status"] == "review"].copy()
    return match_df, review_df


def evaluate(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> Dict[str, float]:
    pred_pairs = {
        (str(r["t1_spine_id"]), str(r["t2_spine_id"]))
        for _, r in pred_df.iterrows()
        if str(r["status"]).startswith("matched") and str(r["t1_spine_id"]) and str(r["t2_spine_id"])
    }
    truth_pairs = {
        (str(r["t1_spine_id"]), str(r["t2_spine_id"])) for _, r in truth_df.iterrows() if r["status"] == "matched"
    }
    tp = len(pred_pairs & truth_pairs)
    fp = len(pred_pairs - truth_pairs)
    fn = len(truth_pairs - pred_pairs)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    auto_df = pred_df[pred_df["status"] == "matched_auto"]
    auto_pairs = {(str(r["t1_spine_id"]), str(r["t2_spine_id"])) for _, r in auto_df.iterrows()}
    auto_tp = len(auto_pairs & truth_pairs)
    auto_precision = auto_tp / max(len(auto_pairs), 1)
    auto_coverage = len(auto_pairs) / max(len(pred_df[pred_df["t2_spine_id"] != ""]), 1)
    review_count = int((pred_df["status"] == "review").sum())
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auto_precision": auto_precision,
        "auto_coverage": auto_coverage,
        "review_count": float(review_count),
    }


def write_fp_outputs(pred_df: pd.DataFrame, truth_df: pd.DataFrame, out_dir: Path) -> None:
    pred_pairs = {
        (str(r["t1_spine_id"]), str(r["t2_spine_id"]))
        for _, r in pred_df.iterrows()
        if str(r["status"]).startswith("matched") and str(r["t1_spine_id"]) and str(r["t2_spine_id"])
    }
    truth_pairs = {
        (str(r["t1_spine_id"]), str(r["t2_spine_id"])) for _, r in truth_df.iterrows() if str(r["status"]) == "matched"
    }
    fp_pairs = sorted(pred_pairs - truth_pairs, key=lambda x: (x[1], x[0]))
    fp_df = pd.DataFrame(fp_pairs, columns=["t1_spine_id", "t2_spine_id"])
    fp_df.to_csv(out_dir / "fp_pairs.csv", index=False)

    if fp_df.empty:
        pd.DataFrame().to_csv(out_dir / "fp_matrix.csv", index=False)
        return

    matrix = pd.crosstab(fp_df["t2_spine_id"], fp_df["t1_spine_id"]).astype(int)
    matrix.to_csv(out_dir / "fp_matrix.csv")

    # For readability, keep at most top 40x40 by FP frequency.
    row_order = matrix.sum(axis=1).sort_values(ascending=False).index[:40]
    col_order = matrix.sum(axis=0).sort_values(ascending=False).index[:40]
    mat_plot = matrix.loc[row_order, col_order]

    plt.figure(figsize=(12, 10))
    plt.imshow(mat_plot.values, aspect="auto")
    plt.colorbar(label="FP count")
    plt.title("False Positive Matrix (t2 rows vs t1 cols)")
    plt.xlabel("Predicted t1_spine_id")
    plt.ylabel("t2_spine_id")
    plt.xticks(range(len(mat_plot.columns)), mat_plot.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(mat_plot.index)), mat_plot.index, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "fp_matrix.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid spine tracking experiments")
    p.add_argument("--mode", choices=["feature_weighted", "toolb_model", "hybrid_confidence"], default="hybrid_confidence")
    p.add_argument("--t1_csv", required=True)
    p.add_argument("--t2_csv", required=True)
    p.add_argument("--manual_matches", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gating_xy", type=float, default=30.0)
    p.add_argument("--gating_z", type=float, default=4.0)
    p.add_argument("--tau_high", type=float, default=0.70)
    p.add_argument("--tau_low", type=float, default=0.45)
    p.add_argument("--w_xy", type=float, default=0.35)
    p.add_argument("--w_z", type=float, default=0.20)
    p.add_argument("--w_feat", type=float, default=0.45)
    p.add_argument("--toolb_dir", default="D:/learning_project_spines/scripts/step2-spine tracking/spine_matching_tool")
    p.add_argument("--isolate_autoaccept", action="store_true")
    p.add_argument("--isolate_max_candidates", type=int, default=1)
    p.add_argument("--isolate_xy_max", type=float, default=25.0)
    p.add_argument("--isolate_z_max", type=float, default=3.0)
    p.add_argument("--isolate_pmatch_min", type=float, default=0.60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_df = load_spines(Path(args.t1_csv))
    t2_df = load_spines(Path(args.t2_csv))
    truth_df: Optional[pd.DataFrame] = None
    if str(args.manual_matches).strip():
        truth_df = load_manual_matches(Path(args.manual_matches))

    scalers = robust_scalers(t1_df, t2_df, FEATURE_KEYS)
    t1_scaled = add_scaled_features(t1_df, scalers, FEATURE_KEYS)
    t2_scaled = add_scaled_features(t2_df, scalers, FEATURE_KEYS)

    candidates = build_candidates(t1_scaled, t2_scaled, args.gating_xy, args.gating_z)
    if candidates.empty:
        raise RuntimeError("No candidate pairs found. Increase gating thresholds.")

    scored = build_feature_weighted_scores(
        candidates=candidates,
        t1_df=t1_scaled,
        t2_df=t2_scaled,
        gating_xy=args.gating_xy,
        gating_z=args.gating_z,
        w_xy=args.w_xy,
        w_z=args.w_z,
        w_feat=args.w_feat,
    )

    toolb_module = None
    toolb_model = None
    if args.mode in ("toolb_model", "hybrid_confidence"):
        toolb_module, toolb_model, _ = maybe_load_toolb_model(Path(args.toolb_dir))
    scored = add_toolb_scores(scored, t1_scaled, t2_scaled, toolb_module, toolb_model)
    scored["stability_score"] = compute_stability_proxy(scored, t1_scaled, t2_scaled)
    scored = finalize_scores(scored, args.mode)

    t1_ids = sorted(t1_scaled["id"].astype(str).unique().tolist())
    t2_ids = sorted(t2_scaled["id"].astype(str).unique().tolist())
    matches = hungarian_match(scored, t1_ids=t1_ids, t2_ids=t2_ids)
    candidate_counts = scored.groupby("t2_spine_id").size().to_dict()
    matches_df, review_df = build_outputs(
        scored_pairs=scored,
        matches=matches,
        t1_df=t1_scaled,
        t2_df=t2_scaled,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        candidate_counts=candidate_counts,
        isolate_autoaccept=args.isolate_autoaccept,
        isolate_max_candidates=args.isolate_max_candidates,
        isolate_xy_max=args.isolate_xy_max,
        isolate_z_max=args.isolate_z_max,
        isolate_pmatch_min=args.isolate_pmatch_min,
    )

    metrics = {
        "mode": args.mode,
        "num_candidates": float(len(scored)),
        "num_matches": float(sum(1 for _, r in matches_df.iterrows() if str(r["status"]).startswith("matched"))),
    }
    if truth_df is not None:
        metrics.update(evaluate(matches_df, truth_df))
        write_fp_outputs(matches_df, truth_df, out_dir)

    scored.sort_values(["t2_spine_id", "final_score"], ascending=[True, False]).to_csv(out_dir / "pair_scores.csv", index=False)
    matches_df.to_csv(out_dir / "matches.csv", index=False)
    review_df.to_csv(out_dir / "review_queue.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved results to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
