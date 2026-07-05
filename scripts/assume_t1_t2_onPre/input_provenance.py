"""
Validate inference export spine IDs against original FOV detected_spines tables.

Checks that IDs in matched/new/lost/unresolved_manual_review exist in the canonical
respan/<timepoint>/Tables/fovN_detected_spines*.csv for each side (coordinates come
from those tables at review time). Does not compare CSV coordinates to TIFF dimensions.

Used automatically by infer_baseline_bridged_pairs.py and the transitive review app load.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None  # type: ignore

# Load-time sanity check: pass if at least this fraction of IDs / coords match FOV tables.
MIN_SANITY_MATCH_FRAC = 0.60


@dataclass
class ProvenanceResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    t1_csv_path: Optional[Path] = None
    t2_csv_path: Optional[Path] = None
    t1_tiff_path: Optional[Path] = None
    t2_tiff_path: Optional[Path] = None
    t1_spine_count: int = 0
    t2_spine_count: int = 0
    matched_ids_checked: int = 0
    new_ids_checked: int = 0
    lost_ids_checked: int = 0
    unresolved_ids_checked: int = 0
    baseline_ids_checked: int = 0

    def message(self) -> str:
        parts = list(self.errors)
        if self.warnings:
            parts.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(parts)


def _canonical_spine_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        return str(int(float(s)))
    return s


# Comparison folder labels (spaces) -> canonical respan folder names (hyphens).
_COMPARISON_DISPLAY_TO_PAIR: dict[str, tuple[str, str]] = {
    "end droplet - return to droplet": ("end-droplet", "return to droplet"),
    "end droplet - end lever": ("end-droplet", "end-lever"),
}


def normalize_timepoint_key(label: str) -> str:
    """Case-insensitive key: ignore spaces, hyphens, underscores."""
    return re.sub(r"[\s_\-]+", "", label.strip().lower())


def list_respan_timepoint_dirs(respan_root: Path) -> List[Path]:
    if not respan_root.is_dir():
        return []
    return sorted(
        (p.resolve() for p in respan_root.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )


def resolve_timepoint_dir(respan_root: Path, timepoint: str) -> Path:
    """
    Map a timepoint label to the actual folder under respan/ (case-insensitive;
    spaces vs hyphens equivalent, e.g. 'end droplet' -> 'end-droplet').
    """
    label = timepoint.strip()
    if not label:
        raise ValueError("Empty timepoint label")

    # Exact folder name (case-insensitive).
    for child in list_respan_timepoint_dirs(respan_root):
        if child.name.lower() == label.lower():
            return child

    want = normalize_timepoint_key(label)
    matches = [
        c for c in list_respan_timepoint_dirs(respan_root) if normalize_timepoint_key(c.name) == want
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"Ambiguous timepoint {timepoint!r} under {respan_root} (matches: {names})"
        )

    available = ", ".join(p.name for p in list_respan_timepoint_dirs(respan_root))
    raise FileNotFoundError(
        f"No respan timepoint folder for {timepoint!r} under {respan_root}. "
        f"Available: {available or '(none)'}"
    )


def canonical_timepoint_name(respan_root: Path, timepoint: str) -> str:
    """Return the on-disk folder name for a timepoint label."""
    return resolve_timepoint_dir(respan_root, timepoint).name


def parse_comparison_timepoints(comparison: str) -> Tuple[str, str]:
    """
    Parse results/fovN/<comparison>/ folder name into canonical respan timepoint names.
    """
    comp = comparison.strip()
    if not comp:
        return "", ""

    low = comp.lower()
    for display, pair in _COMPARISON_DISPLAY_TO_PAIR.items():
        if low == display.lower():
            return pair

    if " - " in comp:
        left, right = comp.split(" - ", 1)
        return left.strip(), right.strip()

    return "", ""


def find_spine_csv(respan_root: Path, timepoint: str, fov: int) -> Path:
    """
    Locate fovN/FOVN_detected_spines*.csv under respan/<timepoint>/Tables/.
    Timepoint folder and CSV filename matching are case-insensitive; spaces/hyphens
    in the timepoint label are equivalent.
    """
    tp_dir = resolve_timepoint_dir(respan_root, timepoint)
    tables = tp_dir / "Tables"
    if not tables.is_dir():
        raise FileNotFoundError(f"Tables folder not found: {tables}")
    pat = re.compile(rf"^fov{fov}.*detected_spines.*\.csv$", re.IGNORECASE)
    matches = sorted(p for p in tables.glob("*.csv") if pat.search(p.name))
    if not matches:
        raise FileNotFoundError(
            f"No detected_spines CSV for timepoint={timepoint!r} (folder {tp_dir.name}) "
            f"fov={fov} under {tables}"
        )
    return matches[0].resolve()


def load_spine_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "spine_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"spine_id": "id"})
    if "id" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing spine id column (id or spine_id)")
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise ValueError(f"{csv_path.name}: missing required column {col!r}")
    work = df.copy()
    work["id"] = work["id"].apply(_canonical_spine_id)
    work = work[work["id"].astype(str).str.len() > 0]
    for col in ("x", "y", "z"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["x", "y", "z"])
    return work.reset_index(drop=True)


def spine_id_set(df: pd.DataFrame) -> Set[str]:
    return set(df["id"].astype(str).tolist())


def tiff_shape_zyx(tiff_path: Path) -> Tuple[int, int, int]:
    if tifffile is None:
        raise ImportError("tifffile is required for provenance validation (pip install tifffile)")
    with tifffile.TiffFile(str(tiff_path)) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) == 3:
        return shape[0], shape[1], shape[2]
    raise ValueError(f"Unexpected TIFF shape {shape} for {tiff_path}")


def pick_tiff_pair(input_dir: Path) -> Tuple[Path, Path]:
    tifs = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")),
        key=lambda p: p.name.lower(),
    )
    t1 = [p for p in tifs if p.name.lower().startswith("t1_")]
    t2 = [p for p in tifs if p.name.lower().startswith("t2_")]
    if len(t1) == 1 and len(t2) == 1:
        return t1[0], t2[0]
    if len(tifs) >= 2:
        return tifs[0], tifs[1]
    raise FileNotFoundError(f"Need two TIFF stacks in {input_dir}")


def stage_session_inputs(
    export_dir: Path,
    *,
    t1_csv_src: Path,
    t2_csv_src: Path,
) -> Tuple[Path, Path]:
    """Copy canonical spine tables next to the export for review load."""
    dest_dir = export_dir / "session_inputs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    t1_dest = dest_dir / "t1_detected_spines.csv"
    t2_dest = dest_dir / "t2_detected_spines.csv"
    shutil.copy2(t1_csv_src, t1_dest)
    shutil.copy2(t2_csv_src, t2_dest)
    return t1_dest.resolve(), t2_dest.resolve()


def _read_id_column(path: Path, *cols: str) -> List[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    for col in cols:
        if col in df.columns:
            return [_canonical_spine_id(x) for x in df[col].dropna() if _canonical_spine_id(x)]
    if len(df.columns) == 1:
        return [_canonical_spine_id(x) for x in df[df.columns[0]].dropna() if _canonical_spine_id(x)]
    return []


def _ids_missing(
    ids: Iterable[str],
    allowed: Set[str],
    *,
    label: str,
    limit: int = 5,
    min_match_frac: float = MIN_SANITY_MATCH_FRAC,
) -> Optional[str]:
    checked = sorted(
        {i for i in (_canonical_spine_id(x) for x in ids) if i and not i.startswith("manual")}
    )
    if not checked:
        return None
    missing = [i for i in checked if i not in allowed]
    if not missing:
        return None
    match_frac = (len(checked) - len(missing)) / len(checked)
    if match_frac >= min_match_frac:
        return None
    sample = ", ".join(missing[:limit])
    more = f" (+{len(missing) - limit} more)" if len(missing) > limit else ""
    return (
        f"{label}: {len(missing)}/{len(checked)} spine ID(s) not in CSV "
        f"({match_frac:.0%} match, need {min_match_frac:.0%}; e.g. {sample}{more})"
    )


def _ids_missing_warning(
    ids: Iterable[str],
    allowed: Set[str],
    *,
    label: str,
    min_match_frac: float = MIN_SANITY_MATCH_FRAC,
) -> Optional[str]:
    """Warning (non-fatal) when some IDs missing but match fraction still passes."""
    checked = sorted(
        {i for i in (_canonical_spine_id(x) for x in ids) if i and not i.startswith("manual")}
    )
    if not checked:
        return None
    missing = [i for i in checked if i not in allowed]
    if not missing:
        return None
    match_frac = (len(checked) - len(missing)) / len(checked)
    if match_frac < min_match_frac:
        return None
    return (
        f"{label}: {len(missing)}/{len(checked)} ID(s) not in CSV but "
        f"{match_frac:.0%} match meets {min_match_frac:.0%} sanity threshold"
    )


def _check_inference_coords_vs_fov(
    inference_df: pd.DataFrame,
    fov_df: pd.DataFrame,
    id_col: str,
    *,
    label: str,
    tol: float = 0.01,
    limit: int = 5,
    min_match_frac: float = MIN_SANITY_MATCH_FRAC,
) -> Optional[str]:
    """
    If inference rows include x/y/z, verify they match the original FOV table per spine ID.
    """
    coord_cols = [c for c in ("x", "y", "z") if c in inference_df.columns]
    if not coord_cols or id_col not in inference_df.columns:
        return None

    lookup = fov_df.set_index("id")[coord_cols]
    mismatches: List[str] = []
    checked = 0
    bad = 0
    for _, row in inference_df.iterrows():
        sid = _canonical_spine_id(row.get(id_col))
        if not sid or sid.startswith("manual"):
            continue
        if sid not in lookup.index:
            continue
        checked += 1
        ref = lookup.loc[sid]
        row_bad = False
        for col in coord_cols:
            inf_val = pd.to_numeric(row.get(col), errors="coerce")
            ref_val = float(ref[col])
            if pd.isna(inf_val):
                continue
            if abs(float(inf_val) - ref_val) > tol:
                row_bad = True
                if len(mismatches) < limit:
                    mismatches.append(
                        f"{sid} {col}: inference={float(inf_val):.3g} vs FOV={ref_val:.3g}"
                    )
                break
        if row_bad:
            bad += 1
        if len(mismatches) >= limit and bad > 0:
            continue

    if checked == 0 or bad == 0:
        return None
    match_frac = (checked - bad) / checked
    if match_frac >= min_match_frac:
        return None
    sample = "; ".join(mismatches[:limit])
    more = f" (+more)" if bad > limit else ""
    return (
        f"{label}: coordinate mismatch vs FOV table ({bad}/{checked} rows, "
        f"{match_frac:.0%} match, need {min_match_frac:.0%}; {sample}{more})"
    )


def _actionable_unresolved_baseline_ids(udf: pd.DataFrame) -> List[str]:
    """
    Baseline IDs that must exist in the FOV table — excludes QC-only unresolved rows
    (e.g. baseline_ignored_or_removed with no T1/T2 anchor to review).
    """
    out: List[str] = []
    for _, row in udf.iterrows():
        reason = str(row.get("reason", "") or "").lower()
        t1 = _canonical_spine_id(row.get("t1_spine_id", ""))
        t2 = _canonical_spine_id(row.get("t2_spine_id", ""))
        if reason == "baseline_ignored_or_removed" and not t1 and not t2:
            continue
        bid = _canonical_spine_id(row.get("baseline_spine_id", ""))
        if bid:
            out.append(bid)
    return out


def _record_id_check(
    result: ProvenanceResult,
    ids: Iterable[str],
    allowed: Set[str],
    *,
    label: str,
) -> None:
    msg = _ids_missing(ids, allowed, label=label)
    if msg:
        result.errors.append(msg)
        return
    warn = _ids_missing_warning(ids, allowed, label=label)
    if warn:
        result.warnings.append(warn)


def validate_input_provenance(
    *,
    export_dir: Path,
    respan_root: Path,
    fov: int,
    t1_timepoint: str,
    t2_timepoint: str,
    t1_tiff: Optional[Path] = None,
    t2_tiff: Optional[Path] = None,
    t1_csv: Optional[Path] = None,
    t2_csv: Optional[Path] = None,
    baseline_timepoint: Optional[str] = None,
) -> ProvenanceResult:
    """
    Verify inference export spine IDs (and optional x/y/z columns) against original
    FOV detected_spines tables for T1 and T2 timepoints.
    """
    result = ProvenanceResult(ok=True)
    if t1_tiff is not None and t1_tiff.is_file():
        result.t1_tiff_path = t1_tiff.resolve()
    if t2_tiff is not None and t2_tiff.is_file():
        result.t2_tiff_path = t2_tiff.resolve()

    try:
        if t1_csv is None:
            t1_csv = find_spine_csv(respan_root, t1_timepoint, fov)
        if t2_csv is None:
            t2_csv = find_spine_csv(respan_root, t2_timepoint, fov)
        result.t1_csv_path = t1_csv.resolve()
        result.t2_csv_path = t2_csv.resolve()
    except (FileNotFoundError, ValueError) as exc:
        result.errors.append(str(exc))
        result.ok = False
        return result

    try:
        t1_df = load_spine_table(t1_csv)
        t2_df = load_spine_table(t2_csv)
    except ValueError as exc:
        result.errors.append(str(exc))
        result.ok = False
        return result

    result.t1_spine_count = len(t1_df)
    result.t2_spine_count = len(t2_df)
    t1_ids = spine_id_set(t1_df)
    t2_ids = spine_id_set(t2_df)

    baseline_ids: Optional[Set[str]] = None
    if baseline_timepoint:
        try:
            baseline_csv = find_spine_csv(respan_root, baseline_timepoint, fov)
            baseline_ids = spine_id_set(load_spine_table(baseline_csv))
        except (FileNotFoundError, ValueError) as exc:
            result.warnings.append(f"baseline spine table not checked: {exc}")

    matched_path = export_dir / "matched.csv"
    if matched_path.is_file():
        mdf = pd.read_csv(matched_path)
        t1_col = "t1_spine_id" if "t1_spine_id" in mdf.columns else None
        t2_col = "t2_spine_id" if "t2_spine_id" in mdf.columns else None
        if t1_col and t2_col:
            result.matched_ids_checked = len(mdf)
            _record_id_check(result, mdf[t1_col], t1_ids, label="matched.csv T1 IDs")
            _record_id_check(result, mdf[t2_col], t2_ids, label="matched.csv T2 IDs")
            if "baseline_spine_id" in mdf.columns and baseline_ids is not None:
                b_ids = mdf["baseline_spine_id"]
                result.baseline_ids_checked = int(b_ids.notna().sum())
                _record_id_check(result, b_ids, baseline_ids, label="matched.csv baseline IDs")
            msg = _check_inference_coords_vs_fov(mdf, t1_df, t1_col, label=f"matched.csv T1 vs {t1_csv.name}")
            if msg:
                result.errors.append(msg)
            msg = _check_inference_coords_vs_fov(mdf, t2_df, t2_col, label=f"matched.csv T2 vs {t2_csv.name}")
            if msg:
                result.errors.append(msg)

    unresolved_path = export_dir / "unresolved_manual_review.csv"
    if unresolved_path.is_file():
        udf = pd.read_csv(unresolved_path)
        result.unresolved_ids_checked = len(udf)
        if "t1_spine_id" in udf.columns:
            _record_id_check(
                result,
                udf["t1_spine_id"],
                t1_ids,
                label=f"unresolved_manual_review.csv T1 IDs vs {t1_csv.name}",
            )
            msg = _check_inference_coords_vs_fov(
                udf, t1_df, "t1_spine_id", label=f"unresolved T1 coords vs {t1_csv.name}"
            )
            if msg:
                result.errors.append(msg)
        if "t2_spine_id" in udf.columns:
            _record_id_check(
                result,
                udf["t2_spine_id"],
                t2_ids,
                label=f"unresolved_manual_review.csv T2 IDs vs {t2_csv.name}",
            )
            msg = _check_inference_coords_vs_fov(
                udf, t2_df, "t2_spine_id", label=f"unresolved T2 coords vs {t2_csv.name}"
            )
            if msg:
                result.errors.append(msg)
        if "baseline_spine_id" in udf.columns and baseline_ids is not None:
            actionable = _actionable_unresolved_baseline_ids(udf)
            result.baseline_ids_checked += len(actionable)
            if actionable:
                _record_id_check(
                    result,
                    actionable,
                    baseline_ids,
                    label="unresolved_manual_review.csv baseline IDs",
                )

    new_path = export_dir / "new.csv"
    if new_path.is_file():
        new_ids = _read_id_column(new_path, "t2_spine_id", "spine_id")
        result.new_ids_checked = len(new_ids)
        _record_id_check(result, new_ids, t2_ids, label="new.csv T2 IDs")

    lost_path = export_dir / "lost.csv"
    if lost_path.is_file():
        lost_ids = _read_id_column(lost_path, "t1_spine_id", "spine_id")
        result.lost_ids_checked = len(lost_ids)
        _record_id_check(result, lost_ids, t1_ids, label="lost.csv T1 IDs")

    if result.errors:
        result.ok = False
    return result


def _baseline_timepoint_from_export(export_dir: Path) -> Optional[str]:
    meta_path = export_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = meta.get("baseline_timepoint")
    return str(val).strip() if val else None


def validate_or_raise(**kwargs) -> ProvenanceResult:
    """Run validation; copy CSVs to session_inputs; raise ValueError on failure."""
    export_dir: Path = kwargs["export_dir"]
    respan_root: Path = kwargs["respan_root"]
    fov: int = kwargs["fov"]
    t1_timepoint: str = kwargs["t1_timepoint"]
    t2_timepoint: str = kwargs["t2_timepoint"]
    input_dir: Path = kwargs.get("input_dir") or (export_dir / "input_files")
    baseline_timepoint: Optional[str] = kwargs.get("baseline_timepoint")
    if baseline_timepoint is None:
        baseline_timepoint = _baseline_timepoint_from_export(export_dir)

    t1_tiff: Optional[Path] = None
    t2_tiff: Optional[Path] = None
    if input_dir.is_dir():
        try:
            t1_tiff, t2_tiff = pick_tiff_pair(input_dir)
        except FileNotFoundError:
            pass

    t1_csv = find_spine_csv(respan_root, t1_timepoint, fov)
    t2_csv = find_spine_csv(respan_root, t2_timepoint, fov)

    result = validate_input_provenance(
        export_dir=export_dir,
        respan_root=respan_root,
        fov=fov,
        t1_timepoint=t1_timepoint,
        t2_timepoint=t2_timepoint,
        t1_tiff=t1_tiff,
        t2_tiff=t2_tiff,
        t1_csv=t1_csv,
        t2_csv=t2_csv,
        baseline_timepoint=baseline_timepoint,
    )

    if result.ok:
        stage_session_inputs(export_dir, t1_csv_src=t1_csv, t2_csv_src=t2_csv)
    else:
        fail_path = export_dir / "validation_failed.txt"
        fail_path.write_text(result.message() + "\n", encoding="utf-8")

    if not result.ok:
        raise ValueError(
            "Input provenance check failed — inference spine IDs do not match the "
            "original FOV detected_spines tables (matched/new/lost/unresolved). "
            "See validation_failed.txt in the export folder.\n"
            + result.message()
        )

    return result
