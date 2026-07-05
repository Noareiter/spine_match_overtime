#!/usr/bin/env python3
"""Headless smoke test: demo workspace -> spine decisions -> print registry CSV."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

os.environ["SPINE_CONFIG"] = str(ROOT / "config" / "annotator_demo.json")

from fastapi.testclient import TestClient  # noqa: E402

from spine_annotator_backend.app import app  # noqa: E402

BASE = "/mtp/viewer"
REGISTRY = (
    ROOT
    / "results"
    / "demo_spine_logic_lab"
    / "respan"
    / "_annotator"
    / "fov1"
    / "spine_registry_wide.csv"
)
TPS = ("pre-droplet", "end-droplet", "end-lever")


def post(client: TestClient, path: str, **kwargs):
    r = client.post(f"{BASE}{path}", **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text}")
    return r.json()


def get(client: TestClient, path: str):
    r = client.get(f"{BASE}{path}")
    if r.status_code >= 400:
        raise RuntimeError(f"{path} -> {r.status_code}: {r.text}")
    return r.json()


def select(client: TestClient, spine_id: str) -> dict:
    return post(client, "/select-spine", json={"t1_spine_id": str(spine_id)})


def save_next(client: TestClient) -> dict:
    return post(client, "/confirm-lineage")


def read_registry() -> list[dict]:
    with REGISTRY.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def row_for_key(rows: list[dict], lineage_key: str) -> dict:
    for row in rows:
        if row.get("lineage_key") == lineage_key:
            return row
    raise KeyError(f"No registry row for {lineage_key}")


def main() -> None:
    print("Demo CSV smoke test\n")
    with TestClient(app) as client:
        load = post(client, "/load?fov=1")
        print(f"Loaded {load['animal_id']} - {load['t1_spine_count']} spines - {load['timepoint_names']}")

        spines = get(client, "/t1-spines?limit=10")["items"]
        ids = [s["spine_id"] for s in spines[:4]]
        if len(ids) < 4:
            raise SystemExit("Need at least 4 demo spines.")

        # 1) Matched lineage - save as-is
        select(client, ids[0])
        save_next(client)
        print(f"  {ids[0]}: matched all TPs -> saved")

        # 2) Lineage Tab->I: ignore all
        select(client, ids[1])
        post(client, "/clear-all-tps")
        post(client, "/set-fate-all-tps", json={"fate": "ignore", "review_mode": "lineage"})
        save_next(client)
        print(f"  {ids[1]}: lineage ignore-all -> saved")

        # 3) Timepoint I at anchor only; later TPs keep auto-matches until save
        res = select(client, ids[2])
        post(client, "/review-mode", json={"mode": "timepoint"})
        anchor = get(client, "/review-mode")["anchor_timepoint"]
        post(
            client,
            "/set-fate",
            json={"timepoint": anchor, "fate": "ignore", "review_mode": "timepoint"},
        )
        state = get(client, "/state")
        pos = state["positions"]
        if pos["end-droplet"].get("spine_id") and pos["end-lever"].get("spine_id"):
            print(f"  {ids[2]}: timepoint I at {anchor}; later TPs still matched (OK)")
        else:
            raise RuntimeError("Later TPs should stay matched after anchor-only ignore")
        save_next(client)
        print(f"  {ids[2]}: saved")

        # 4) Lineage: matched T1+T2, clear T3 -> lost after last seen (T2)
        select(client, ids[3])
        post(client, "/clear-tp", json={"timepoint": "end-lever"})
        save_next(client)
        print(f"  {ids[3]}: clear end-lever only -> saved")

    if not REGISTRY.is_file():
        raise SystemExit(f"Registry not written: {REGISTRY}")

    rows = read_registry()
    r0 = row_for_key(rows, ids[0])
    assert r0["last_seen_tp"] == "end-lever", r0
    assert r0["status_pre-droplet"] == "matched"

    r2 = row_for_key(rows, ids[2])
    assert r2["fate_pre-droplet"] == "ignore", r2
    assert r2["status_end-droplet"] == "matched", r2
    assert r2["status_end-lever"] == "matched", r2

    r3 = row_for_key(rows, ids[3])
    assert r3["last_seen_tp"] == "end-droplet", r3
    assert "lost_at_end-droplet" in (r3.get("event_end-droplet") or ""), r3

    print(f"\nRegistry: {REGISTRY.relative_to(ROOT)}\n")
    print(REGISTRY.read_text(encoding="utf-8"))
    print("All assertions passed.")


if __name__ == "__main__":
    main()
