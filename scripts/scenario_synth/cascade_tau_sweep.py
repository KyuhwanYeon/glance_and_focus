"""Simulate cascade at multiple anomaly thresholds without re-running VLM.

Reuses the always_on VLM dump (which has VLM areas for every frame). For
each candidate threshold τ, simulates cascade by:
  - Frames with anomaly_score > τ: use always_on's areas (gate fired)
  - Frames with anomaly_score ≤ τ: empty areas (gate skipped → no rerank)
Then applies the same exclude rerank + collision check as collision_analysis.

Outputs:
  exp/collision/tau_sweep.csv          (tau, n_calls, collision metrics)
  exp/collision/tau_sweep.png          trade-off plot

Run in navsim env:
    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/cascade_tau_sweep.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MplPath

sys.path.insert(0, "/workspace/CoachDrive/scripts/counterfactual_evaluation")
sys.path.insert(0, "/workspace/CoachDrive/scripts/scenario_synth")
from topk_helper import combined_score                                  # noqa: E402
from collision_analysis import (                                        # noqa: E402
    CASE_BODY_LW_M, hazard_world_pose, hazard_in_frame_ego,
    waypoints_in_obb, vlm_rerank_idx,
)

MANIFEST = "/workspace/CoachDrive/scripts/scenario_synth/benchmark_manifest.yaml"
GTRS_PKLS_ROOT = Path("/workspace/CoachDrive/exp/benchmark_pkls")
ALWAYS_ON_ROOT = Path("/workspace/CoachDrive/exp/vlm_dump/always_on")
CLASSIFIED_CSV = "/workspace/CoachDrive/exp/anomaly_classify/classified.csv"
VOCAB_PATH = "/workspace/CoachDrive/traj_final/8192.npy"
OUT_DIR = Path("/workspace/CoachDrive/exp/collision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.05,0.06,0.07,0.0844,0.10,0.12",
                    help="Comma-separated τ values to sweep.")
    ap.add_argument("--risk_threshold", type=float, default=0.5)
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    print(f"[sweep] τ values: {taus}")

    manifest = yaml.safe_load(open(MANIFEST))
    log_dir = Path(manifest["defaults"]["log_dir"])
    behaviors = {b["id"]: b for b in manifest["behaviors"]}
    classified = pd.read_csv(CLASSIFIED_CSV)
    classified = classified.set_index(["scenario", "variant", "frame_idx"], drop=False)

    vocab_xy = np.load(VOCAB_PATH)[..., :2].astype(np.float32)

    # Pre-load all GTRS + always_on VLM data to avoid re-loading per τ
    print("[sweep] preloading GTRS + always_on VLM ...")
    gtrs_db: dict[tuple, dict] = {}
    vlm_db: dict[tuple, dict] = {}
    haz_meta: dict[tuple, tuple] = {}    # (sc, var) -> (haz_world, lw, case_id)
    log_cache: dict[str, list] = {}
    for sc in manifest["scenarios"]:
        log = log_cache.setdefault(
            sc["id"], pickle.load(open(log_dir / sc["log"], "rb")))
        end_frame = log[sc["end_frame"]]
        variants = ["clean"] + [
            f"{c['id']}__{b['id']}" for c in manifest["cases"] for b in manifest["behaviors"]
        ]
        for variant in variants:
            gp = GTRS_PKLS_ROOT / sc["id"] / f"{variant}.pkl"
            vp = ALWAYS_ON_ROOT / sc["id"] / f"{variant}.pkl"
            if not gp.exists() or not vp.exists():
                continue
            gtrs_db[(sc["id"], variant)] = pickle.load(open(gp, "rb"))
            vlm_db[(sc["id"], variant)] = pickle.load(open(vp, "rb"))
            if variant != "clean":
                case_id, beh_id = variant.split("__")
                lw = CASE_BODY_LW_M.get(case_id)
                if lw is None:
                    continue
                beh = behaviors[beh_id]
                haz_world = hazard_world_pose(end_frame,
                                              float(beh["end_x"]), float(beh["end_y"]))
                haz_meta[(sc["id"], variant)] = (haz_world, lw, case_id)
    print(f"[sweep] loaded {len(gtrs_db)} (sc, var) pairs")

    rows = []
    t0 = time.time()
    for tau in taus:
        n_calls = 0
        per_var_collide = {}    # (sc, var) -> {ever, n_frames, n_col}
        for (sc, var), gtrs in gtrs_db.items():
            if (sc, var) not in haz_meta:
                continue
            haz_world, lw, case_id = haz_meta[(sc, var)]
            sc_meta = next(s for s in manifest["scenarios"] if s["id"] == sc)
            log = log_cache[sc]
            ever = 0
            n_frames = 0
            n_col = 0
            for fi in range(sc_meta["start_frame"], sc_meta["end_frame"] + 1):
                token = log[fi]["token"]
                if token not in gtrs or "imi" not in gtrs[token]:
                    continue
                key3 = (sc, var, fi)
                if key3 not in classified.index:
                    continue
                ascore = float(classified.loc[key3, "anomaly_score"])
                gate_open = ascore > tau
                areas = []
                if gate_open and token in vlm_db[(sc, var)]:
                    areas = vlm_db[(sc, var)][token].get("areas", [])
                    n_calls += 1
                comb = combined_score(gtrs[token]).astype(np.float32)
                idx = vlm_rerank_idx(comb, vocab_xy, areas, args.risk_threshold)
                # Collision check
                wp = vocab_xy[idx]
                e2g = np.asarray(log[fi]["ego2global"], dtype=np.float64)
                hx, hy, hyaw = hazard_in_frame_ego(haz_world, e2g)
                in_obb = waypoints_in_obb(wp, hx, hy, hyaw, lw[0], lw[1])
                if int(in_obb.sum()) > 0:
                    n_col += 1
                    ever = 1
                n_frames += 1
            per_var_collide[(sc, var)] = (ever, n_frames, n_col)
        n_total_frames = sum(v[1] for v in per_var_collide.values())
        n_total_col = sum(v[2] for v in per_var_collide.values())
        n_ever = sum(v[0] for v in per_var_collide.values())
        n_var = len(per_var_collide)
        rows.append({
            "tau": tau,
            "vlm_calls": n_calls,
            "vlm_call_rate": n_calls / max(1, sum(
                (sc_meta["end_frame"] - sc_meta["start_frame"] + 1)
                * (1 + len(manifest["cases"]) * len(manifest["behaviors"]))
                for sc_meta in manifest["scenarios"])),
            "frame_collision_rate": n_total_col / max(1, n_total_frames),
            "variant_collision_rate": n_ever / max(1, n_var),
            "n_variants": n_var,
            "n_variants_collided": n_ever,
        })
        print(f"  τ={tau:.4f}  calls={n_calls:5d}  "
              f"frame_col={rows[-1]['frame_collision_rate']:6.2%}  "
              f"variant_col={rows[-1]['variant_collision_rate']:6.2%}  "
              f"({(time.time()-t0)/60:.1f}min)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tau_sweep.csv", index=False)
    print(f"[sweep] wrote {OUT_DIR / 'tau_sweep.csv'}")

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(df.tau, df.vlm_calls, marker="o", color="#3a8ad6")
    ax[0].set_xlabel("τ (anomaly threshold)")
    ax[0].set_ylabel("# VLM calls (lower = cheaper)")
    ax[0].set_title("Cost vs threshold")
    ax[0].grid(alpha=0.3)
    for _, r in df.iterrows():
        ax[0].annotate(f"{int(r.vlm_calls)}", (r.tau, r.vlm_calls), fontsize=8)

    ax[1].plot(df.vlm_calls, df.variant_collision_rate, marker="o",
               color="#d35a3a", label="variant ever-collided")
    ax[1].plot(df.vlm_calls, df.frame_collision_rate, marker="s",
               color="#3a8ad6", label="frame collision")
    ax[1].set_xlabel("# VLM calls")
    ax[1].set_ylabel("collision rate")
    ax[1].set_title("Collision vs cost trade-off (each point = a τ)")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    for _, r in df.iterrows():
        ax[1].annotate(f"τ={r.tau:.3f}", (r.vlm_calls, r.variant_collision_rate),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tau_sweep.png", dpi=120)
    plt.close(fig)
    print(f"[sweep] wrote {OUT_DIR / 'tau_sweep.png'}")


if __name__ == "__main__":
    main()
