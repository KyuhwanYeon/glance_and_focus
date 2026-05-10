"""Stage 4: collision-rate analysis across policies.

For every (scenario, variant, frame), determines the trajectory the planner
would have chosen under three policies:

  baseline_gtrs   GTRS-Dense alone, top-1 of combined_score over the 8192 vocab
  always_on      GTRS + VLM rerank using the always_on dump
                 (every frame queried)
  cascade        GTRS + VLM rerank using the cascade dump
                 (only frames with anomaly_score > τ queried)

Then checks whether the chosen trajectory collides with the hazard:
  - hazard's world pose is fixed (placed in the END frame's ego coords from
    benchmark_manifest.yaml then converted to global once)
  - chosen trajectory's 40 ego-frame waypoints are projected to world via the
    current frame's ego2global
  - if any waypoint enters the hazard's oriented bounding box (length × width
    from CASE_BODY_LW_M, with a small safety margin) → collision flag

Outputs (under exp/collision/):
  per_frame.csv          one row per (sc, var, frame, policy)
  per_variant.csv        one row per (sc, var, policy)
  summary.txt            policy-level totals: collision rate, VLM calls, tokens
  per_policy_bar.png     collision rate × policy comparison

Usage (navsim env, after both vlm_dump policies completed):

    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/collision_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MplPath

sys.path.insert(0, "/workspace/CoachDrive/scripts/counterfactual_evaluation")
from topk_helper import combined_score                                  # noqa: E402

MANIFEST = "/workspace/CoachDrive/scripts/scenario_synth/benchmark_manifest.yaml"
GTRS_PKLS_ROOT = Path("/workspace/CoachDrive/exp/benchmark_pkls")
VLM_DUMP_ROOT = Path("/workspace/CoachDrive/exp/vlm_dump")
VOCAB_PATH = "/workspace/CoachDrive/traj_final/8192.npy"
DEFAULT_OUT_ROOT = Path("/workspace/CoachDrive/exp/collision")

# Hazard footprint (length, width) m — copied from viz_benchmark.py to avoid
# importing scenario_synth modules that drag in nuplan deps.
CASE_BODY_LW_M = {
    "pedestrian_human":       (0.6, 0.6),
    "pedestrian_duck_mascot": (0.7, 0.7),
    "animal_bear":            (0.84, 0.62),
    "animal_wolf":            (0.56, 0.96),
    "construction_cone":      (6.38, 0.39),    # cone_cluster3: 3 cones spread +/-3m forward + cone diameter
    "construction_dumpster":  (2.13, 3.09),
    "vehicle_wrecked":        (4.7, 1.85),   # burning_suv (SUV size)
    "vehicle_normal":         (4.2, 1.8),
}
SAFETY_MARGIN_M = 0.5    # collision = waypoint inside (L+1) × (W+1) box
RISK_THRESHOLD = 0.5


# ── Hazard pose helpers ───────────────────────────────────────────────

def hazard_world_pose(end_frame: dict, end_x: float, end_y: float) -> dict:
    """Return hazard {x_w, y_w, z_w, yaw_w} in world frame, derived from
    end-frame ego coords + face_ego yaw. z_w must be kept — dropping it makes
    the world→ego inverse leak the world-z column (~615 m) into ego x/y."""
    end_e2g = np.asarray(end_frame["ego2global"], dtype=np.float64)
    p_world = end_e2g @ np.array([end_x, end_y, 0.0, 1.0])
    end_yaw_ego = float(np.arctan2(-end_y, -end_x))
    end_heading = float(np.arctan2(end_e2g[1, 0], end_e2g[0, 0]))
    yaw_world = end_heading + end_yaw_ego
    return {"x_w": float(p_world[0]), "y_w": float(p_world[1]),
            "z_w": float(p_world[2]), "yaw_w": float(yaw_world)}


def hazard_in_frame_ego(haz_world: dict, frame_e2g: np.ndarray) -> tuple[float, float, float]:
    """world → current frame ego."""
    inv = np.linalg.inv(frame_e2g)
    p = inv @ np.array([haz_world["x_w"], haz_world["y_w"],
                        haz_world.get("z_w", 0.0), 1.0])
    cur_heading = float(np.arctan2(frame_e2g[1, 0], frame_e2g[0, 0]))
    yaw_e = haz_world["yaw_w"] - cur_heading
    return float(p[0]), float(p[1]), yaw_e


# ── Collision check ───────────────────────────────────────────────────

def waypoints_in_obb(waypoints_xy: np.ndarray,
                     hx: float, hy: float, hyaw: float,
                     length_m: float, width_m: float) -> np.ndarray:
    """Boolean per-waypoint, True if inside the OBB of length × width
    centered at (hx, hy) with yaw `hyaw` (radians)."""
    dx = waypoints_xy[:, 0] - hx
    dy = waypoints_xy[:, 1] - hy
    c, s = np.cos(-hyaw), np.sin(-hyaw)
    local_x = dx * c - dy * s
    local_y = dx * s + dy * c
    half_L = length_m / 2 + SAFETY_MARGIN_M
    half_W = width_m / 2 + SAFETY_MARGIN_M
    return (np.abs(local_x) <= half_L) & (np.abs(local_y) <= half_W)


# ── Reranking under VLM ───────────────────────────────────────────────

def vlm_rerank_idx(combined: np.ndarray, vocab_xy: np.ndarray,
                   vlm_areas: list, risk_threshold: float = RISK_THRESHOLD) -> int:
    """Return the index (0..8191) of the chosen trajectory after applying
    "exclude" rerank. Falls back to GTRS top-1 when filtering removes all."""
    if not vlm_areas:
        return int(np.argmax(combined))
    polygons: list = []
    risks: list = []
    for a in vlm_areas:
        poly = a.get("ego_xy_polygon", [])
        if len(poly) < 3:
            continue
        polygons.append([(float(x), float(y)) for x, y in poly])
        risks.append(float(a.get("risk", 0.0)))
    if not polygons or not any(r >= risk_threshold for r in risks):
        return int(np.argmax(combined))
    flat_wp = vocab_xy.reshape(-1, 2)
    n = vocab_xy.shape[0]
    excluded = np.zeros(n, dtype=bool)
    for poly, risk in zip(polygons, risks):
        if risk < risk_threshold:
            continue
        path = MplPath(np.asarray(poly, dtype=np.float32))
        inside = path.contains_points(flat_wp).reshape(vocab_xy.shape[:2])
        excluded |= inside.any(axis=1)
    if excluded.all():
        return int(np.argmax(combined))   # fallback
    masked = np.where(excluded, -np.inf, combined)
    return int(np.argmax(masked))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--policies", default="cascade,always_on",
                    help="Comma-separated VLM policies (must have <policy>/<sc>/<var>.pkl).")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--scenario_filter", default=None)
    ap.add_argument("--risk_threshold", type=float, default=RISK_THRESHOLD)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(open(args.manifest))
    log_dir = Path(manifest["defaults"]["log_dir"])
    cases = {c["id"]: c for c in manifest["cases"]}
    behaviors = {b["id"]: b for b in manifest["behaviors"]}

    if args.scenario_filter:
        keep = set(s.strip() for s in args.scenario_filter.split(","))
        manifest["scenarios"] = [s for s in manifest["scenarios"] if s["id"] in keep]
    print(f"[col] {len(manifest['scenarios'])} scenarios, "
          f"{len(cases)} cases, {len(behaviors)} behaviors")

    vocab = np.load(VOCAB_PATH)              # (8192, 40, 3)
    vocab_xy = vocab[..., :2].astype(np.float32)
    print(f"[col] vocab {vocab.shape}")

    policies = [p.strip() for p in args.policies.split(",")]

    # Streamed per-frame CSV
    per_frame_path = out_root / "per_frame.csv"
    fpf = open(per_frame_path, "w", newline="")
    pf_writer = csv.writer(fpf)
    pf_writer.writerow([
        "scenario", "variant", "case", "behavior", "frame_idx",
        "policy", "chosen_idx", "n_collisions_waypoints", "collided",
        "vlm_call_used",
    ])

    # ── per-(scenario, variant) loop ──
    t0 = time.time()
    n_pairs = sum(len(cases) * len(behaviors) for _ in manifest["scenarios"]) + len(manifest["scenarios"])
    pair_i = 0
    for sc in manifest["scenarios"]:
        log = pickle.load(open(log_dir / sc["log"], "rb"))
        end_frame_dict = log[sc["end_frame"]]

        # All variants in this scenario, including clean
        variants = ["clean"] + [
            f"{c['id']}__{b['id']}" for c in manifest["cases"] for b in manifest["behaviors"]
        ]

        for variant in variants:
            pair_i += 1
            # Load GTRS pkl
            gtrs_pkl = GTRS_PKLS_ROOT / sc["id"] / f"{variant}.pkl"
            if not gtrs_pkl.exists():
                print(f"[col] missing {gtrs_pkl}, skipping")
                continue
            gtrs = pickle.load(open(gtrs_pkl, "rb"))

            # VLM dumps per policy
            vlm = {p: {} for p in policies}
            for p in policies:
                vp = VLM_DUMP_ROOT / p / sc["id"] / f"{variant}.pkl"
                if vp.exists():
                    vlm[p] = pickle.load(open(vp, "rb"))

            # Hazard world pose for this (sc, var); clean → no hazard
            haz_world = None
            haz_lw = None
            haz_case = None
            if variant != "clean":
                case_id, beh_id = variant.split("__")
                haz_lw = CASE_BODY_LW_M.get(case_id)
                if haz_lw is None:
                    print(f"[col] no footprint for case {case_id}, skip")
                    continue
                beh = behaviors[beh_id]
                haz_world = hazard_world_pose(end_frame_dict,
                                              float(beh["end_x"]),
                                              float(beh["end_y"]))
                haz_case = case_id

            # iterate frames in this variant's GTRS pkl
            for fi in range(sc["start_frame"], sc["end_frame"] + 1):
                frame = log[fi]
                token = frame["token"]
                if token not in gtrs:
                    continue
                data = gtrs[token]
                if "imi" not in data:
                    continue
                comb = combined_score(data).astype(np.float32)

                # Hazard in current ego (skip clean)
                hx = hy = hyaw = None
                if haz_world is not None:
                    frame_e2g = np.asarray(frame["ego2global"], dtype=np.float64)
                    hx, hy, hyaw = hazard_in_frame_ego(haz_world, frame_e2g)

                for policy in ["baseline_gtrs"] + policies:
                    if policy == "baseline_gtrs":
                        chosen = int(np.argmax(comb))
                        vlm_call_used = 0
                    else:
                        entry = vlm[policy].get(token, {})
                        vlm_call_used = int(not entry.get("skipped_by_cascade", False)
                                             and entry.get("raw") is not None)
                        areas = entry.get("areas", [])
                        chosen = vlm_rerank_idx(comb, vocab_xy, areas,
                                                 risk_threshold=args.risk_threshold)

                    # Collision check (skip for clean)
                    n_wp_in = 0
                    collided = 0
                    if haz_world is not None:
                        wp = vocab_xy[chosen]                    # (40, 2)
                        in_obb = waypoints_in_obb(
                            wp, hx, hy, hyaw,
                            length_m=haz_lw[0], width_m=haz_lw[1],
                        )
                        n_wp_in = int(in_obb.sum())
                        collided = int(n_wp_in > 0)

                    pf_writer.writerow([
                        sc["id"], variant,
                        haz_case if variant != "clean" else "clean",
                        variant.split("__")[1] if variant != "clean" else "clean",
                        fi, policy, chosen, n_wp_in, collided, vlm_call_used,
                    ])
            if pair_i % 25 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / pair_i) * (n_pairs - pair_i) if pair_i else 0
                print(f"[col] {pair_i}/{n_pairs}  {sc['id']}/{variant}  "
                      f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

    fpf.close()
    print(f"[col] wrote {per_frame_path}")

    # Aggregate per (sc, var, policy)
    df = pd.read_csv(per_frame_path)
    pv = (df[df.case != "clean"]
          .groupby(["scenario", "variant", "case", "behavior", "policy"])
          .agg(n_frames=("frame_idx", "count"),
               n_collided_frames=("collided", "sum"),
               ever_collided=("collided", "max"),
               vlm_calls=("vlm_call_used", "sum"))
          .reset_index())
    pv["collision_rate"] = pv.n_collided_frames / pv.n_frames
    pv.to_csv(out_root / "per_variant.csv", index=False)
    print(f"[col] wrote {out_root / 'per_variant.csv'}  ({len(pv)} rows)")

    # Summary per policy (over modified variants only — clean has no hazard)
    summary_rows = []
    for policy in df.policy.unique():
        sub = df[(df.case != "clean") & (df.policy == policy)]
        if sub.empty:
            continue
        n = len(sub)
        n_col = int(sub.collided.sum())
        ever = int(sub.groupby(["scenario", "variant"]).collided.max().sum())
        n_var = sub.groupby(["scenario", "variant"]).ngroups
        n_calls = int(sub.vlm_call_used.sum())
        summary_rows.append({
            "policy": policy,
            "frames": n,
            "frame_collisions": n_col,
            "frame_collision_rate": n_col / n if n else 0,
            "variants": n_var,
            "variants_ever_collided": ever,
            "variant_collision_rate": ever / n_var if n_var else 0,
            "vlm_calls": n_calls,
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_root / "summary.csv", index=False)
    print("\n=== POLICY SUMMARY ===")
    print(sdf.to_string(index=False))

    # Plot: collision_rate (variant-level) per policy
    if len(sdf) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(sdf))
        ax.bar(x - 0.2, sdf["frame_collision_rate"], width=0.4,
               label="frame collision rate", color="#3a8ad6")
        ax.bar(x + 0.2, sdf["variant_collision_rate"], width=0.4,
               label="variant ever-collided rate", color="#d35a3a")
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["policy"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("rate")
        ax.set_title("Collision rate per policy")
        ax.legend()
        # annotations
        for i, row in sdf.iterrows():
            ax.text(i - 0.2, row["frame_collision_rate"] + 0.01,
                    f"{row['frame_collision_rate']:.2%}", ha="center", fontsize=8)
            ax.text(i + 0.2, row["variant_collision_rate"] + 0.01,
                    f"{row['variant_collision_rate']:.2%}", ha="center", fontsize=8)
            ax.text(i, -0.06, f"VLM calls: {row['vlm_calls']}",
                    ha="center", color="#555", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_root / "per_policy_bar.png", dpi=120)
        plt.close(fig)
        print(f"[col] wrote {out_root / 'per_policy_bar.png'}")


if __name__ == "__main__":
    main()
