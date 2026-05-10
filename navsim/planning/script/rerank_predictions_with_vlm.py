"""Substitute merged_predictions trajectories with VLM-reranked top-1.

Input:
  --predictions   merged_predictions pkl from run_pdm_score_*_gpu.py
                  (must contain GTRS subscores: imi, no_at_fault_collisions,
                  drivable_area_compliance, time_to_collision_within_bound,
                  ego_progress, driving_direction_compliance, lane_keeping,
                  traffic_light_compliance)
  --vlm_risk      VLM risk pkl from run_vlm_dump_on_predictions.py

For each token:
  - score = combined_score(data) over the 8192 vocab
  - any candidate whose 40-step waypoints intersect a polygon with
    risk >= --risk_threshold is excluded; if all are excluded we fall
    back to a soft penalty (same as rerank_with_vlm.py).
  - new top-1 = argmax(filtered_score), wrapped in Trajectory at 10Hz/4s.
  - predictions[token]['trajectory'] is overwritten in place.

Output predictions are scored with score_from_predictions.sh /
run_pdm_score_from_predictions.py.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory

_DEVKIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DEVKIT / "scripts" / "counterfactual_evaluation"))

from rerank_with_vlm import compute_overlap_and_penalty  # noqa: E402
from topk_helper import combined_score                    # noqa: E402


def compute_overlap_and_buffered_penalty(
    vocab_xy: np.ndarray,
    ego_polygons: list,
    risks: np.ndarray,
    risk_threshold: float,
    radius_m: float,
    ego_half_width_m: float = 0.0,
):
    """Distance-aware penalty with optional ego footprint (tube).

    For each waypoint, compute Euclidean distance d to the polygon (d=0
    inside). Subtract the ego half-width to model the vehicle footprint as
    a disk of that radius around each waypoint -- the effective contact
    distance is `d_eff = max(0, d - ego_half_width_m)`. Then apply a
    linear-falloff weight over an extra `radius_m`:

      weight = clip(1 - d_eff / radius_m, 0, 1)

      ego_half_width=0, radius=0   -> hard inside/outside (caller should
                                       use the original compute_overlap_*).
      ego_half_width>0, radius=0   -> Minkowski-inflated hard polygon
                                       (waypoint counts iff its footprint
                                       overlaps polygon).
      ego_half_width=0, radius>0   -> point-based linear falloff over R.
      ego_half_width>0, radius>0   -> footprint contact zone (full cost)
                                       + linear soft buffer of width R
                                       beyond the contact zone.

    Returns (overlap_count_risky, penalty), shape (N,).
    `overlap_count_risky` counts trajectories with at least one waypoint
    inside the *full* danger zone (d_eff < radius_m, or d_eff == 0 if
    radius_m == 0) and risk >= risk_threshold -- used for hard exclusion.
    """
    n = vocab_xy.shape[0]
    if not ego_polygons:
        return (np.zeros(n, dtype=np.int32),
                np.zeros(n, dtype=np.float32))
    flat_wp = vocab_xy.reshape(-1, 2).astype(np.float64)
    pts = shapely.points(flat_wp[:, 0], flat_wp[:, 1])
    overlap_risky = np.zeros(n, dtype=np.int32)
    penalty = np.zeros(n, dtype=np.float32)
    for poly, risk in zip(ego_polygons, risks):
        if len(poly) < 3:
            continue
        p = Polygon(poly)
        if not p.is_valid:
            p = p.buffer(0)
            if p.is_empty:
                continue
        d = shapely.distance(p, pts)                              # 0 if inside
        d_eff = np.clip(d - ego_half_width_m, 0.0, None)
        if radius_m > 0:
            weight_flat = np.clip(1.0 - d_eff / radius_m, 0.0, 1.0)
        else:
            # radius_m==0: only the contact zone (d_eff==0) contributes
            weight_flat = (d_eff <= 0).astype(np.float32)
        weight = weight_flat.reshape(vocab_xy.shape[:2]).astype(np.float32)
        per_traj = weight.sum(axis=1)                             # (8192,)
        if float(risk) >= risk_threshold:
            within = (weight > 0).any(axis=1).astype(np.int32)
            overlap_risky += within
        penalty += float(risk) * per_traj
    return overlap_risky, penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--vlm_risk", required=True)
    ap.add_argument("--vocab", default=str(_DEVKIT / "traj_final" / "8192.npy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk_threshold", type=float, default=0.5)
    ap.add_argument("--lambda_penalty", type=float, default=8.0)
    ap.add_argument("--mode", choices=["exclude_only", "mix"], default="exclude_only",
                    help="exclude_only: original behavior (hard exclude risk>=τ; "
                         "soft penalty only as fallback when all candidates excluded). "
                         "mix: always apply soft penalty over ALL polygons + hard "
                         "exclude on risk>=τ.")
    ap.add_argument("--radius_m", type=float, default=0.0,
                    help="Soft-falloff radius (m) BEYOND the ego footprint. "
                         "weight = clip(1 - max(0, d - ego_half_width)/radius_m, 0, 1). "
                         "0 disables soft falloff (use with --ego_half_width_m for "
                         "hard inflated polygon, or both 0 = original point test).")
    ap.add_argument("--ego_half_width_m", type=float, default=0.0,
                    help="Ego vehicle half-width (m). Treats each waypoint as a "
                         "disk of this radius (Minkowski sum) so any candidate "
                         "whose footprint touches the polygon counts as contact. "
                         "Pacifica is ~2.0m wide -> 1.0 is realistic.")
    args = ap.parse_args()

    print(f"[rerank-preds] preds   : {args.predictions}")
    preds = pickle.load(open(args.predictions, "rb"))
    print(f"[rerank-preds] vlm     : {args.vlm_risk}")
    vlm = pickle.load(open(args.vlm_risk, "rb"))
    print(f"[rerank-preds] vocab   : {args.vocab}")
    vocab = np.load(args.vocab)
    vocab_xy = vocab[..., :2].astype(np.float32)
    print(f"[rerank-preds] preds={len(preds)} vlm={len(vlm)} vocab={vocab.shape}")
    print(f"[rerank-preds] risk_threshold={args.risk_threshold} "
          f"lambda_penalty={args.lambda_penalty} radius_m={args.radius_m} "
          f"ego_half_width_m={args.ego_half_width_m}")

    sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    n_skip = n_no_risk = n_excluded = n_fallback = n_penalty_soft = n_changed = 0

    for token, data in preds.items():
        if "imi" not in data:
            n_skip += 1
            continue
        old_score = combined_score(data).astype(np.float32)

        polygons, polygon_risks = [], []
        if token in vlm:
            for area in vlm[token].get("areas", []):
                poly = area.get("ego_xy_polygon", [])
                if len(poly) >= 3:
                    polygons.append([(float(x), float(y)) for x, y in poly])
                    polygon_risks.append(float(area.get("risk", 0.5)))

        if args.radius_m > 0 or args.ego_half_width_m > 0:
            overlap_count, penalty = compute_overlap_and_buffered_penalty(
                vocab_xy,
                polygons,
                np.asarray(polygon_risks, dtype=np.float32),
                risk_threshold=args.risk_threshold,
                radius_m=args.radius_m,
                ego_half_width_m=args.ego_half_width_m,
            )
        else:
            overlap_count, penalty = compute_overlap_and_penalty(
                vocab_xy,
                polygons,
                np.asarray(polygon_risks, dtype=np.float32),
                risk_threshold=args.risk_threshold,
            )
        excluded_mask = overlap_count > 0
        n_remaining = int((~excluded_mask).sum())
        any_risky = any(r >= args.risk_threshold for r in polygon_risks)

        if args.mode == "exclude_only":
            if not polygons or not any_risky:
                mode = "no_risk"
                filtered = old_score.copy()
                n_no_risk += 1
            elif n_remaining > 0:
                mode = "exclude"
                filtered = old_score.copy()
                filtered[excluded_mask] = -np.inf
                n_excluded += 1
            else:
                mode = "penalty_fallback"
                filtered = old_score - args.lambda_penalty * penalty
                n_fallback += 1
        else:  # mix
            # Always apply soft penalty (every polygon, weighted by its risk).
            filtered = old_score - args.lambda_penalty * penalty
            if any_risky and n_remaining > 0:
                # Hard-exclude only the candidates that hit a risk>=τ polygon.
                mode = "exclude+penalty"
                filtered[excluded_mask] = -np.inf
                n_excluded += 1
            elif any_risky:
                mode = "penalty_fallback"
                n_fallback += 1
            elif polygons:
                mode = "penalty_only"   # mild polygons (risk<τ) still nudge ranking
                n_penalty_soft += 1
            else:
                mode = "no_risk"
                n_no_risk += 1

        old_top = int(np.argmax(old_score))
        new_top = int(np.argmax(filtered))
        if new_top != old_top:
            n_changed += 1
            new_traj = vocab[new_top].astype(np.float32)
            data["trajectory"] = Trajectory(new_traj, sampling)
        data["vlm_mode_used"] = mode
        data["vlm_top1_idx"] = new_top
        data["vlm_old_top1_idx"] = old_top

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(args.out, "wb"))
    print(f"[rerank-preds] mode={args.mode} | no_risk={n_no_risk} "
          f"exclude={n_excluded} penalty_only={n_penalty_soft} "
          f"fallback={n_fallback} skipped={n_skip} changed_best={n_changed}")
    print(f"[rerank-preds] saved -> {args.out}")


if __name__ == "__main__":
    main()
