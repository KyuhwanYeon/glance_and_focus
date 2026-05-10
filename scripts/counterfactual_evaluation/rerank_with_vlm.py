"""Re-rank GTRS-Dense's 8192 trajectory candidates using VLM-detected risk
polygons.

For each token:
  1. Compute the GTRS combined_score s_old (8192,).
  2. Project each VLM risk polygon (already in ego frame) onto the BEV plane;
     test each of the 8192 x 40 candidate waypoints for inclusion.
  3. EXCLUDE any candidate that has at least one waypoint inside a polygon
     whose risk >= --risk_threshold. The chosen trajectory is then the
     highest-combined-score one among the SURVIVING candidates -- i.e. the
     suboptimal-but-safe pick rather than the unconstrained optimum.
  4. If every candidate is filtered out (very rare; only when polygons cover
     the whole drivable region), fall back to a soft penalty:
       s_new = s_old - lambda * sum_p (risk_p * overlap_count_p)
     so we still emit something rather than crashing the rerank.

Augmented per-token outputs:
  vlm_overlap_count        (8192,) int32  -- # waypoints inside risky polygons
  vlm_excluded_mask        (8192,) bool   -- True if filtered out
  vlm_penalty              (8192,) float32-- soft penalty (only used on fallback)
  vlm_filtered_score       (8192,) float32-- s_old with -inf at excluded
  vlm_combined_score       (8192,) float32-- = vlm_filtered_score, kept for back-compat
  vlm_top_k_indices        (K,) int64
  vlm_top_k_scores         (K,) float32
  vlm_top_k_trajectories   (K, 40, 3) float32
  vlm_mode_used            'exclude' | 'penalty_fallback' | 'no_risk'
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).parent))
from topk_helper import combined_score   # noqa: E402


def compute_overlap_and_penalty(vocab_xy: np.ndarray,
                                  ego_polygons: list,
                                  risks: np.ndarray,
                                  risk_threshold: float):
    """vocab_xy: (8192, 40, 2). ego_polygons: list of [(x,y), ...].
    Returns:
      overlap_count_risky:  (8192,) int32 -- waypoints inside polygons with
                                              risk >= risk_threshold
      penalty:              (8192,) float32 -- sum over ALL polygons of
                                              (risk * overlap_count)
    """
    n = vocab_xy.shape[0]
    if not ego_polygons:
        return (np.zeros(n, dtype=np.int32),
                np.zeros(n, dtype=np.float32))
    flat_wp = vocab_xy.reshape(-1, 2)
    overlap_risky = np.zeros(n, dtype=np.int32)
    penalty = np.zeros(n, dtype=np.float32)
    for poly, risk in zip(ego_polygons, risks):
        if len(poly) < 3:
            continue
        path = MplPath(np.asarray(poly, dtype=np.float32))
        inside_flat = path.contains_points(flat_wp)
        inside = inside_flat.reshape(vocab_xy.shape[:2])
        overlap = inside.sum(axis=1).astype(np.int32)
        if float(risk) >= risk_threshold:
            overlap_risky += overlap
        penalty += float(risk) * overlap.astype(np.float32)
    return overlap_risky, penalty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--vlm_risk", required=True)
    ap.add_argument("--vocab", default="/workspace/CoachDrive/traj_final/8192.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk_threshold", type=float, default=0.5,
                    help="polygons with risk >= this threshold trigger exclusion")
    ap.add_argument("--lambda_penalty", type=float, default=8.0,
                    help="penalty mode multiplier (only used on fallback)")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    print(f"[rerank] loading predictions ({args.predictions}) ...")
    preds = pickle.load(open(args.predictions, "rb"))
    print(f"[rerank] loading vlm risks   ({args.vlm_risk}) ...")
    risks_db = pickle.load(open(args.vlm_risk, "rb"))
    print(f"[rerank] loading vocab       ({args.vocab}) ...")
    vocab = np.load(args.vocab)                                  # (8192, 40, 3)
    vocab_xy = vocab[..., :2].astype(np.float32)
    print(f"[rerank] vocab shape {vocab.shape}, "
          f"risk_threshold={args.risk_threshold}, top_k={args.top_k}")

    n_aug = n_skipped = n_with_risk = n_changed_best = 0
    n_exclude = n_fallback = n_no_risk = 0
    for token, data in preds.items():
        if "imi" not in data:
            n_skipped += 1
            continue
        old_score = combined_score(data).astype(np.float32)

        polygons: list = []
        polygon_risks: list = []
        if token in risks_db:
            for area in risks_db[token].get("areas", []):
                poly = area.get("ego_xy_polygon", [])
                if len(poly) >= 3:
                    polygons.append([(float(x), float(y)) for x, y in poly])
                    polygon_risks.append(float(area.get("risk", 0.5)))
        if polygons:
            n_with_risk += 1

        overlap_count, penalty = compute_overlap_and_penalty(
            vocab_xy, polygons,
            np.asarray(polygon_risks, dtype=np.float32),
            risk_threshold=args.risk_threshold,
        )
        excluded_mask = overlap_count > 0
        n_remaining = int((~excluded_mask).sum())

        if not polygons or not any(r >= args.risk_threshold for r in polygon_risks):
            mode = "no_risk"
            filtered_score = old_score.copy()
        elif n_remaining > 0:
            mode = "exclude"
            filtered_score = old_score.copy()
            filtered_score[excluded_mask] = -np.inf
        else:
            # All filtered out -> fall back to soft penalty
            mode = "penalty_fallback"
            filtered_score = old_score - args.lambda_penalty * penalty

        order = np.argsort(-filtered_score)
        top_k = order[:args.top_k].astype(np.int64)

        data["vlm_overlap_count"] = overlap_count
        data["vlm_excluded_mask"] = excluded_mask
        data["vlm_penalty"] = penalty
        data["vlm_filtered_score"] = filtered_score
        data["vlm_combined_score"] = filtered_score   # back-compat alias
        data["vlm_top_k_indices"] = top_k
        data["vlm_top_k_scores"] = filtered_score[top_k]
        data["vlm_top_k_trajectories"] = vocab[top_k].astype(np.float32)
        data["vlm_mode_used"] = mode

        if mode == "exclude":
            n_exclude += 1
        elif mode == "penalty_fallback":
            n_fallback += 1
        else:
            n_no_risk += 1

        old_best_idx = int(np.argmax(old_score))
        new_best_idx = int(top_k[0])
        if old_best_idx != new_best_idx:
            n_changed_best += 1
        n_aug += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(preds, open(args.out, "wb"))
    print(f"[rerank] augmented {n_aug} tokens (n_with_risk={n_with_risk}); "
          f"exclude={n_exclude} no_risk={n_no_risk} "
          f"penalty_fallback={n_fallback}; "
          f"changed_best={n_changed_best} -> {args.out}")


if __name__ == "__main__":
    main()
