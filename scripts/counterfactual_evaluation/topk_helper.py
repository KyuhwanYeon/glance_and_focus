"""Shared helper: compute top-K trajectory candidates from a results_map of
GTRS-Dense subscores, using the same combined-score formula as the model's
selection logic in hydra_model.py:399-414.

Augments each token's dict with:
  - 'top_k_indices'      (K,)        int64
  - 'top_k_scores'       (K,)        float32 — combined score (higher = better)
  - 'top_k_trajectories' (K, 40, 3)  float32 — vocab[indices]
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def combined_score(data: dict) -> np.ndarray:
    """Combined ranking score over the 8192 vocab entries, matching
    HydraTrajHead's selection formula. Returns shape (8192,)."""
    eps = 1e-12
    return (
        0.03 * np.log(_softmax(data["imi"], axis=-1) + eps)
        + 0.1 * np.log(_sigmoid(data["traffic_light_compliance"]) + eps)
        + 0.1 * np.log(_sigmoid(data["no_at_fault_collisions"]) + eps)
        + 0.9 * np.log(_sigmoid(data["drivable_area_compliance"]) + eps)
        + 0.2 * np.log(_sigmoid(data["driving_direction_compliance"]) + eps)
        + 6.0 * np.log(
            7.0 * _sigmoid(data["time_to_collision_within_bound"])
            + 7.0 * _sigmoid(data["ego_progress"])
            + 3.0 * _sigmoid(data["lane_keeping"])
            + eps
        )
    )


def augment_with_top_k(results_map: dict, vocab_path: str, top_k: int = 10):
    """In-place: add top_k_indices/scores/trajectories to each token's dict."""
    vocab = np.load(vocab_path)        # (8192, 40, 3)
    n_aug = 0
    for tok, data in results_map.items():
        if "imi" not in data:
            continue
        s = combined_score(data)
        order = np.argsort(-s)
        idx = order[:top_k].astype(np.int64)
        data["top_k_indices"] = idx
        data["top_k_scores"] = s[idx].astype(np.float32)
        data["top_k_trajectories"] = vocab[idx].astype(np.float32)
        n_aug += 1
    print(f"[topk] augmented {n_aug} tokens with top-{top_k} candidates")
