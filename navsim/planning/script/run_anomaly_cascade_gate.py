"""Compute SigLIP2-kNN anomaly score per token and derive a cascade-gated
vlm_risk pkl from an always-on dump.

For each token in `--predictions`:
  1. Locate the CAM_F0 image via the navsim log .pkls.
  2. Crop fwd_strip (center-65% horizontal, vertical 40-90%) and embed
     with SigLIP2 (google/siglip-base-patch16-224).
  3. kNN-k cosine distance to a reference bank of normal driving frames.
     score > tau  -> anomalous; keep its VLM `areas` from the always-on dump.
     score <= tau -> normal; clear its `areas` (cascade skips VLM).

Outputs:
  --out_vlm       cascade-gated vlm_risk pkl (drop-in replacement for
                  the always-on vlm_risk pkl in rerank_predictions_with_vlm.py)
  --out_scores    optional: {token: anomaly_score} pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from transformers import AutoModel, AutoProcessor


SIGLIP = "google/siglip-base-patch16-224"


def _crop_fwd_strip(img: Image.Image,
                     w_frac: float = 0.65,
                     h_top: float = 0.4,
                     h_bot: float = 0.9) -> Image.Image:
    """Match anomaly_poc.py 'fwd_strip' crop."""
    W, H = img.size
    x0 = int(W * (1 - w_frac) / 2)
    x1 = int(W * (1 + w_frac) / 2)
    y0 = int(H * h_top)
    y1 = int(H * h_bot)
    return img.crop((x0, y0, x1, y1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="merged_predictions pkl; we score every token in it")
    ap.add_argument("--vlm_risk", required=True,
                    help="always-on vlm_risk pkl to gate")
    ap.add_argument("--navsim_logs", default="/dataset/navsim_logs/test")
    ap.add_argument("--sensor_blobs", default="/dataset/sensor_blobs/test")
    ap.add_argument("--ref_embeds", required=True,
                    help="path to ref_embeds_diverse_fwd_strip.npy "
                         "(see README for download link)")
    ap.add_argument("--tau", type=float, default=0.084423,
                    help="kNN distance threshold; score>tau -> anomalous")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out_vlm", required=True)
    ap.add_argument("--out_scores", default=None)
    args = ap.parse_args()

    print(f"[gate] loading predictions:    {args.predictions}")
    preds = pickle.load(open(args.predictions, "rb"))
    print(f"[gate] loading vlm risk (all): {args.vlm_risk}")
    vlm = pickle.load(open(args.vlm_risk, "rb"))
    print(f"[gate] loading reference:      {args.ref_embeds}")
    ref = np.load(args.ref_embeds).astype(np.float32)
    target = set(preds.keys())
    print(f"[gate] {len(target)} tokens to score; ref={ref.shape}")

    log_root = Path(args.navsim_logs)
    sensor_root = Path(args.sensor_blobs)
    log_pkls = sorted(log_root.glob("*.pkl"))
    print(f"[gate] scanning {len(log_pkls)} navsim log pkls for token->image paths...")
    token_to_path = {}
    t0 = time.time()
    for lp in log_pkls:
        try:
            frames = pickle.load(open(lp, "rb"))
        except Exception as e:
            print(f"  [warn] failed to read {lp.name}: {e}")
            continue
        for f in frames:
            tok = f.get("token")
            if tok in target and tok not in token_to_path:
                cam = f["cams"]["CAM_F0"]
                token_to_path[tok] = sensor_root / cam["data_path"]
    print(f"[gate]   mapped {len(token_to_path)}/{len(target)} tokens "
          f"in {time.time()-t0:.0f}s")
    missing = target - set(token_to_path)
    if missing:
        print(f"[gate]   warn: {len(missing)} tokens have no image; left unscored")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gate] loading {SIGLIP} on {device}...")
    processor = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP).to(device).eval()

    tokens_ordered = list(token_to_path.keys())
    paths = [str(token_to_path[t]) for t in tokens_ordered]

    @torch.no_grad()
    def embed_batches(paths, B):
        out = []
        for i in range(0, len(paths), B):
            chunk = paths[i:i + B]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            imgs = [_crop_fwd_strip(im) for im in imgs]
            inp = processor(images=imgs, return_tensors="pt").to(device)
            feats = model.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy())
            if ((i // B) % 5) == 0:
                print(f"  embedded {min(i + B, len(paths))}/{len(paths)}")
        return np.concatenate(out, axis=0)

    t0 = time.time()
    embeds = embed_batches(paths, args.batch).astype(np.float32)
    print(f"[gate] embedded {len(embeds)} frames in {time.time()-t0:.0f}s")

    dists = cdist(embeds, ref, metric="cosine")
    knn = np.sort(dists, axis=1)[:, :args.k].mean(axis=1)
    scores = {tok: float(s) for tok, s in zip(tokens_ordered, knn)}

    n_anom = sum(1 for s in scores.values() if s > args.tau)
    n_total = len(scores)
    pct_max = (knn > args.tau).mean() * 100 if n_total else 0.0
    print(f"[gate] tau={args.tau:.5f}  k={args.k}")
    print(f"[gate] anomaly score: min={knn.min():.4f} median={np.median(knn):.4f} "
          f"mean={knn.mean():.4f} max={knn.max():.4f}")
    print(f"[gate] anomalous: {n_anom}/{n_total} ({pct_max:.1f}%)")

    cascade = {}
    for tok, rec in vlm.items():
        new = dict(rec)
        s = scores.get(tok)
        if s is None or s <= args.tau:
            new["areas"] = []
            new["skipped_by_cascade"] = True
        else:
            new["skipped_by_cascade"] = False
        new["anomaly_score"] = s
        cascade[tok] = new

    Path(args.out_vlm).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cascade, open(args.out_vlm, "wb"))
    print(f"[gate] saved cascade vlm_risk -> {args.out_vlm}")

    if args.out_scores:
        Path(args.out_scores).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(scores, open(args.out_scores, "wb"))
        print(f"[gate] saved per-token anomaly scores -> {args.out_scores}")


if __name__ == "__main__":
    main()
