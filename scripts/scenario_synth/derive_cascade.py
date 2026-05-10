"""Derive a cascade VLM dump from the always_on dump by gating on a
(optionally smoothed) anomaly score.

For each frame:
  - anomaly_score_smoothed > tau  -> keep the always_on areas (gate fires)
  - anomaly_score_smoothed <= tau -> empty areas + skipped_by_cascade=True

The smoothing window is a centered rolling mean per (scenario, variant) over
the per-frame anomaly score from exp/anomaly_classify/classified.csv.

Default config (tau=0.07, window=9) is the operational sweet spot:
36.6% variant collision rate at 40% of always_on's VLM calls — within 2.1 pp
of always_on quality at 60 % cost reduction.

Usage:
  /opt/conda/envs/navsim/bin/python scripts/scenario_synth/derive_cascade.py \\
      --tau 0.07 --window 9
"""
import argparse
import pickle
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/workspace/CoachDrive/exp/vlm_dump/always_on")
ap.add_argument("--dst", default="/workspace/CoachDrive/exp/vlm_dump/cascade")
ap.add_argument("--classified",
                default="/workspace/CoachDrive/exp/anomaly_classify/classified.csv",
                help="Per-frame anomaly_score table (used for smoothing).")
ap.add_argument("--tau", type=float, default=0.07,
                help="Threshold on smoothed anomaly_score; gate fires when > tau.")
ap.add_argument("--window", type=int, default=9,
                help="Centered rolling-mean window per (scenario, variant). "
                     "Use 1 for no smoothing.")
args = ap.parse_args()

src, dst = Path(args.src), Path(args.dst)
print(f"[derive] tau={args.tau}  window={args.window}")
print(f"[derive] src={src}  dst={dst}")

# Smooth anomaly scores per (sc, var) once, then index by token via classified
print(f"[derive] loading {args.classified}")
df = pd.read_csv(args.classified)
if args.window > 1:
    smoothed = []
    for _, sub in df.groupby(["scenario", "variant"], sort=False):
        sub = sub.sort_values("frame_idx").copy()
        sub["smoothed"] = (sub["anomaly_score"]
                           .rolling(window=args.window, center=True, min_periods=1)
                           .mean())
        smoothed.append(sub)
    df = pd.concat(smoothed, ignore_index=True)
else:
    df["smoothed"] = df["anomaly_score"]
score_by_key = df.set_index(["scenario", "variant", "frame_idx"])["smoothed"].to_dict()

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

n_files = n_kept = n_gated = 0
for sc_dir in sorted(src.iterdir()):
    if not sc_dir.is_dir():
        continue
    out_sc = dst / sc_dir.name
    out_sc.mkdir(parents=True)
    for pkl in sorted(sc_dir.glob("*.pkl")):
        d = pickle.load(open(pkl, "rb"))
        out = {}
        var = pkl.stem
        sc = sc_dir.name
        for tok, rec in d.items():
            fi = rec.get("frame_idx")
            ascore_smooth = float(score_by_key.get((sc, var, fi),
                                                   rec.get("anomaly_score", 0.0)))
            if ascore_smooth > args.tau:
                # gate fires — keep always_on detections, but record the
                # smoothed score that gated this frame
                rec_out = dict(rec)
                rec_out["anomaly_score_smoothed"] = ascore_smooth
                out[tok] = rec_out
                n_kept += 1
            else:
                out[tok] = {
                    "areas": [], "raw": None, "tokens": None,
                    "latency_ms": 0.0, "skipped_by_cascade": True,
                    "frame_idx": fi,
                    "speed_mps": rec.get("speed_mps"),
                    "anomaly_score": rec.get("anomaly_score"),
                    "anomaly_score_smoothed": ascore_smooth,
                }
                n_gated += 1
        with open(out_sc / pkl.name, "wb") as fh:
            pickle.dump(out, fh)
        n_files += 1

print(f"[derive] files={n_files}  kept_areas={n_kept}  gated_skipped={n_gated}  "
      f"call_rate={n_kept / max(1, n_kept + n_gated) * 100:.1f}%")
print(f"[derive] wrote {dst}")
