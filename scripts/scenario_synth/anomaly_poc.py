"""SigLIP2 anomaly-detection PoC: embedding extraction.

Stage 1 of the anomaly pipeline. Embeds every test frame (clean + modified)
in the benchmark and a reference clean-driving corpus from other NAVSIM logs,
under three crop variants:

  no_crop     full F0 image
  center_50   center 50% horizontal (preserves full vertical)
  fwd_strip   center 50% horizontal × lower-vertical band 40-90%
              (the "forward driving FOV" where road hazards appear)

Output:
  exp/anomaly_poc/scores.csv           — per-frame metadata + global-ref
                                         kNN/Mahalanobis scores (no_crop only)
  exp/anomaly_poc/embeds_<crop>.npy    — (N_test, 768) embeddings, one per crop
  exp/anomaly_poc/ref_embeds.npy       — (N_ref, 768) reference embeddings (no_crop)
  exp/anomaly_poc/score_box_per_case.png
  exp/anomaly_poc/score_timeseries.png

Run in navsim env (transformers + sklearn already there):

    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/anomaly_poc.py
    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/anomaly_poc.py --crop fwd_strip
    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/anomaly_poc.py --crop all   # all 3 crops
"""
from __future__ import annotations

import argparse
import glob
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoProcessor

MANIFEST = "/workspace/CoachDrive/scripts/scenario_synth/benchmark_manifest.yaml"
SENSOR_ROOT = Path("/dataset/sensor_blobs/test")
LOG_DIR = Path("/dataset/navsim_logs/test")
BENCHMARK_DIR = Path("/workspace/CoachDrive/exp/benchmark")
OUT_DIR = Path("/workspace/CoachDrive/exp/anomaly_poc")
SIGLIP_MODEL = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
N_REF_LOGS = 50
N_FRAMES_PER_REF_LOG = 5

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Crops ─────────────────────────────────────────────────────────────

def _crop_center_strip(img: Image.Image,
                       w_frac: float = 0.5,
                       h_top_frac: float = 0.0,
                       h_bot_frac: float = 1.0) -> Image.Image:
    W, H = img.size
    x0 = int(W * (1 - w_frac) / 2)
    x1 = int(W * (1 + w_frac) / 2)
    y0 = int(H * h_top_frac)
    y1 = int(H * h_bot_frac)
    return img.crop((x0, y0, x1, y1))


CROP_FNS = {
    "no_crop":   lambda im: im,
    "center_50": lambda im: _crop_center_strip(im, w_frac=0.5),
    "fwd_strip": lambda im: _crop_center_strip(im, w_frac=0.65,
                                               h_top_frac=0.4, h_bot_frac=0.9),
}


# ── Embedding ─────────────────────────────────────────────────────────

def load_encoder() -> tuple:
    print(f"[poc] loading {SIGLIP_MODEL} on {DEVICE} ...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(DEVICE).eval()
    return model, processor


@torch.no_grad()
def embed_paths(paths: list[str], model, processor, crop_name: str = "no_crop",
                batch: int = BATCH) -> np.ndarray:
    crop_fn = CROP_FNS[crop_name]
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        imgs = [crop_fn(im) for im in imgs]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Reference corpus ──────────────────────────────────────────────────

def build_reference_paths(our_log_set: set[str]) -> list[str]:
    """N_REF_LOGS × N_FRAMES_PER_REF_LOG clean F0 frame paths from logs NOT
    in our scenario set."""
    all_logs = sorted(glob.glob(str(LOG_DIR / "*.pkl")))
    other_logs = [p for p in all_logs if Path(p).name not in our_log_set]
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(other_logs), size=min(N_REF_LOGS, len(other_logs)),
                        replace=False)
    paths: list[str] = []
    for li in chosen:
        log = pickle.load(open(other_logs[li], "rb"))
        if len(log) < 60:
            continue
        idxs = np.linspace(20, len(log) - 20, N_FRAMES_PER_REF_LOG, dtype=int)
        for fi in idxs:
            paths.append(str(SENSOR_ROOT / log[fi]["cams"]["CAM_F0"]["data_path"]))
    return paths


# ── Test set ──────────────────────────────────────────────────────────

def build_test_table(manifest: dict) -> pd.DataFrame:
    rows = []
    for sc in manifest["scenarios"]:
        log = pickle.load(open(LOG_DIR / sc["log"], "rb"))
        for fi in range(sc["start_frame"], sc["end_frame"] + 1):
            rows.append(dict(
                scenario=sc["id"], variant="clean",
                case="clean", behavior="clean",
                frame_idx=fi,
                img_path=str(SENSOR_ROOT / log[fi]["cams"]["CAM_F0"]["data_path"]),
            ))
        for ca in manifest["cases"]:
            for be in manifest["behaviors"]:
                v = f"{ca['id']}__{be['id']}"
                for fi in range(sc["start_frame"], sc["end_frame"] + 1):
                    rows.append(dict(
                        scenario=sc["id"], variant=v,
                        case=ca["id"], behavior=be["id"],
                        frame_idx=fi,
                        img_path=str(BENCHMARK_DIR / sc["id"] / v / f"f{fi:04d}.jpg"),
                    ))
    return pd.DataFrame(rows)


# ── Scoring ───────────────────────────────────────────────────────────

def score_mahalanobis(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    mu = ref.mean(0)
    cov = np.cov(ref.T) + 1e-3 * np.eye(ref.shape[1])
    inv = np.linalg.inv(cov)
    diff = test - mu
    return np.einsum("ij,jk,ik->i", diff, inv, diff)


def score_knn(ref: np.ndarray, test: np.ndarray, k: int = 5) -> np.ndarray:
    d = cdist(test, ref, metric="cosine")
    return np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)


# ── Plots ─────────────────────────────────────────────────────────────

def plot_score_breakdown(df: pd.DataFrame, score_col: str, out_path: Path) -> None:
    cases = sorted(df.case.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.boxplot([df[df.case == c][score_col].values for c in cases],
                       labels=cases, vert=True, patch_artist=True)
    for p in parts["boxes"]:
        p.set_facecolor((0.7, 0.85, 1.0))
    ax.set_ylabel(score_col)
    ax.set_title(f"Anomaly score per case ({score_col})")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[poc] wrote {out_path}")


def plot_per_scenario_timeseries(df: pd.DataFrame, score_col: str,
                                 out_path: Path) -> None:
    case_colors = {
        "pedestrian_human":       "#3060f0",
        "pedestrian_duck_mascot": "#3098f0",
        "animal_bear":            "#9d6a3a",
        "animal_wolf":            "#666",
        "construction_cone":      "#f0a030",
        "construction_dumpster":  "#3da030",
        "vehicle_wrecked":        "#a040a0",
        "vehicle_normal":         "#3da0c0",
        "clean":                  "#000",
    }
    scenarios = sorted(df.scenario.unique())
    cols = 3
    rows = (len(scenarios) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), sharey=True)
    axes = np.atleast_2d(axes)
    for i, sc in enumerate(scenarios):
        ax = axes[i // cols, i % cols]
        sub = df[df.scenario == sc]
        frames = sorted(sub.frame_idx.unique())
        end_fi = max(frames)
        for case in sorted(sub.case.unique()):
            sub2 = sub[sub.case == case].sort_values("frame_idx")
            means = sub2.groupby("frame_idx")[score_col].mean().reindex(frames).values
            xs = np.array(frames) - end_fi
            ax.plot(xs, means, color=case_colors.get(case, "#666"),
                    label=case, lw=1.4 if case != "clean" else 2.4,
                    ls="-" if case != "clean" else "--")
        ax.set_title(sc, fontsize=10)
        ax.axvline(0, color="red", lw=0.6, ls=":")
        ax.set_xlabel("frame idx − end")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(f"Per-scenario anomaly score over time ({score_col})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[poc] wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="all",
                    choices=["all"] + list(CROP_FNS.keys()),
                    help="Image crop to embed under. 'all' = run all three.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a crop if its embeds_<crop>.npy already exists.")
    args = ap.parse_args()

    with open(MANIFEST) as f:
        manifest = yaml.safe_load(f)
    our_log_set = {s["log"] for s in manifest["scenarios"]}

    df = build_test_table(manifest)
    print(f"[poc] test set: {len(df)} frames "
          f"({(df.variant == 'clean').sum()} clean + {(df.variant != 'clean').sum()} modified)")

    ref_paths = build_reference_paths(our_log_set)
    print(f"[poc] reference: {len(ref_paths)} clean F0 frames from "
          f"{N_REF_LOGS} other NAVSIM logs")

    model, processor = load_encoder()

    crops = list(CROP_FNS.keys()) if args.crop == "all" else [args.crop]
    for crop in crops:
        emb_path = OUT_DIR / f"embeds_{crop}.npy"
        if args.skip_existing and emb_path.exists():
            print(f"[poc] {crop}: existing {emb_path}, skipping")
            continue
        print(f"\n[poc] === crop={crop} ===")
        t0 = time.time()
        emb = embed_paths(df.img_path.tolist(), model, processor, crop_name=crop)
        np.save(emb_path, emb)
        print(f"[poc] {crop}: embeds shape {emb.shape}  ({time.time()-t0:.1f}s)")

        if crop == "no_crop":
            # Build reference + global-ref baseline scores from no_crop embeddings
            t1 = time.time()
            ref = embed_paths(ref_paths, model, processor, crop_name=crop)
            np.save(OUT_DIR / "ref_embeds.npy", ref)
            print(f"[poc] reference: shape {ref.shape}  ({time.time()-t1:.1f}s)")
            df["score_maha"] = score_mahalanobis(ref, emb)
            df["score_knn5"] = score_knn(ref, emb, k=5)
            df.to_csv(OUT_DIR / "scores.csv", index=False)
            print(f"[poc] wrote {OUT_DIR / 'scores.csv'}")

            labels = (df.variant != "clean").astype(int).values
            print(f"[poc] AUC (Mahalanobis)    = {roc_auc_score(labels, df.score_maha.values):.4f}")
            print(f"[poc] AUC (kNN cosine k=5) = {roc_auc_score(labels, df.score_knn5.values):.4f}")
            print("\n[poc] mean score per case:")
            print(df.groupby("case")[["score_maha", "score_knn5"]].mean().sort_values("score_knn5"))

            plot_score_breakdown(df, "score_knn5", OUT_DIR / "score_box_per_case.png")
            plot_per_scenario_timeseries(df, "score_knn5", OUT_DIR / "score_timeseries.png")

    print(f"\n[poc] DONE — see {OUT_DIR} for outputs")


if __name__ == "__main__":
    main()
