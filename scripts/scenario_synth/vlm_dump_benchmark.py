"""Run Cosmos-Reason2 over the full scenario_synth benchmark.

Two policies, one script:

  --policy always_on   call VLM on every frame  (ground-truth dataset)
  --policy cascade     call VLM only when anomaly_score > --tau
                       (deployment-realistic; uses exp/anomaly_classify/classified.csv)

Per call we save:
  prompt_tokens, completion_tokens, total_tokens, latency_ms
  raw VLM JSON, parsed `areas` (back-projected ego polygons)

Output layout:
  exp/vlm_dump/<policy>/<scenario>/<variant>.pkl
      token -> {"areas": [...], "raw": "...", "tokens": {...}, "latency_ms": ...}
  exp/vlm_dump/<policy>/calls.csv
      one row per VLM call (scenario, variant, frame_idx, tokens, latency_ms)
  exp/vlm_dump/<policy>/_summary.txt
      total calls / total tokens / total seconds / mean latency

Reuses the existing single-image VLM infrastructure under
scripts/counterfactual_evaluation/.

Usage (navsim env):

    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/vlm_dump_benchmark.py \\
        --policy cascade --tau 0.0962          # ~75 min
    /opt/conda/envs/navsim/bin/python scripts/scenario_synth/vlm_dump_benchmark.py \\
        --policy always_on                      # overnight (~8 h)
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, "/workspace/CoachDrive/scripts/counterfactual_evaluation")
from cosmos_vlm import get_cosmos                                       # noqa: E402
from cosmos_risk_to_bev import _bbox_to_circle_polygon                  # noqa: E402
from vlm_prompt import PROMPT_TEMPLATE, DRIVING_CMD_MAP                 # noqa: E402
from run_vlm_risk_dump_singleimage import (                             # noqa: E402
    _try_parse, _recover_partial_detections, _dedupe_detections,
)


MANIFEST = "/workspace/CoachDrive/scripts/scenario_synth/benchmark_manifest.yaml"
BENCHMARK_DIR = Path("/workspace/CoachDrive/exp/benchmark")
SENSOR_ROOT = Path("/dataset/sensor_blobs/test")
CLASSIFIED_CSV = "/workspace/CoachDrive/exp/anomaly_classify/classified.csv"


# ── VLM call with token accounting ────────────────────────────────────

@torch.no_grad()
def call_vlm_with_tokens(model, processor, img_bgr: np.ndarray,
                         prompt: str, max_new_tokens: int = 512) -> dict:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You localize hazards in driving images and output JSON only."}]},
        {"role": "user",
         "content": [
             {"type": "image", "image": pil},
             {"type": "text",  "text": prompt},
         ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    in_tokens = int(inputs["input_ids"].shape[1])

    t0 = time.time()
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency_ms = (time.time() - t0) * 1000
    gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
    out_tokens = int(gen_only.shape[1])

    answer = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return {
        "raw": answer,
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "latency_ms": latency_ms,
    }


def _coerce_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_and_backproject(raw: str, K, R, t, W, H) -> list:
    parsed = _try_parse(raw) or {"detections": _recover_partial_detections(raw)}
    detections = _dedupe_detections(parsed.get("detections", []))
    areas = []
    for det in detections:
        bbox_norm = det.get("bbox", [])
        if len(bbox_norm) != 4:
            continue
        try:
            bbox_norm_f = [float(c) for c in bbox_norm]
        except (TypeError, ValueError):
            continue
        ego_poly, _ = _bbox_to_circle_polygon(
            bbox_norm_f, K, R, t, W_img=W, H_img=H,
        )
        x1 = int(bbox_norm_f[0] / 1000 * W)
        y1 = int(bbox_norm_f[1] / 1000 * H)
        x2 = int(bbox_norm_f[2] / 1000 * W)
        y2 = int(bbox_norm_f[3] / 1000 * H)
        areas.append({
            "label": str(det.get("label", "")),
            "risk": _coerce_float(det.get("risk", 0.0)),
            "reason": str(det.get("reason", "")),
            "bbox_pixels": [x1, y1, x2, y2],
            "bbox_norm": bbox_norm_f,
            "ego_xy_polygon": ego_poly,
        })
    return areas


# ── Frame iteration ───────────────────────────────────────────────────

def iter_frames(manifest: dict, classified_df: pd.DataFrame,
                policy: str, tau: float):
    """Yield (scenario, variant, frame_idx, img_path, K, R, t, speed_mps,
              token, anomaly_score) for every frame to process."""
    log_dir = Path(manifest["defaults"]["log_dir"])
    classified_df = classified_df.set_index(
        ["scenario", "variant", "frame_idx"], drop=False
    )

    for sc in manifest["scenarios"]:
        log = pickle.load(open(log_dir / sc["log"], "rb"))
        for variant in ["clean"] + [
            f"{c['id']}__{b['id']}" for c in manifest["cases"] for b in manifest["behaviors"]
        ]:
            for fi in range(sc["start_frame"], sc["end_frame"] + 1):
                key = (sc["id"], variant, fi)
                if key not in classified_df.index:
                    continue
                row = classified_df.loc[key]
                score = float(row.anomaly_score)

                if policy == "cascade" and score <= tau:
                    yield {
                        "skipped_by_cascade": True,
                        "scenario": sc["id"], "variant": variant, "frame_idx": fi,
                        "anomaly_score": score,
                    }
                    continue

                frame = log[fi]
                cam = frame["cams"]["CAM_F0"]
                speed_mps = float(np.linalg.norm(frame["ego_dynamic_state"][2:4]))
                if variant == "clean":
                    img_path = SENSOR_ROOT / cam["data_path"]
                else:
                    img_path = BENCHMARK_DIR / sc["id"] / variant / f"f{fi:04d}.jpg"
                cmd_idx = int(np.argmax(np.asarray(
                    frame.get("driving_command", [0, 1, 0, 0]))))
                yield {
                    "skipped_by_cascade": False,
                    "scenario": sc["id"], "variant": variant, "frame_idx": fi,
                    "img_path": str(img_path),
                    "speed_mps": speed_mps,
                    "K": np.array(cam["cam_intrinsic"], dtype=np.float64),
                    "R": np.array(cam["sensor2lidar_rotation"], dtype=np.float64),
                    "t": np.array(cam["sensor2lidar_translation"], dtype=np.float64),
                    "token": frame["token"],
                    "anomaly_score": score,
                    "driving_command_idx": cmd_idx,
                }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["always_on", "cascade"], required=True)
    ap.add_argument("--tau", type=float, default=0.0962,
                    help="Cascade threshold on anomaly_score; ignored for always_on.")
    ap.add_argument("--out_root", default="/workspace/CoachDrive/exp/vlm_dump")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--classified_csv", default=CLASSIFIED_CSV)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a (scenario, variant) if its pkl already exists.")
    ap.add_argument("--scenario_filter", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root) / args.policy
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[vlm] policy={args.policy}  tau={args.tau if args.policy=='cascade' else 'n/a'}")
    print(f"[vlm] manifest={args.manifest}")
    print(f"[vlm] classified={args.classified_csv}")
    manifest = yaml.safe_load(open(args.manifest))
    classified_df = pd.read_csv(args.classified_csv)
    print(f"[vlm] classified rows: {len(classified_df)}")

    if args.scenario_filter:
        keep = set(s.strip() for s in args.scenario_filter.split(","))
        manifest["scenarios"] = [s for s in manifest["scenarios"] if s["id"] in keep]
        classified_df = classified_df[classified_df.scenario.isin(keep)]
        print(f"[vlm] filtered to scenarios: {sorted(keep)}")

    print(f"[vlm] loading Cosmos-Reason2 ...")
    model, processor = get_cosmos()

    # Open calls.csv for streaming write
    calls_csv = open(out_root / "calls.csv", "w", newline="")
    csv_writer = csv.writer(calls_csv)
    csv_writer.writerow([
        "scenario", "variant", "frame_idx", "token", "anomaly_score",
        "called", "prompt_tokens", "completion_tokens", "total_tokens",
        "latency_ms", "n_areas",
    ])

    # Per-(scenario, variant) result dict, dumped after each (sc, var) finishes
    per_var: dict[tuple, dict] = {}

    def flush_var(sc: str, var: str):
        if (sc, var) not in per_var:
            return
        sc_dir = out_root / sc
        sc_dir.mkdir(parents=True, exist_ok=True)
        with open(sc_dir / f"{var}.pkl", "wb") as fh:
            pickle.dump(per_var[(sc, var)], fh)

    n_called = 0
    n_skipped_cascade = 0
    sum_tokens = 0
    sum_latency = 0.0
    t_start = time.time()

    frames = list(iter_frames(manifest, classified_df, args.policy, args.tau))
    n_total = len(frames)
    print(f"[vlm] {n_total} frame iterations")

    last_key = None
    n_skipped_existing = 0
    for i, f in enumerate(frames):
        sc = f["scenario"]
        var = f["variant"]
        key = (sc, var)

        # --skip_existing: if pkl already exists for this (sc, var), skip the
        # entire frame (no image read, no VLM call). last_key stays put so the
        # next active key still triggers flush of the previous active key.
        if args.skip_existing and (out_root / sc / f"{var}.pkl").exists():
            n_skipped_existing += 1
            continue

        # If we just moved to a new (scenario, variant), flush the previous one
        if last_key is not None and last_key != key:
            flush_var(*last_key)
        last_key = key

        if f["skipped_by_cascade"]:
            n_skipped_cascade += 1
            csv_writer.writerow([sc, var, f["frame_idx"], "", f["anomaly_score"],
                                 0, 0, 0, 0, 0, 0])
            # Still record an empty entry per frame so rerank knows "no risk"
            d = per_var.setdefault(key, {})
            d[f.get("token", "")] = {"areas": [], "raw": None,
                                      "tokens": None, "latency_ms": 0.0,
                                      "skipped_by_cascade": True}
            continue

        img = cv2.imread(f["img_path"])
        if img is None:
            print(f"[vlm] missing {f['img_path']}, skip")
            continue
        H, W = img.shape[:2]
        prompt = PROMPT_TEMPLATE.format(
            speed_mps=f["speed_mps"], speed_kph=f["speed_mps"] * 3.6,
            behavior=DRIVING_CMD_MAP[f["driving_command_idx"]])
        try:
            res = call_vlm_with_tokens(model, processor, img, prompt,
                                       max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"  FAILED {sc}/{var}/f{f['frame_idx']}: {e}")
            continue
        areas = parse_and_backproject(res["raw"], f["K"], f["R"], f["t"], W, H)
        n_called += 1
        sum_tokens += res["total_tokens"]
        sum_latency += res["latency_ms"] / 1000.0

        d = per_var.setdefault(key, {})
        d[f["token"]] = {
            "areas": areas, "raw": res["raw"],
            "tokens": {"prompt": res["prompt_tokens"],
                       "completion": res["completion_tokens"],
                       "total": res["total_tokens"]},
            "latency_ms": res["latency_ms"],
            "frame_idx": f["frame_idx"], "speed_mps": f["speed_mps"],
            "anomaly_score": f["anomaly_score"],
        }
        csv_writer.writerow([sc, var, f["frame_idx"], f["token"],
                             f["anomaly_score"], 1,
                             res["prompt_tokens"], res["completion_tokens"],
                             res["total_tokens"], int(res["latency_ms"]), len(areas)])

        if (n_called % 50) == 0:
            elapsed = time.time() - t_start
            done_frac = (i + 1) / max(1, n_total)
            eta = elapsed / done_frac - elapsed
            print(f"  [{i + 1}/{n_total} {100*done_frac:5.1f}%] called={n_called} "
                  f"skipped_cascade={n_skipped_cascade} tokens={sum_tokens} "
                  f"vlm_seconds={sum_latency:.0f} elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min")

    calls_csv.close()

    # Final flush (last variant) + safety re-save of all
    if last_key is not None:
        flush_var(*last_key)
    for (sc, var), data in per_var.items():
        sc_dir = out_root / sc
        sc_dir.mkdir(parents=True, exist_ok=True)
        with open(sc_dir / f"{var}.pkl", "wb") as fh:
            pickle.dump(data, fh)

    elapsed = time.time() - t_start
    summary = (
        f"policy           = {args.policy}\n"
        f"tau (cascade)    = {args.tau}\n"
        f"frames iterated  = {n_total}\n"
        f"VLM calls        = {n_called}\n"
        f"cascade skipped  = {n_skipped_cascade}\n"
        f"existing skipped = {n_skipped_existing}\n"
        f"total tokens     = {sum_tokens}\n"
        f"mean tokens/call = {sum_tokens/max(1,n_called):.1f}\n"
        f"total VLM secs   = {sum_latency:.1f}\n"
        f"mean latency_ms  = {1000*sum_latency/max(1,n_called):.1f}\n"
        f"wall clock min   = {elapsed/60:.1f}\n"
    )
    with open(out_root / "_summary.txt", "w") as fh:
        fh.write(summary)
    print("\n[vlm] === SUMMARY ===")
    print(summary)
    print(f"[vlm] outputs under {out_root}")


if __name__ == "__main__":
    main()
