"""Run single-image Cosmos-Reason2 hazard detection on every token in a
saved merged_predictions pkl.

For each token:
  1. Locate the navsim log frame via SceneLoader (built from the same
     scene_filter as scoring) -> CAM_F0 image, K/R/t, ego speed.
  2. Call the lightweight Cosmos-Reason2 wrapper used by
     scripts/counterfactual_evaluation/run_vlm_risk_dump_singleimage.py.
  3. Parse + dedupe + back-project bbox to ego polygon.

Output schema matches run_vlm_risk_dump_singleimage.py so it can be fed
straight into rerank_predictions_with_vlm.py:
  {token: {raw, image_size, log_frame_idx, ego_speed_mps,
           areas:[{label, risk, reason, bbox_pixels, ego_xy_polygon,
                   ego_speed_mps}]}}

Inputs:
  PREDICTIONS_PATH=...      env: merged_predictions pkl
  +vlm_out=...              hydra: output vlm risk pkl path
  train_test_split=...      hydra: same scene_filter as scoring
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader

_DEVKIT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_DEVKIT / "scripts" / "counterfactual_evaluation"))

from cosmos_risk_to_bev import _denormalize, project_polygon_to_ego  # noqa: E402
from cosmos_vlm import get_cosmos                                    # noqa: E402
from run_vlm_risk_dump_singleimage import (                          # noqa: E402
    _dedupe_detections,
    _recover_partial_detections,
    _try_parse,
    call_cosmos_image_light,
)
from vlm_prompt import DRIVING_CMD_MAP, PROMPT_TEMPLATE               # noqa: E402

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    predictions_path = os.environ.get("PREDICTIONS_PATH") or cfg.get("predictions_path")
    if not predictions_path:
        raise SystemExit("set PREDICTIONS_PATH=... or +predictions_path=...")
    vlm_out = os.environ.get("VLM_OUT") or cfg.get("vlm_out")
    if not vlm_out:
        raise SystemExit("set VLM_OUT=... or +vlm_out=...")
    max_new_tokens = int(cfg.get("vlm_max_new_tokens", 512))

    print(f"[vlm-on-preds] loading predictions: {predictions_path}")
    preds = pickle.load(open(predictions_path, "rb"))
    pred_tokens = set(preds.keys())
    print(f"[vlm-on-preds] {len(pred_tokens)} tokens in predictions")

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    eval_set = pred_tokens & set(scene_loader.tokens)
    tokens_by_log = scene_loader.get_tokens_list_per_log()
    print(f"[vlm-on-preds] eval_set={len(eval_set)} (preds ∩ scene)")

    sensor_root = Path(cfg.original_sensor_path)
    log_root = Path(cfg.navsim_log_path)

    print("[vlm-on-preds] loading Cosmos-Reason2...")
    model, processor = get_cosmos()

    out: dict = {}
    t_start = time.time()
    n_done = 0
    total = len(eval_set)
    for log_name, tokens_in_log in tokens_by_log.items():
        wanted = [t for t in tokens_in_log if t in eval_set]
        if not wanted:
            continue
        log_pkl = log_root / f"{log_name}.pkl"
        if not log_pkl.exists():
            print(f"  [warn] log not found: {log_pkl}")
            continue
        log_frames = pickle.load(open(log_pkl, "rb"))
        frame_by_tok = {f["token"]: f for f in log_frames}
        for token in wanted:
            frame = frame_by_tok.get(token)
            if frame is None:
                print(f"  [warn] token {token[:8]} missing in {log_name}")
                continue
            cam = frame["cams"]["CAM_F0"]
            K = np.array(cam["cam_intrinsic"], dtype=np.float64)
            R = np.array(cam["sensor2lidar_rotation"], dtype=np.float64)
            tvec = np.array(cam["sensor2lidar_translation"], dtype=np.float64)
            img_path = sensor_root / cam["data_path"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [warn] image read failed: {img_path}")
                continue
            H, W = img.shape[:2]

            ego_dyn = frame.get("ego_dynamic_state") or [0.0, 0.0, 0.0, 0.0]
            speed_mps = float(ego_dyn[0]) if len(ego_dyn) >= 1 else 0.0
            cmd_idx = int(np.argmax(
                np.asarray(frame.get("driving_command", [0, 1, 0, 0]))
            ))
            behavior = DRIVING_CMD_MAP.get(cmd_idx, "go straight")
            prompt = PROMPT_TEMPLATE.format(
                speed_mps=speed_mps, speed_kph=speed_mps * 3.6,
                behavior=behavior,
            )

            t0 = time.time()
            try:
                ans = call_cosmos_image_light(
                    model, processor, img, prompt, max_new_tokens
                )
            except Exception as e:
                print(f"  [warn] cosmos failed for {token[:8]}: {e}")
                continue
            dt = time.time() - t0

            parsed = _try_parse(ans)
            dets = []
            if isinstance(parsed, dict):
                dets = parsed.get("detections") or parsed.get("hazards") or []
            elif isinstance(parsed, list):
                dets = parsed
            if not dets:
                dets = _recover_partial_detections(ans)
            dets = _dedupe_detections(dets)

            areas = []
            for d in dets:
                bb_raw = d.get("bbox")
                if not isinstance(bb_raw, (list, tuple)) or len(bb_raw) != 4:
                    continue
                try:
                    bb = [float(c) for c in bb_raw]
                except (TypeError, ValueError):
                    continue
                if max(abs(c) for c in bb) < 30:
                    continue  # reject 3D / ego coords slipping through
                ego_xy, _ = project_polygon_to_ego(bb, K, R, tvec, W, H)
                bbox_px = _denormalize(bb, W, H)
                areas.append({
                    "label":          str(d.get("label", "?")),
                    "risk":           float(d.get("risk", 0.5)),
                    "reason":         str(d.get("reason", "")),
                    "bbox_pixels":    bbox_px,
                    "ego_xy_polygon": ego_xy,
                    "ego_speed_mps":  speed_mps,
                })

            out[token] = {
                "raw":           ans,
                "image_size":    [W, H],
                "log_frame_idx": int(frame.get("frame_idx", -1)),
                "ego_speed_mps": speed_mps,
                "areas":         areas,
            }
            n_done += 1
            if n_done % 10 == 0 or n_done == total:
                elapsed = time.time() - t_start
                print(f"  [{n_done:4d}/{total}] {token[:8]} "
                      f"speed={speed_mps:.1f}m/s {dt:4.1f}s "
                      f"areas={len(areas)} elapsed={elapsed:.0f}s")

    Path(vlm_out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(vlm_out, "wb"))
    n_with_risk = sum(1 for v in out.values() if v["areas"])
    print(f"[vlm-on-preds] dumped {n_done} tokens "
          f"({n_with_risk} with risk areas) in {time.time()-t_start:.0f}s "
          f"-> {vlm_out}")


if __name__ == "__main__":
    main()
