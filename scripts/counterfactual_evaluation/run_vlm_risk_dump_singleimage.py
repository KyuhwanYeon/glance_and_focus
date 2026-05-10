"""Per-frame single-image VLM risk dump.

For each target frame f_i:
  1. Read the original 1920x1080 F0 camera image (no video, no clip).
  2. Read ego speed from f_i's ego_dynamic_state and pass it to the VLM as text.
  3. Ask the VLM (lightweight call, no <think> reasoning) for hazards in the
     image, with a 2D bbox per hazard, plus a risk score that reflects the
     ego speed (a stationary pedestrian when ego is fast = high risk; same
     pedestrian when ego is parked = lower risk).
  4. Back-project each bbox to ego BEV polygon for the rerank pipeline.

Why single-image:
  Cosmos-Reason2 video mode is unable to do per-frame spatial localization --
  it anchors a single bbox to the moment the object first becomes visible
  (or hallucinates many narrow bboxes at low resolution). Single-image
  localization is reliable per frame. We compensate for the missing motion
  context by feeding ego speed (and direction-of-travel) as text.

Output schema matches `run_vlm_risk_dump.py`:
  {token: {'raw': str, 'image_size': [W,H], 'log_frame_idx': int,
           'areas': [{'label','risk','reason','bbox_pixels','ego_xy_polygon',
                      'ego_speed_mps'}]}}
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from cosmos_vlm import get_cosmos                                     # noqa: E402
from cosmos_risk_to_bev import (                                      # noqa: E402
    project_polygon_to_ego, _denormalize,
    make_bev_canvas, draw_risk_polygons_bev,
)
from vlm_prompt import PROMPT_TEMPLATE, DRIVING_CMD_MAP                # noqa: E402, F401


def _strip_md(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _try_parse(text: str):
    text = _strip_md(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for op, cl in [("{", "}"), ("[", "]")]:
        i = text.find(op)
        if i < 0:
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == op:
                    depth += 1
                elif ch == cl:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            break
    return None


def _recover_partial_detections(text: str) -> list:
    """Even if the outer JSON is truncated mid-detection (max_new_tokens hit),
    recover all complete `{...}` objects inside the detections array."""
    text = _strip_md(text)
    m = re.search(r'"detections"\s*:\s*\[', text)
    start = m.end() if m else text.find("[") + 1
    if start <= 0:
        return []
    detections: list = []
    i = start
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            break
        depth, in_str, esc, j, end_j = 0, False, False, i, -1
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_j = j; break
            j += 1
        if end_j < 0:
            break  # incomplete object - stop
        try:
            obj = json.loads(text[i:end_j + 1])
            if isinstance(obj, dict):
                detections.append(obj)
        except json.JSONDecodeError:
            pass
        i = end_j + 1
    return detections


def _dedupe_detections(detections: list, iou_thresh: float = 0.7) -> list:
    """Drop near-duplicate detections (model frequently repeats the same bbox).
    Keep the first occurrence of each (label, bbox-cluster)."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = ua + ub - inter
        return inter / denom if denom > 0 else 0.0
    kept: list = []
    for d in detections:
        bb = d.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        try:
            bb = [float(c) for c in bb]
        except (TypeError, ValueError):
            continue
        is_dup = False
        for k in kept:
            if k.get("label") != d.get("label"):
                continue
            if iou(bb, k["_bbox"]) >= iou_thresh:
                is_dup = True; break
        if not is_dup:
            d2 = dict(d); d2["_bbox"] = bb
            kept.append(d2)
    for k in kept:
        k.pop("_bbox", None)
    return kept


def _risk_color(r: float):
    r = float(np.clip(r, 0.0, 1.0))
    if r < 0.05:
        return (160, 160, 160)
    if r < 0.5:
        return (60, 200, 220)
    return (40, 40, 230)


def _draw_camera_overlay(img: np.ndarray, areas: list) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for a in areas:
        risk = float(a.get("risk", 0.5))
        col = _risk_color(risk)
        bbox = a.get("bbox_pixels")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 4)
            tag = f"{str(a.get('label','?'))[:22]} ({risk:.2f})"
            cv2.putText(out, tag, (x1, max(28, y1 - 8)),
                        font, 0.9, col, 2, cv2.LINE_AA)
    return out


def _wrap_text(text: str, max_chars: int):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + 1 + len(w) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _make_text_strip(areas: list, speed_mps: float, width: int, height: int = 160):
    strip = np.full((height, width, 3), 18, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(strip, f"ego speed = {speed_mps:.1f} m/s ({speed_mps*3.6:.0f} km/h)   "
                f"Cosmos detected {len(areas)} object(s):",
                (12, 22), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if not areas:
        cv2.putText(strip, "(no hazards in this frame)",
                    (12, 50), font, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        return strip
    y = 48
    max_chars = max(50, width // 11)
    for i, a in enumerate(areas[:4]):
        risk = float(a.get("risk", 0.5))
        col = _risk_color(risk)
        head = f"  [{i+1}] {str(a.get('label','?'))[:28]}  risk={risk:.2f}"
        cv2.putText(strip, head, (12, y), font, 0.5, col, 1, cv2.LINE_AA)
        y += 18
        for line in _wrap_text(str(a.get("reason", ""))[:220], max_chars)[:2]:
            cv2.putText(strip, "    " + line, (12, y), font, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
            y += 16
            if y > height - 8:
                break
        if y > height - 8:
            break
    return strip


def _render_frame_composite(img: np.ndarray, areas: list, log_idx: int,
                              speed_mps: float) -> np.ndarray:
    cam_overlay = _draw_camera_overlay(img, areas)
    cam_small = cv2.resize(cam_overlay, (960, 540))
    bev, ego_to_px = make_bev_canvas(panel_h=540, panel_w=360,
                                       x_max_fwd=40.0, y_half_lat=15.0)
    bev_input = [{"label": a.get("label", "?"),
                   "risk":  a.get("risk", 0.5),
                   "ego_xy": a.get("ego_xy_polygon") or []}
                  for a in areas]
    bev = draw_risk_polygons_bev(bev, ego_to_px, bev_input, alpha=0.55)
    composite = np.hstack([cam_small, bev])
    cv2.putText(composite, f"f{log_idx:03d}  ego={speed_mps:.1f}m/s",
                 (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                 (255, 255, 255), 2, cv2.LINE_AA)
    text_strip = _make_text_strip(areas, speed_mps,
                                    width=composite.shape[1], height=160)
    return np.vstack([composite, text_strip])


def call_cosmos_image_light(model, processor, img_bgr: np.ndarray,
                             prompt: str, max_new_tokens: int = 512) -> str:
    """Bypass the heavy reasoning + JSON enforcer wrapper. Returns raw answer."""
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
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--sensor_blobs", default="/dataset/sensor_blobs/test")
    ap.add_argument("--scene_token", default=None)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug_dir", default=None,
                    help="optional dir to dump per-frame composite jpg + raw response")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    log_frames = pickle.load(open(args.log, "rb"))
    target = list(enumerate(log_frames))
    if args.scene_token:
        target = [(i, f) for i, f in target if f["scene_token"] == args.scene_token]
    if args.start_idx is not None:
        target = [(i, f) for i, f in target if i >= args.start_idx]
    if args.end_idx is not None:
        target = [(i, f) for i, f in target if i <= args.end_idx]
    if args.frame_stride > 1:
        target = target[::args.frame_stride]

    n = len(target)
    print(f"[vlm-img-dump] {n} target frames")
    if n == 0:
        print("[vlm-img-dump] nothing to do"); return

    sensor_blobs = Path(args.sensor_blobs)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    model, processor = get_cosmos()

    out_data: dict = {}
    t_total = time.time()
    for tgt_i, (log_idx, frame) in enumerate(target):
        token = frame["token"]
        cam = frame["cams"]["CAM_F0"]
        K = np.array(cam["cam_intrinsic"], dtype=np.float64)
        R = np.array(cam["sensor2lidar_rotation"], dtype=np.float64)
        t = np.array(cam["sensor2lidar_translation"], dtype=np.float64)
        img_path = sensor_blobs / cam["data_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{tgt_i+1:3d}/{n}] f{log_idx:03d} read FAILED")
            continue
        H, W = img.shape[:2]

        ego_dynamic = frame.get("ego_dynamic_state") or [0.0, 0.0, 0.0, 0.0]
        speed_mps = float(ego_dynamic[0]) if len(ego_dynamic) >= 1 else 0.0
        prompt = PROMPT_TEMPLATE.format(
            speed_mps=speed_mps, speed_kph=speed_mps * 3.6)

        t0 = time.time()
        try:
            answer = call_cosmos_image_light(
                model, processor, img, prompt, args.max_new_tokens)
        except Exception as e:
            print(f"  [{tgt_i+1:3d}/{n}] f{log_idx:03d} cosmos FAILED: {e}")
            continue
        dt = time.time() - t0

        if debug_dir:
            (debug_dir / f"f{log_idx:04d}_response.txt").write_text(
                f"=== EGO SPEED: {speed_mps:.2f} m/s ===\n"
                f"=== ANSWER ===\n{answer}\n")

        parsed = _try_parse(answer)
        detections = []
        if isinstance(parsed, dict):
            detections = parsed.get("detections") or parsed.get("hazards") or []
        elif isinstance(parsed, list):
            detections = parsed
        # Fallback for truncated JSON (max_new_tokens hit mid-detection):
        # recover complete detection objects.
        if not detections:
            detections = _recover_partial_detections(answer)
        n_raw = len(detections)
        detections = _dedupe_detections(detections)
        if n_raw != len(detections):
            print(f"      ! deduped {n_raw} -> {len(detections)} detections")

        new_areas = []
        for d in detections:
            if not isinstance(d, dict):
                continue
            bbox_raw = d.get("bbox")
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue
            try:
                bb_floats = [float(c) for c in bbox_raw]
            except (TypeError, ValueError):
                continue
            if max(abs(c) for c in bb_floats) < 30:
                # 3D / ego coords slipping through; reject
                continue
            ego_xy, _ = project_polygon_to_ego(bb_floats, K, R, t, W, H)
            bbox_px = _denormalize(bb_floats, W, H)
            new_areas.append({
                "label":          str(d.get("label", "?")),
                "risk":           float(d.get("risk", 0.5)),
                "reason":         str(d.get("reason", "")),
                "bbox_pixels":    bbox_px,
                "ego_xy_polygon": ego_xy,
                "ego_speed_mps":  speed_mps,
            })

        out_data[token] = {
            "raw":           answer,
            "image_size":    [W, H],
            "log_frame_idx": log_idx,
            "ego_speed_mps": speed_mps,
            "areas":         new_areas,
        }

        if debug_dir:
            composite = _render_frame_composite(img, new_areas, log_idx, speed_mps)
            cv2.imwrite(str(debug_dir / f"f{log_idx:04d}.jpg"),
                        composite, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        labels = [a["label"] for a in new_areas]
        risks = [round(a["risk"], 2) for a in new_areas]
        print(f"  [{tgt_i+1:3d}/{n}] f{log_idx:03d} ego={speed_mps:.1f}m/s "
              f"{dt:5.1f}s n={len(new_areas)} labels={labels} risks={risks}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out_data, open(args.out, "wb"))
    n_with_risk = sum(1 for v in out_data.values() if v["areas"])
    elapsed = time.time() - t_total
    print(f"\n[vlm-img-dump] dumped {len(out_data)} tokens "
          f"({n_with_risk} with risk areas) in {elapsed:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
