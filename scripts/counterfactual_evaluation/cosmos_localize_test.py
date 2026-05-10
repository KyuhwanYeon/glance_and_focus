"""Probe Cosmos-Reason2's 2D/3D localization output.

Asks Cosmos for each physical hazard's bbox + ground-contact pixel, parses
JSON, back-projects the ground pixel to ego-frame (x, y) using ground-plane
assumption (z=0), and compares to the known synthetic-hazard position.

Reports:
  - raw VLM JSON
  - per-hazard bbox + ground-point overlay on the image (saved as PNG)
  - back-projected (x_e, y_e) vs ground truth + Euclidean error
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


_LOCALIZE_PROMPT = """Identify every PHYSICAL HAZARD on or near the road that the ego vehicle should avoid (debris, traffic cones, potholes, fallen objects, animals, pedestrians in danger zones, sharp obstacles, etc.). Do NOT include normal moving traffic / parked cars unless they pose an immediate risk.

For EACH hazard, output a JSON object with these fields:
  - "label": short noun phrase
  - "bbox":  [x1, y1, x2, y2] PIXEL coordinates (top-left -> bottom-right)
  - "ground_point": [u, v] PIXEL of the GROUND CONTACT POINT where the hazard meets the road
  - "reason": one short sentence

Return a JSON object with key "hazards" mapping to a list of the above. If no hazards, return {"hazards": []}.
ASCII only. No markdown fences. No prose outside the JSON.
"""


def _extract_json(text: str):
    """Find the largest balanced JSON object in `text` and parse it."""
    cands = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < len(text):
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
                        cands.append(text[i:j + 1])
                        break
            j += 1
        i = j + 1 if depth == 0 else i + 1
    cands.sort(key=len, reverse=True)
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def call_cosmos_localize(img_bgr, prompt=_LOCALIZE_PROMPT, max_new_tokens=1024):
    from cosmos_vlm import get_cosmos
    import torch
    from PIL import Image

    model, processor = get_cosmos()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    messages = [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You are a driving safety perception system. You output precise pixel coordinates."}]},
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
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
    gen_only = gen_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    # strip <think> if present
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    parsed = _extract_json(cleaned) or _extract_json(response)
    return response, parsed


def pixel_to_ego_ground(u, v, K, R_cam2lidar, t_cam2lidar):
    """Back-project pixel (u,v) to the GROUND PLANE in lidar/ego frame
    (z=0). Returns (x_e, y_e) or None if the ray doesn't hit the ground
    in front of the camera.
    """
    Kinv = np.linalg.inv(K)
    ray_cam = Kinv @ np.array([u, v, 1.0])
    ray_lidar = R_cam2lidar @ ray_cam
    # Ground intersection: t_cam2lidar.z + s * ray_lidar.z = 0
    if abs(ray_lidar[2]) < 1e-6:
        return None
    s = -t_cam2lidar[2] / ray_lidar[2]
    if s <= 0:
        return None
    p_lidar = R_cam2lidar @ (s * ray_cam) + t_cam2lidar
    return float(p_lidar[0]), float(p_lidar[1])


def _denormalize(coords, W, H):
    """Cosmos / Qwen3-VL family commonly emit normalized 0-1000 coords. Detect
    by max value and rescale to pixel units. Returns list of floats."""
    arr = [float(c) for c in coords]
    m = max(arr) if arr else 0
    if m <= 1.5:                    # fractional 0..1
        sx, sy = float(W), float(H)
    elif m <= 1100:                 # normalized 0..1000 (Qwen convention)
        sx = sy = 0.001
        # scale: pixel = normalized/1000 * imgsize
        return [arr[i] / 1000.0 * (W if i % 2 == 0 else H) for i in range(len(arr))]
    else:
        return arr                  # already pixels
    return [arr[i] * (sx if i % 2 == 0 else sy) for i in range(len(arr))]


def _bbox_to_ground_point(bbox):
    """Bottom-center of a [x1,y1,x2,y2] bbox as a (u,v) ground contact pixel."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, max(y1, y2))


def _normalize_ground_point(gp, bbox, W, H):
    """Cosmos sometimes returns a 4-element bbox where a 2-element point
    was asked, or vice versa. Reduce to a single (u,v) pixel pair."""
    if isinstance(gp, (list, tuple)):
        gp_p = _denormalize(gp, W, H)
        if len(gp_p) == 2:
            return tuple(gp_p)
        if len(gp_p) == 4:           # treat as tiny bbox -> bottom-center
            return _bbox_to_ground_point(gp_p)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bb_p = _denormalize(bbox, W, H)
        return _bbox_to_ground_point(bb_p)
    return None


def draw_overlays(img, hazards, gt_pixel=None):
    out = img.copy()
    H_img, W_img = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, h in enumerate(hazards):
        b = h.get("bbox")
        gp = h.get("ground_point")
        label = str(h.get("label", "?"))[:30]
        color = (60, 220, 60) if i == 0 else (60, 60, 220)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            bb_p = _denormalize(b, W_img, H_img)
            x1, y1, x2, y2 = (int(round(v)) for v in bb_p)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, label, (x1, max(20, y1 - 8)), font, 1.0,
                        color, 2, cv2.LINE_AA)
        gp_resolved = _normalize_ground_point(gp, b, W_img, H_img)
        if gp_resolved is not None:
            u, v = (int(round(c)) for c in gp_resolved)
            cv2.circle(out, (u, v), 12, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(out, (u, v), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(out, "GP", (u + 14, v + 6), font, 0.9,
                        (0, 255, 255), 2, cv2.LINE_AA)
    if gt_pixel is not None:
        u, v = (int(round(c)) for c in gt_pixel)
        cv2.drawMarker(out, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 4,
                       cv2.LINE_AA)
        cv2.putText(out, "GT", (u + 14, v - 14), font, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--scene_token", required=True)
    ap.add_argument("--frame_offset", type=int, required=True)
    ap.add_argument("--anchor_xy", default="18,0",
                    help="ego-frame (dx,dy) anchor point at the FIRST processed "
                         "frame (matching the cone-only run)")
    ap.add_argument("--anchor_at_frame", type=int, default=4,
                    help="frame_offset where the cone was anchored")
    ap.add_argument("--cone_h", type=float, default=0.7)
    ap.add_argument("--cone_radius", type=float, default=0.3)
    ap.add_argument("--out_dir", default="exp/vlm_sanity/cosmos_localize")
    args = ap.parse_args()

    from gt_motion_overlay import SENSOR_BLOBS
    from bev_grid import project_lidar_pts
    from cosmos_video_test import SyntheticHazard

    # 1) load frame + cone
    log_data = pickle.load(open(args.log, "rb"))
    sf = [f for f in log_data if f["scene_token"] == args.scene_token]
    cam = sf[args.frame_offset]["cams"]["CAM_F0"]
    K = np.array(cam["cam_intrinsic"], dtype=np.float64)
    R = np.array(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.array(cam["sensor2lidar_translation"], dtype=np.float64)
    ego2global = np.array(sf[args.frame_offset]["ego2global"], dtype=np.float64)

    img = cv2.imread(str(SENSOR_BLOBS / cam["data_path"]))
    H_img, W_img = img.shape[:2]

    # render cone (anchored at args.anchor_at_frame's ego pose)
    dx, dy = (float(s) for s in args.anchor_xy.split(","))
    haz = SyntheticHazard(kind="cone", radius_m=args.cone_radius, cone_h=args.cone_h)
    ego2g_anchor = np.array(sf[args.anchor_at_frame]["ego2global"], dtype=np.float64)
    haz.anchor_at_ego(ego2g_anchor, dx, dy)
    img_with_cone = haz.draw_camera(img.copy(), K, R, t, ego2global)

    # 2) ground-truth: cone position in ego frame at this frame
    x_gt, y_gt = haz.to_ego(ego2global)
    print(f"[loc] GT cone ego xy: ({x_gt:.2f}, {y_gt:.2f}) m")
    # GT pixel: project cone base
    gt_pix, in_front_gt = project_lidar_pts(
        np.array([[x_gt, y_gt, 0.0]]), K, R, t)
    gt_pixel = tuple(gt_pix[0]) if in_front_gt[0] else None
    if gt_pixel is not None:
        print(f"[loc] GT cone base pixel: ({gt_pixel[0]:.0f}, {gt_pixel[1]:.0f})")

    # 3) call Cosmos
    print(f"[loc] image size: {W_img}x{H_img}")
    print(f"[loc] calling Cosmos for hazard localization ...")
    raw, parsed = call_cosmos_localize(img_with_cone)
    print(f"[loc] === raw response ===\n{raw}\n=== end raw ===\n")
    if parsed is None:
        print("[loc] !! could not parse JSON, aborting")
        return
    hazards = parsed.get("hazards") or parsed.get("Hazards") or []
    if isinstance(hazards, dict):
        hazards = [hazards]
    print(f"[loc] parsed {len(hazards)} hazards: {hazards}")

    # 4) back-project each ground_point + compute error
    print("\n[loc] === back-projection ===")
    rows = []
    for h in hazards:
        gp_raw = h.get("ground_point")
        bb_raw = h.get("bbox")
        gp = _normalize_ground_point(gp_raw, bb_raw, W_img, H_img)
        if gp is None:
            continue
        u, v = float(gp[0]), float(gp[1])
        bp = pixel_to_ego_ground(u, v, K, R, t)
        if bp is None:
            print(f"  [{h.get('label')}] gp=({u:.0f},{v:.0f}) -> ground "
                  "ray doesn't hit z=0 in front of cam")
            continue
        x_e, y_e = bp
        err = float(np.hypot(x_e - x_gt, y_e - y_gt)) if gt_pixel else None
        msg = (f"  [{h.get('label')}] pixel=({u:.0f},{v:.0f}) "
               f"-> ego=({x_e:+6.2f},{y_e:+6.2f}) m  "
               f"|d_to_GT|={'%.2f' % err if err is not None else 'n/a'} m")
        print(msg)
        rows.append({"label": h.get("label"), "ground_pixel": [u, v],
                     "ego_xy": [x_e, y_e], "err_m": err})

    # 5) save overlay + JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlays(img_with_cone, hazards, gt_pixel=gt_pixel)
    img_path = out_dir / f"f{args.frame_offset:03d}_overlay.jpg"
    cv2.imwrite(str(img_path), overlay)
    print(f"\n[loc] overlay -> {img_path}")
    summary = {
        "frame_offset": args.frame_offset,
        "image_size": [W_img, H_img],
        "gt_ego_xy": [x_gt, y_gt],
        "gt_pixel": list(gt_pixel) if gt_pixel else None,
        "vlm_raw": raw,
        "vlm_hazards": hazards,
        "back_projected": rows,
    }
    json_path = out_dir / f"f{args.frame_offset:03d}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                       default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"[loc] summary -> {json_path}")


if __name__ == "__main__":
    main()
