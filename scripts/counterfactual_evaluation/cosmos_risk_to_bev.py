"""Ask Cosmos-Reason2 to enumerate RISKY GROUND AREAS as polygons, then
back-project each polygon vertex to ego frame on the ground plane and
render a top-down BEV map of the risk zones.

Output (per frame):
  exp/vlm_sanity/cosmos_risk_bev/f{NNN}_overlay.jpg  # camera | BEV side-by-side
  exp/vlm_sanity/cosmos_risk_bev/f{NNN}_summary.json # parsed polygons + ego xy
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

from cosmos_localize_test import (  # noqa: E402
    _denormalize, _extract_json, pixel_to_ego_ground,
)


_RISK_AREA_PROMPT = """List EVERY visible object on or near the road in this image, including:
  - all persons / pedestrians (any size, any pose, any location, partially-occluded counts)
  - cyclists, motorcyclists, animals
  - traffic cones, debris, potholes, road damage, construction items
  - vehicles that are stopped or parked in or beside driving lanes

For each object output a JSON entry:
  {
    "label": "<short noun>",
    "bbox":  [x1, y1, x2, y2],   // pixel coords; Qwen-VL 0-1000 normalized is fine
    "footprint": [[u1,v1], ...], // OPTIONAL ground-contact polygon
    "risk":  <float 0..1>,        // 1.0=collision imminent, 0.7=in path, 0.4=on shoulder
    "reason": "<short sentence>"
  }

Return {"areas": [...]}. If you see nothing relevant, return {"areas": []}.
DO NOT skip pedestrians, no matter how small or far. ASCII only, no markdown.
"""


def call_cosmos_risk_areas(img_bgr, prompt=_RISK_AREA_PROMPT,
                            max_new_tokens=2048):
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
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    parsed = _extract_json(cleaned) or _extract_json(response)
    return response, parsed


def _bbox_to_circle_polygon(bbox_norm, K, R_cam2lidar, t_cam2lidar,
                              W_img, H_img, n_pts: int = 24,
                              fallback_radius: float = 0.6,
                              safety_margin_m: float = 0.8):
    """Convert a 2D image bbox to a ground-footprint polygon in ego frame.

    Strategy: back-project the bbox bottom-center (= ground contact point) to
    ego (x_e, y_e) on z=0. Estimate footprint radius from the bbox's pixel
    width interpreted on the ground plane (project bottom-left & bottom-right
    too); fall back to `fallback_radius` if the width-projection is degenerate.

    Returns (ego_xy_list, pixel_xy_list)."""
    bb_p = _denormalize([float(c) for c in bbox_norm], W_img, H_img)
    x1, y1, x2, y2 = bb_p
    u_c, v_b = (x1 + x2) / 2.0, max(y1, y2)
    bp = pixel_to_ego_ground(u_c, v_b, K, R_cam2lidar, t_cam2lidar)
    if bp is None:
        return [], []
    x_e, y_e = bp
    # estimate radius from bbox width on the ground
    bp_l = pixel_to_ego_ground(x1, v_b, K, R_cam2lidar, t_cam2lidar)
    bp_r = pixel_to_ego_ground(x2, v_b, K, R_cam2lidar, t_cam2lidar)
    if bp_l is not None and bp_r is not None:
        base_r = max(0.2, 0.5 * float(np.hypot(bp_l[0] - bp_r[0],
                                                 bp_l[1] - bp_r[1])))
    else:
        base_r = fallback_radius
    # Apply safety margin so the BEV "avoid zone" is wider than the bare
    # hazard footprint -- this is the radius used for trajectory penalty.
    radius = base_r + safety_margin_m
    th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ego_xy = [(x_e + radius * float(np.cos(a)),
                y_e + radius * float(np.sin(a))) for a in th]
    return ego_xy, [(u_c, v_b)]


def project_polygon_to_ego(footprint_pixels, K, R_cam2lidar, t_cam2lidar,
                             W_img, H_img):
    """`footprint_pixels` may be:
       (a) list of [u, v] pairs       -> polygon
       (b) [x1,y1,x2,y2] flat list    -> bbox
       (c) [[x1,y1,x2,y2]] nested     -> bbox
    For polygon, back-project each vertex on z=0. For bbox, back-project
    bbox bottom-center and synthesize a circular footprint sized from the
    bbox's projected width.
    """
    if not footprint_pixels:
        return [], []
    flat_bbox = (
        isinstance(footprint_pixels, (list, tuple)) and len(footprint_pixels) == 4
        and all(isinstance(c, (int, float)) for c in footprint_pixels)
    )
    nested_bbox = (
        isinstance(footprint_pixels, (list, tuple)) and len(footprint_pixels) == 1
        and isinstance(footprint_pixels[0], (list, tuple))
        and len(footprint_pixels[0]) == 4
    )
    if flat_bbox:
        return _bbox_to_circle_polygon(footprint_pixels, K, R_cam2lidar,
                                         t_cam2lidar, W_img, H_img)
    if nested_bbox:
        return _bbox_to_circle_polygon(footprint_pixels[0], K, R_cam2lidar,
                                         t_cam2lidar, W_img, H_img)
    # Polygon path
    ego_xy = []
    pixel_xy = []
    for pt in footprint_pixels:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        upx, vpx = _denormalize([float(pt[0]), float(pt[1])], W_img, H_img)
        bp = pixel_to_ego_ground(upx, vpx, K, R_cam2lidar, t_cam2lidar)
        if bp is None:
            continue
        ego_xy.append(bp)
        pixel_xy.append((upx, vpx))
    return ego_xy, pixel_xy


# ---------------------------------------------------------------------------
# BEV rendering
# ---------------------------------------------------------------------------

def make_bev_canvas(panel_h: int = 540, panel_w: int = 400,
                     x_max_fwd: float = 30.0, y_half_lat: float = 12.0):
    canvas = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
    cx = panel_w // 2
    cy = panel_h - 30
    px_per_m_fwd = (cy - 10) / x_max_fwd
    px_per_m_lat = (panel_w / 2 - 10) / y_half_lat

    def ego_to_px(x_ego: float, y_ego: float):
        return (int(cx - y_ego * px_per_m_lat),
                int(cy - x_ego * px_per_m_fwd))

    font = cv2.FONT_HERSHEY_SIMPLEX
    # range rings (forward)
    for d in (5, 10, 15, 20, 25, 30):
        if d > x_max_fwd:
            break
        _, v = ego_to_px(d, 0)
        cv2.line(canvas, (10, v), (panel_w - 10, v), (60, 60, 60), 1)
        cv2.putText(canvas, f"{d}m", (panel_w - 38, v + 4), font, 0.38,
                    (140, 140, 140), 1, cv2.LINE_AA)
    # lane lines (visual reference: +/- 1.75 m, +/- 5.25 m for 3-lane road)
    for ywall in (-5.25, -1.75, 1.75, 5.25):
        u_top, _ = ego_to_px(x_max_fwd, ywall)
        u_bot, _ = ego_to_px(0, ywall)
        cv2.line(canvas, (u_top, 10), (u_bot, cy), (50, 50, 50), 1, cv2.LINE_AA)
    # ego triangle
    ego_pts = np.array([[cx, cy - 14], [cx - 9, cy + 4], [cx + 9, cy + 4]],
                        dtype=np.int32)
    cv2.fillPoly(canvas, [ego_pts], (80, 220, 80))
    cv2.polylines(canvas, [ego_pts], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "BEV (top-down)", (10, 18), font, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas, ego_to_px


def draw_risk_polygons_bev(canvas, ego_to_px, areas, alpha=0.55):
    """areas: list of {'label', 'ego_xy': [(x,y), ...], 'risk': float}."""
    H, W = canvas.shape[:2]
    overlay = canvas.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for a in areas:
        ego_xy = a.get("ego_xy") or []
        if len(ego_xy) < 3:
            # single-point hazard: draw a circle
            if ego_xy:
                cx_b, cy_b = ego_to_px(*ego_xy[0])
                r = a.get("risk", 0.5)
                color = _risk_color(r)
                cv2.circle(overlay, (cx_b, cy_b), 12, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx_b, cy_b), 12, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        poly_px = np.array([ego_to_px(x, y) for x, y in ego_xy], dtype=np.int32)
        # clamp obviously bad
        if (poly_px < -2000).any() or (poly_px > 4000).any():
            continue
        risk = float(a.get("risk", 0.5))
        color = _risk_color(risk)
        cv2.fillPoly(overlay, [poly_px], color)
        cv2.polylines(canvas, [poly_px], True, (255, 255, 255), 1, cv2.LINE_AA)
        # label at centroid
        cx_b = int(poly_px[:, 0].mean())
        cy_b = int(poly_px[:, 1].mean())
        tag = f"{str(a.get('label', '?'))[:18]} ({risk:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, font, 0.42, 1)
        x0 = max(2, min(W - tw - 4, cx_b - tw // 2))
        y0 = max(th + 4, min(H - 2, cy_b))
        cv2.rectangle(canvas, (x0 - 2, y0 - th - 2), (x0 + tw + 2, y0 + 2),
                      (0, 0, 0), -1)
        cv2.putText(canvas, tag, (x0, y0), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)


def _risk_color(r: float):
    """Yellow (low) -> orange -> red (high), in BGR."""
    r = float(np.clip(r, 0.0, 1.0))
    g = int(220 * max(0.0, 1.0 - r))
    return (0, g, 220)


def draw_polygons_camera(img, areas, alpha=0.40):
    out = img.copy()
    overlay = out.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for a in areas:
        # Draw the bbox if present (from the VLM raw bbox field)
        bbox = a.get("bbox_pixels")
        color = _risk_color(float(a.get("risk", 0.5)))
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
            tag = f"{str(a.get('label', '?'))[:24]} ({a.get('risk', 0.5):.2f})"
            cv2.putText(out, tag, (x1, max(28, y1 - 10)),
                        font, 1.1, color, 3, cv2.LINE_AA)
        pix = a.get("pixel_xy") or []
        if pix:
            u, v = pix[0]
            cv2.circle(overlay, (int(u), int(v)), 30, color, -1, cv2.LINE_AA)
            cv2.circle(out, (int(u), int(v)), 30, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (int(u), int(v)), 6, (0, 0, 0), -1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--scene_token", required=True)
    ap.add_argument("--frame_offset", type=int, required=True)
    ap.add_argument("--anchor_xy", default="18,0")
    ap.add_argument("--anchor_at_frame", type=int, default=4)
    ap.add_argument("--cone_h", type=float, default=0.7)
    ap.add_argument("--cone_radius", type=float, default=0.3)
    ap.add_argument("--out_dir", default="exp/vlm_sanity/cosmos_risk_bev")
    args = ap.parse_args()

    from gt_motion_overlay import SENSOR_BLOBS
    from cosmos_video_test import SyntheticHazard

    log_data = pickle.load(open(args.log, "rb"))
    sf = [f for f in log_data if f["scene_token"] == args.scene_token]
    cam = sf[args.frame_offset]["cams"]["CAM_F0"]
    K = np.array(cam["cam_intrinsic"], dtype=np.float64)
    R = np.array(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.array(cam["sensor2lidar_translation"], dtype=np.float64)
    ego2global = np.array(sf[args.frame_offset]["ego2global"], dtype=np.float64)

    img = cv2.imread(str(SENSOR_BLOBS / cam["data_path"]))
    H_img, W_img = img.shape[:2]

    # render synthetic cone (so VLM has something to find)
    dx, dy = (float(s) for s in args.anchor_xy.split(","))
    haz = SyntheticHazard(kind="cone", radius_m=args.cone_radius,
                           cone_h=args.cone_h)
    ego2g_anchor = np.array(sf[args.anchor_at_frame]["ego2global"],
                              dtype=np.float64)
    haz.anchor_at_ego(ego2g_anchor, dx, dy)
    img_with_cone = haz.draw_camera(img.copy(), K, R, t, ego2global)

    x_gt, y_gt = haz.to_ego(ego2global)
    print(f"[risk-bev] GT cone ego xy: ({x_gt:.2f}, {y_gt:.2f}) m")

    # ---- call Cosmos for risky areas
    print("[risk-bev] calling Cosmos for risky-area polygons ...")
    raw, parsed = call_cosmos_risk_areas(img_with_cone)
    print(f"[risk-bev] === raw response ===\n{raw}\n=== end raw ===\n")
    if parsed is None:
        print("[risk-bev] !! could not parse JSON — aborting")
        return
    raw_areas = parsed.get("areas") or []
    if isinstance(raw_areas, dict):
        raw_areas = [raw_areas]
    print(f"[risk-bev] parsed {len(raw_areas)} areas")

    # ---- back-project each polygon (fall back to bbox if footprint absent)
    proc_areas = []
    for a in raw_areas:
        footprint = a.get("footprint") or a.get("polygon")
        if not footprint:
            footprint = a.get("bbox")          # fallback
        ego_xy, pixel_xy = project_polygon_to_ego(
            footprint or [], K, R, t, W_img, H_img)
        # also denormalize the raw bbox so we can draw it on the camera
        bbox_raw = a.get("bbox")
        bbox_px = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox_px = _denormalize([float(c) for c in bbox_raw], W_img, H_img)
        proc_areas.append({
            "label":  str(a.get("label", "?")),
            "risk":   float(a.get("risk", 0.5)),
            "reason": str(a.get("reason", "")),
            "ego_xy": ego_xy,
            "pixel_xy": pixel_xy,
            "bbox_pixels": bbox_px,
        })
        print(f"  [{a.get('label','?')}] risk={float(a.get('risk',0)):.2f} | "
              f"ego polygon: " + ", ".join(f"({x:+.1f},{y:+.1f})" for x, y in ego_xy))

    # ---- render
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # camera overlay
    cam_overlay = draw_polygons_camera(img_with_cone, proc_areas, alpha=0.35)
    # mark GT cone center on camera too (white cross)
    from bev_grid import project_lidar_pts
    gt_pix, in_f = project_lidar_pts(np.array([[x_gt, y_gt, 0.0]]), K, R, t)
    if in_f[0]:
        gu, gv = int(gt_pix[0, 0]), int(gt_pix[0, 1])
        cv2.drawMarker(cam_overlay, (gu, gv), (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 4, cv2.LINE_AA)
        cv2.putText(cam_overlay, "GT", (gu + 14, gv - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # downsample camera so the side-by-side composite is reasonable
    cam_small = cv2.resize(cam_overlay, (800, 450))

    # BEV
    bev, ego_to_px = make_bev_canvas(panel_h=450, panel_w=320,
                                       x_max_fwd=30.0, y_half_lat=12.0)
    # mark GT cone on BEV
    gx_px, gy_px = ego_to_px(float(x_gt), float(y_gt))
    cv2.drawMarker(bev, (gx_px, gy_px), (0, 0, 255), cv2.MARKER_CROSS,
                   16, 2, cv2.LINE_AA)
    bev = draw_risk_polygons_bev(bev, ego_to_px, proc_areas, alpha=0.55)

    composite = np.hstack([cam_small, bev])
    out_img = out_dir / f"f{args.frame_offset:03d}_overlay.jpg"
    cv2.imwrite(str(out_img), composite)
    print(f"\n[risk-bev] overlay -> {out_img}")

    out_json = out_dir / f"f{args.frame_offset:03d}_summary.json"
    out_json.write_text(json.dumps({
        "frame_offset": args.frame_offset,
        "image_size": [W_img, H_img],
        "gt_ego_xy": [x_gt, y_gt],
        "vlm_raw": raw,
        "areas": [{
            "label": a["label"], "risk": a["risk"], "reason": a["reason"],
            "ego_xy": a["ego_xy"],
        } for a in proc_areas],
    }, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "item") else o))
    print(f"[risk-bev] summary -> {out_json}")


if __name__ == "__main__":
    main()
