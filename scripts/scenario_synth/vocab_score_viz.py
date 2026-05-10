"""Visualize the 8192-anchor trajectory vocab on a camera image, colored by
GTRS-Dense inference score for one token.

For one (scenario, variant, token):
  - load `traj_final/8192.npy`                  -> (8192, 40, 3) anchors in ego frame
  - load `exp/benchmark_pkls/<sc>/<var>.pkl`    -> per-token subscore arrays (8192,)
  - look up the matching frame in the log to grab K, R, t and the F0 image path
  - draw every anchor as a polyline on the image, colored by score (cmap)
  - drop a vertical colorbar on the right

Anchors are drawn lowest-score first so the highest-score (most-likely) trajectories
sit on top.

Usage (navsim env has cv2 + matplotlib + nuplan modules for log loading):

    /opt/conda/envs/navsim/bin/python \\
        scripts/scenario_synth/vocab_score_viz.py \\
        --scenario boston_03 --variant animal_bear__center

By default the end_frame token is used; pass --token <hex> to pick a different one.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

sys.path.insert(0, "/workspace/CoachDrive/scripts/counterfactual_evaluation")
from topk_helper import combined_score                                   # noqa: E402

REPO = Path("/workspace/CoachDrive")
MANIFEST = REPO / "scripts/scenario_synth/benchmark_manifest.yaml"
VOCAB_PATH = REPO / "traj_final/8192.npy"
GTRS_PKLS_ROOT = REPO / "exp/benchmark_pkls"
BENCHMARK_DIR = REPO / "exp/benchmark"
SENSOR_ROOT = Path("/dataset/sensor_blobs/test")
OUT_ROOT = REPO / "exp/vocab_score_viz"

SUBSCORE_KEYS = (
    "imi", "no_at_fault_collisions", "drivable_area_compliance",
    "time_to_collision_within_bound", "ego_progress",
    "driving_direction_compliance", "lane_keeping",
    "traffic_light_compliance",
)


def project_to_image(P_ego: np.ndarray, K, R_cam2lidar, t_cam2lidar):
    P_cam = (P_ego - t_cam2lidar) @ R_cam2lidar
    in_front = P_cam[..., 2] > 0.5
    z = np.where(P_cam[..., 2] > 1e-3, P_cam[..., 2], 1.0)
    px = P_cam @ K.T
    return np.stack([px[..., 0] / z, px[..., 1] / z], axis=-1), in_front


def colors_from_scores(scores: np.ndarray, cmap_name: str,
                       lo_pct: float, hi_pct: float) -> tuple[np.ndarray, float, float]:
    """Map (N,) scores -> (N, 3) BGR uint8 via matplotlib cmap with percentile clipping."""
    lo = float(np.percentile(scores, lo_pct))
    hi = float(np.percentile(scores, hi_pct))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    norm = np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(norm)                                # (N,4) RGBA in [0,1]
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    bgr = rgb[:, ::-1].copy()                        # cv2 wants BGR
    return bgr, lo, hi


def draw_vocab(img: np.ndarray, vocab_xy: np.ndarray, colors_bgr: np.ndarray,
               order: np.ndarray, K, R, t, alpha: float,
               linewidth: int, top1_idx: int | None = None,
               top1_linewidth: int = 6,
               top1_color_bgr: tuple = (0, 0, 230)) -> np.ndarray:
    """Project all anchors and draw polylines on a copy of `img`, blended at `alpha`.

    If `top1_idx` is given, the corresponding anchor is drawn on top of the
    blended image at full opacity with a thicker line + waypoint dots so the
    chosen trajectory pops above the score cloud."""
    H, W = img.shape[:2]
    overlay = img.copy()

    pts3d = np.zeros((*vocab_xy.shape[:2], 3), np.float64)               # (N, 40, 3)
    pts3d[..., :2] = vocab_xy
    flat = pts3d.reshape(-1, 3)
    pixels, in_front = project_to_image(flat, K, R, t)
    pixels = pixels.reshape(*vocab_xy.shape[:2], 2)
    in_front = in_front.reshape(*vocab_xy.shape[:2])

    for idx in order:
        in_f = in_front[idx]
        if not in_f.any():
            continue
        pts = pixels[idx]
        col = (int(colors_bgr[idx, 0]), int(colors_bgr[idx, 1]),
               int(colors_bgr[idx, 2]))
        last = None
        for j in range(pts.shape[0]):
            if not in_f[j]:
                last = None
                continue
            u, v = int(pts[j, 0]), int(pts[j, 1])
            if u < -2000 or u > W + 2000 or v < -2000 or v > H + 2000:
                last = None
                continue
            if last is not None:
                cv2.line(overlay, last, (u, v), col, linewidth, cv2.LINE_AA)
            last = (u, v)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    if top1_idx is not None:
        in_f = in_front[top1_idx]
        pts = pixels[top1_idx]
        col = top1_color_bgr
        # outline (black) for contrast, then top1 color on top
        outline = (0, 0, 0)
        last = None
        for j in range(pts.shape[0]):
            if not in_f[j]:
                last = None
                continue
            u, v = int(pts[j, 0]), int(pts[j, 1])
            if u < -2000 or u > W + 2000 or v < -2000 or v > H + 2000:
                last = None
                continue
            if last is not None:
                cv2.line(blended, last, (u, v), outline,
                         top1_linewidth + 3, cv2.LINE_AA)
                cv2.line(blended, last, (u, v), col,
                         top1_linewidth, cv2.LINE_AA)
            cv2.circle(blended, (u, v), max(3, top1_linewidth // 2 + 1),
                       col, -1, cv2.LINE_AA)
            last = (u, v)

    return blended


def draw_colorbar(img: np.ndarray, lo: float, hi: float, cmap_name: str,
                  label: str) -> np.ndarray:
    """Append a vertical colorbar strip on the right edge of `img`."""
    H, W = img.shape[:2]
    bar_w = 60
    pad = 28                                                             # right padding for tick labels
    panel_w = bar_w + 260
    panel = np.full((H, panel_w, 3), 255, np.uint8)

    cmap = matplotlib.colormaps[cmap_name]
    grad = np.linspace(1.0, 0.0, H - 2 * pad).reshape(-1, 1)             # high at top
    rgba = cmap(grad.repeat(bar_w, axis=1))
    bar = (rgba[..., :3] * 255).astype(np.uint8)[..., ::-1]              # BGR
    panel[pad:H - pad, 12:12 + bar_w] = bar
    cv2.rectangle(panel, (12, pad), (12 + bar_w, H - pad),
                  (80, 80, 80), 1, cv2.LINE_AA)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    n_ticks = 6
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        v = hi - frac * (hi - lo)
        y = int(pad + frac * (H - 2 * pad))
        cv2.line(panel, (12 + bar_w, y), (12 + bar_w + 6, y),
                 (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{v:+.2f}", (12 + bar_w + 10, y + 4),
                    FONT, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    return np.hstack([img, panel])


def render_bev(vocab_xy: np.ndarray, colors_bgr: np.ndarray,
               order: np.ndarray, *, top1_idx: int | None,
               top1_color_bgr: tuple, linewidth: int, top1_linewidth: int,
               x_max_fwd: float = 50.0, y_half_lat: float = 14.0,
               panel_w: int = 800, panel_h: int = 1080) -> np.ndarray:
    """Top-down BEV: every anchor as a polyline, colored by score, white bg.

    +X forward (up), +Y left. No ego/grid/legend — just trajectories."""
    canvas = np.full((panel_h, panel_w, 3), 255, np.uint8)

    def to_px(x_e: float, y_e: float) -> tuple[int, int]:
        x_px = int(round(panel_w / 2 - y_e * (panel_w / (2 * y_half_lat))))
        y_px = int(round(panel_h - 8 - x_e * (panel_h - 16) / x_max_fwd))
        return x_px, y_px

    for idx in order:
        col = (int(colors_bgr[idx, 0]), int(colors_bgr[idx, 1]),
               int(colors_bgr[idx, 2]))
        last = None
        for x, y in vocab_xy[idx]:
            if x < -1.0 or x > x_max_fwd + 1 or abs(y) > y_half_lat + 1:
                last = None
                continue
            u, v = to_px(float(x), float(y))
            if last is not None:
                cv2.line(canvas, last, (u, v), col, linewidth, cv2.LINE_AA)
            last = (u, v)

    if top1_idx is not None:
        last = None
        for x, y in vocab_xy[top1_idx]:
            if x < -1.0 or x > x_max_fwd + 1 or abs(y) > y_half_lat + 1:
                last = None
                continue
            u, v = to_px(float(x), float(y))
            if last is not None:
                cv2.line(canvas, last, (u, v), (0, 0, 0),
                         top1_linewidth + 3, cv2.LINE_AA)
                cv2.line(canvas, last, (u, v), top1_color_bgr,
                         top1_linewidth, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), max(3, top1_linewidth // 2 + 1),
                       top1_color_bgr, -1, cv2.LINE_AA)
            last = (u, v)

    return canvas


def find_scenario(manifest: dict, scenario_id: str) -> dict:
    for s in manifest["scenarios"]:
        if s["id"] == scenario_id:
            return s
    raise SystemExit(f"scenario '{scenario_id}' not found in manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="boston_03")
    ap.add_argument("--variant", default="animal_bear__center",
                    help="benchmark variant id, or 'clean' to use the original sensor image")
    ap.add_argument("--token", default=None,
                    help="frame token; default = end_frame's token")
    ap.add_argument("--pkl", default=None,
                    help="override predictions pkl (default: exp/benchmark_pkls/<sc>/<var>.pkl)")
    ap.add_argument("--vocab", default=str(VOCAB_PATH))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--score", default="combined",
                    choices=("combined",) + SUBSCORE_KEYS,
                    help="which score field to color by")
    ap.add_argument("--cmap", default="Blues",
                    help="matplotlib colormap (e.g. Blues, viridis, turbo)")
    ap.add_argument("--top1_color", default="255,0,0",
                    help="top-1 polyline color as 'R,G,B' (default red)")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="overlay blend strength")
    ap.add_argument("--linewidth", type=int, default=1)
    ap.add_argument("--top1_linewidth", type=int, default=6,
                    help="thickness for the highlighted top-1 trajectory")
    ap.add_argument("--no_top1", action="store_true",
                    help="don't highlight the top-1 trajectory")
    ap.add_argument("--bev_only", action="store_true",
                    help="skip the camera overlay and dump only a BEV "
                         "(top-down) image of the candidates on white bg")
    ap.add_argument("--bev_panel_w", type=int, default=800)
    ap.add_argument("--bev_panel_h", type=int, default=1080)
    ap.add_argument("--bev_x_max", type=float, default=50.0)
    ap.add_argument("--bev_y_half", type=float, default=14.0)
    ap.add_argument("--uniform_color", default=None,
                    help="if set (R,G,B), draw every candidate in this single "
                         "color instead of using the score colormap")
    ap.add_argument("--lo_pct", type=float, default=2.0,
                    help="lower percentile for colormap normalization")
    ap.add_argument("--hi_pct", type=float, default=99.0,
                    help="upper percentile for colormap normalization")
    ap.add_argument("--output", default=None,
                    help="output jpg path (default under exp/vocab_score_viz/...)")
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    sc_meta = find_scenario(manifest, args.scenario)
    log_dir = Path(manifest["defaults"]["log_dir"])
    log_path = log_dir / sc_meta["log"]

    pkl_path = Path(args.pkl) if args.pkl else GTRS_PKLS_ROOT / args.scenario / f"{args.variant}.pkl"
    with open(pkl_path, "rb") as f:
        preds = pickle.load(f)
    with open(log_path, "rb") as f:
        log = pickle.load(f)

    tok2idx = {fr["token"]: i for i, fr in enumerate(log)}

    if args.token is None:
        end_tok = log[sc_meta["end_frame"]]["token"]
        if end_tok in preds:
            token = end_tok
        else:
            token = next(iter(preds))
    else:
        token = args.token
    if token not in preds:
        raise SystemExit(f"token {token} not in {pkl_path}")
    if token not in tok2idx:
        raise SystemExit(f"token {token} not in log {log_path}")

    list_idx = tok2idx[token]
    frame = log[list_idx]
    cam = frame["cams"]["CAM_F0"]
    K = np.array(cam["cam_intrinsic"], dtype=np.float64)
    R = np.array(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.array(cam["sensor2lidar_translation"], dtype=np.float64)

    if args.variant == "clean":
        img_path = SENSOR_ROOT / cam["data_path"]
    else:
        img_path = BENCHMARK_DIR / args.scenario / args.variant / f"f{list_idx:04d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"cannot read image: {img_path}")

    vocab = np.load(args.vocab)                                          # (8192, 40, 3)
    vocab_xy = vocab[..., :2].astype(np.float64)
    data = preds[token]
    if args.score == "combined":
        scores = combined_score(data).astype(np.float64)
        score_label = "combined score (log-prob)"
    else:
        scores = np.asarray(data[args.score], dtype=np.float64)
        score_label = args.score

    colors_bgr, lo, hi = colors_from_scores(scores, args.cmap,
                                            args.lo_pct, args.hi_pct)
    if args.uniform_color:
        ur, ug, ub = (int(v) for v in args.uniform_color.split(","))
        colors_bgr = np.broadcast_to(np.array([ub, ug, ur], np.uint8),
                                     colors_bgr.shape).copy()
    order = np.argsort(scores)                                           # low score first

    top1 = None if args.no_top1 else int(np.argmax(scores))
    r, g, b = (int(v) for v in args.top1_color.split(","))
    top1_bgr = (b, g, r)

    if args.bev_only:
        overlaid = render_bev(
            vocab_xy, colors_bgr, order,
            top1_idx=top1, top1_color_bgr=top1_bgr,
            linewidth=args.linewidth, top1_linewidth=args.top1_linewidth,
            x_max_fwd=args.bev_x_max, y_half_lat=args.bev_y_half,
            panel_w=args.bev_panel_w, panel_h=args.bev_panel_h,
        )
    else:
        overlaid = draw_vocab(img, vocab_xy, colors_bgr, order, K, R, t,
                              args.alpha, args.linewidth,
                              top1_idx=top1, top1_linewidth=args.top1_linewidth,
                              top1_color_bgr=top1_bgr)
        overlaid = draw_colorbar(overlaid, lo, hi, args.cmap, score_label)

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_bev" if args.bev_only else ""
        out_path = OUT_ROOT / args.scenario / args.variant / f"{token}{suffix}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlaid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[viz] wrote {out_path}  ({overlaid.shape[1]}x{overlaid.shape[0]})")
    print(f"[viz] score={args.score} cmap={args.cmap} "
          f"range=[{lo:+.3f}, {hi:+.3f}] (pct {args.lo_pct}-{args.hi_pct})")


if __name__ == "__main__":
    main()
