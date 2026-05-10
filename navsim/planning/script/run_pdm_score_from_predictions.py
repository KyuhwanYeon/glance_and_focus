"""Compute PDMS from a saved merged_predictions pkl, skipping agent inference.

Loads the pickle dumped via SUBSCORE_PATH in run_pdm_score_one_stage_gpu.py /
run_pdm_score_gpu_v2.py — schema {token: {'trajectory': Trajectory, ...}} —
and runs only the PDMS scoring + aggregation.

Reuses default_run_pdm_score_gpu hydra config so scene_filter, simulator,
scorer, traffic_agents, metric_cache_path are configured identically to the
inference run that produced the predictions.

Predictions path:
  PREDICTIONS_PATH=/path/to/merged.pkl   (env var)
  +predictions_path=/path/to/merged.pkl  (hydra override)
"""
import logging
import os
import pickle
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import List

import hydra
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import PDMResults, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.pdm_scoring_helpers import (
    compute_final_scores,
    create_scene_aggregators,
    infer_start_adjacent_mapping,
    run_pdm_score,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)

    predictions_path = os.environ.get("PREDICTIONS_PATH") or cfg.get("predictions_path")
    if not predictions_path:
        raise SystemExit(
            "Set PREDICTIONS_PATH=... env var or pass +predictions_path=... ; "
            "expects the merged_predictions pkl saved via SUBSCORE_PATH.")
    print(f"[from-preds] loading predictions: {predictions_path}")
    merged_predictions = pickle.load(open(predictions_path, "rb"))
    print(f"[from-preds] {len(merged_predictions)} prediction tokens")

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_tokens = set(scene_loader.tokens)
    cache_tokens = set(metric_cache_loader.tokens)
    pred_tokens = set(merged_predictions.keys())
    eval_set = scene_tokens & cache_tokens & pred_tokens

    print(
        f"[from-preds] tokens — scene={len(scene_tokens)} "
        f"cache={len(cache_tokens)} preds={len(pred_tokens)} "
        f"=> evaluate={len(eval_set)}"
    )
    missing = (scene_tokens & cache_tokens) - pred_tokens
    if missing:
        logger.warning(
            f"{len(missing)} scene+cache tokens missing from predictions; skipped"
        )
    if not eval_set:
        raise SystemExit("[from-preds] no overlapping tokens to score.")

    data_points = []
    for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items():
        sub = [t for t in tokens_list if t in eval_set]
        if not sub:
            continue
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": sub,
            "model_trajectory": {t: merged_predictions[t] for t in sub},
        })

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)
    pdm_score_df = pd.concat(score_rows)

    start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
    pdm_score_df = create_scene_aggregators(
        start_adjacent_mapping,
        pdm_score_df,
        instantiate(cfg.simulator.proposal_sampling),
    )
    pdm_score_df = compute_final_scores(pdm_score_df)

    num_ok = pdm_score_df["valid"].sum()
    num_fail = len(pdm_score_df) - num_ok
    failed = (
        pdm_score_df[~pdm_score_df["valid"]]["token"].to_list() if num_fail else []
    )

    score_cols = [
        c
        for c in pdm_score_df.columns
        if (
            (
                any(s.name in c for s in fields(PDMResults))
                or c in ("two_frame_extended_comfort", "score")
            )
            and c != "pdm_score"
        )
    ]
    avg_row = pdm_score_df[score_cols].mean(skipna=True)
    avg_row["token"] = "average_all_frames"
    avg_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
    pdm_score_df.loc[len(pdm_score_df)] = avg_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_csv = save_path / f"{timestamp}_from_preds.csv"
    pdm_score_df.to_csv(out_csv)

    logger.info(
        f"\n[from-preds] success={num_ok} fail={num_fail} "
        f"avg_score={pdm_score_df['score'].mean():.4f} -> {out_csv}"
    )
    if num_fail:
        logger.info(f"failed tokens: {failed}")


if __name__ == "__main__":
    main()
