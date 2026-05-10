#!/bin/bash
# Compute PDMS from a pre-dumped merged_predictions pkl (no agent inference).
#
# Usage:
#   PREDICTIONS_PATH=/path/to/merged.pkl \
#   TRAIN_TEST_SPLIT=navtest_sample200 \
#   bash scripts/evaluation/score_from_predictions.sh
set -e

source /workspace/CoachDrive/navsim_export_env.sh

export TRAIN_TEST_SPLIT=navtest_sample500
CACHE_PATH=${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache
EXP_NAME=score_from_preds_${TRAIN_TEST_SPLIT}
export PREDICTIONS_PATH=/workspace/CoachDrive/exp/gtrs_dense_subscore/gtrs_dense_baseline_navtest_sample500.pkl

echo "[score-from-preds] split=${TRAIN_TEST_SPLIT}  preds=${PREDICTIONS_PATH}"

export PROGRESS_MODE="eval"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_from_predictions.py" \
    experiment_name=$EXP_NAME \
    metric_cache_path=$CACHE_PATH \
    train_test_split=$TRAIN_TEST_SPLIT \
    +cache_path=null
