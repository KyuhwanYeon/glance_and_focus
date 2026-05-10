#!/bin/bash
# End-to-end: PDMS for VLM-reranked predictions on a sampled split.
#
# Pipeline:
#   1) run VLM (Cosmos-Reason2) on every token in the predictions pkl
#   2) rerank: substitute trajectory with VLM-filtered top-1
#   3) score the new pkl via run_pdm_score_from_predictions.py
#
# Configure via env:
#   TRAIN_TEST_SPLIT      (default navtest_sample200)
#   PREDICTIONS_PATH      input merged_predictions pkl (baseline GTRS-Dense)
#   VLM_OUT               output vlm risk pkl (default: under exp/vlm_on_preds)
#   RERANKED_PRED_OUT     output reranked predictions pkl
#   RISK_THRESHOLD        rerank exclusion threshold (default 0.5)
#   SKIP_VLM=1            skip stage 1 if VLM_OUT already exists
set -e
source /workspace/CoachDrive/navsim_export_env.sh

export TRAIN_TEST_SPLIT=navtest_sample1000
CACHE_PATH="${NAVSIM_DEVKIT_ROOT}/cache/${TRAIN_TEST_SPLIT}_metric_cache"
export PREDICTIONS_PATH=/workspace/CoachDrive/exp/gtrs_dense_subscore/gtrs_dense_baseline_navtest_sample1000.pkl


OUT_DIR="${NAVSIM_EXP_ROOT}/vlm_on_preds/${TRAIN_TEST_SPLIT}"
mkdir -p "$OUT_DIR"

VLM_OUT="${VLM_OUT:-$OUT_DIR/vlm_risk.pkl}"
RERANKED_PRED_OUT="${RERANKED_PRED_OUT:-$OUT_DIR/predictions_reranked.pkl}"
RISK_THRESHOLD="${RISK_THRESHOLD:-0.5}"

export VLM_OUT

echo "[score-with-vlm] split=$TRAIN_TEST_SPLIT"
echo "[score-with-vlm] preds  = $PREDICTIONS_PATH"
echo "[score-with-vlm] vlm    = $VLM_OUT"
echo "[score-with-vlm] reranked preds = $RERANKED_PRED_OUT"

# 1) VLM dump
if [[ "${SKIP_VLM:-0}" == "1" && -f "$VLM_OUT" ]]; then
    echo "[score-with-vlm] (1/3) SKIP_VLM=1 and $VLM_OUT exists -> skipping"
else
    echo "[score-with-vlm] (1/3) running VLM dump"
    python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_vlm_dump_on_predictions.py" \
        train_test_split="$TRAIN_TEST_SPLIT" \
        experiment_name="vlm_dump_${TRAIN_TEST_SPLIT}" \
        +vlm_out="$VLM_OUT" \
        +cache_path=null
fi

# 2) Rerank
echo "[score-with-vlm] (2/3) reranking trajectories with VLM risk"
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/rerank_predictions_with_vlm.py" \
    --predictions "$PREDICTIONS_PATH" \
    --vlm_risk "$VLM_OUT" \
    --out "$RERANKED_PRED_OUT" \
    --risk_threshold "$RISK_THRESHOLD"

# 3) Score
echo "[score-with-vlm] (3/3) computing PDMS on reranked predictions"
EXP_NAME="score_from_preds_vlm_${TRAIN_TEST_SPLIT}"
export PREDICTIONS_PATH="$RERANKED_PRED_OUT"
python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_from_predictions.py" \
    experiment_name="$EXP_NAME" \
    metric_cache_path="$CACHE_PATH" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    +cache_path=null
