#!/usr/bin/env bash
set -euo pipefail

# Usage: 5_train_bert.bash "<gpu_ids>" "<outdir>" [extra-args...]
# Example: 5_train_bert.bash "0,1,2,3" OUTPUT/RP/BERT_fixed --num_train_epochs 20 ...

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"<gpu_ids>\" <outdir> [extra-args...]"
  exit 1
fi

GPU_IDS="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# Count procs = number of GPUs exposed
NPROC=$(( $(tr -cd ',' <<<"$GPU_IDS" | wc -c) + 1 ))

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Launching $NPROC processes (gpus: $GPU_IDS)"
echo "Output dir: $OUTDIR"

torchrun \
  --nproc-per-node="$NPROC" \
  --standalone \
  --rdzv-backend=c10d \
  finetune_simplified.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --output_dir "$OUTDIR" \
    --overwrite_output_dir \
    "$@"
