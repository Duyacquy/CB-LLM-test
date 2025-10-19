#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-SetFit/sst2}"
backbone="${2:-roberta}"   # roberta | bert | gpt2
shift 2 || true

# ví dụ: --use_relu --saga_epoch 800
extra_args=("$@")

python backbone_probe.py \
  --dataset="$dataset" \
  --backbone="$backbone" \
  "${extra_args[@]}"