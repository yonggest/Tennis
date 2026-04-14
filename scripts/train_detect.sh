#!/usr/bin/env bash
# 训练目标检测模型（球员 / 球拍 / 网球）
#
# 用法:  bash scripts/train_detect.sh <data.yaml>
# 示例:  bash scripts/train_detect.sh runs/in-out/tennis-track-26.yolo/data.yaml
# 输出:  runs/detect/exp/detect/

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "用法: bash scripts/train_detect.sh <data.yaml>" >&2
  echo "示例: bash scripts/train_detect.sh runs/in-out/tennis-track-26.yolo/data.yaml" >&2
  exit 1
fi

MODEL="models/yolo26x.pt"
DATA="$1"
EPOCHS=100
BATCH=2
IMGSZ=1920
LR0=0.001
FREEZE=10
NAME="detect"

cd "$(dirname "$0")/.."

CMD=".venv/bin/python train_yolo.py \
  --model  $MODEL \
  --data   $DATA \
  --epochs $EPOCHS \
  --batch  $BATCH \
  --imgsz  $IMGSZ \
  --lr0    $LR0 \
  --freeze $FREEZE"

echo "════════════════════════════════════════════════════════════"
echo "\$ $CMD"
echo "════════════════════════════════════════════════════════════"
echo ""

eval "$CMD"
