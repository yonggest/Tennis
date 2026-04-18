#!/usr/bin/env bash
# render.py 的便捷包装。视频路径从 JSON 的 video 字段自动读取。
#
# 用法:
#   bash render.sh <输入.posed.json> <输出.mp4>   # 单文件
#   bash render.sh <输入目录> <输出目录>            # 批量（*.posed.json → *.mp4）

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: bash render.sh <输入.posed.json|目录> <输出.mp4|目录>" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

cd "$(dirname "$0")/.."

declare -a in_list=()
declare -a out_list=()

if [[ -f "$INPUT" ]]; then
  in_list+=("$INPUT")
  out_list+=("$OUTPUT")
elif [[ -d "$INPUT" ]]; then
  while IFS= read -r f; do
    stem="$(basename "$f" .posed.json)"
    in_list+=("$f")
    out_list+=("$OUTPUT/${stem}.mp4")
  done < <(find "$INPUT" -maxdepth 1 -name "*.posed.json" | sort)
else
  echo "错误: 输入路径不存在: $INPUT" >&2; exit 1
fi

if [[ ${#in_list[@]} -eq 0 ]]; then
  echo "错误: 未找到 *.posed.json 文件" >&2; exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  找到 ${#in_list[@]} 个文件"
echo "════════════════════════════════════════════════════════════"

ok=0; fail=0
for i in "${!in_list[@]}"; do
  in_json="${in_list[$i]}"
  out_mp4="${out_list[$i]}"
  mkdir -p "$(dirname "$out_mp4")"

  echo ""
  echo "── $(basename "$in_json")"
  echo "   → $out_mp4"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python render.py\033[0m -j \"$in_json\" -o \"$out_mp4\""

  if .venv/bin/python render.py -j "$in_json" -o "$out_mp4"; then
    ok=$((ok + 1))
  else
    echo "  [FAILED] $in_json" >&2
    fail=$((fail + 1))
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  完成: $ok 成功  $fail 失败"
echo "════════════════════════════════════════════════════════════"
