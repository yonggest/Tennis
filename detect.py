"""
第一阶段：球场检测 + 物体检测，结果保存为 COCO JSON。

用法：
    python detect.py -i <video>
    python detect.py -i <video> -o results/my.json -m models/yolo26x.pt -s models/court_seg.pt
输出：
    <video>.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys

import cv2

from utils import video_info, iter_frames, save_coco
from court_detector import CourtDetector
from objects_detector import ObjectsDetector


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',        required=True,                help='输入视频路径')
    p.add_argument('-o', '--output',        default=None,                 help='输出 JSON 路径（默认：输入同名，后缀改为 .json）')
    p.add_argument('-m', '--object-model', default='models/yolo26x.pt',  help='物体检测模型路径（球员/球拍/球）')
    p.add_argument('-s', '--court-model',  default='models/court_seg.pt', help='球场分割模型路径')
    p.add_argument('-c', '--conf',         type=float, default=0.5,      help='检测置信度阈值')
    p.add_argument('-z', '--imgsz',        type=int,   default=1920,     help='推理图片尺寸')
    p.add_argument('-d', '--device',       default=None,                  help='推理设备：cpu / cuda / mps（默认自动）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '.json'

    print("─" * 60)
    print(f"  input         {args.input}")
    print(f"  output        {output_path}")
    print(f"  object-model  {args.object_model}")
    print(f"  court-model   {args.court_model}")
    print(f"  conf          {args.conf:<10}  imgsz   {args.imgsz}")
    print(f"  device        {args.device or 'auto'}")
    print("─" * 60, flush=True)

    fps, width, height, n_frames = video_info(args.input)

    # ── 球场检测（仅第一帧）──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    _, first_frame = cap.read()
    cap.release()

    court = CourtDetector(seg_model=args.court_model)
    kps   = court.predict(first_frame)

    # ── 物体检测（全部帧，全图推理）──────────────────────────────────────────
    objects = ObjectsDetector(args.object_model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    players, rackets, balls = objects.run(
        iter_frames(args.input),
        total=n_frames,
    )

    save_coco(width, height, players, rackets, balls, output_path,
              fps=fps, court_kps=kps)


if __name__ == '__main__':
    main()
