"""
评测目标检测模型。

预测结果和 GT 标注都先转换为 class name，在名称空间统一比较，
无需关心不同模型的 class index 差异。

用法:
  python eval_detect.py --data <data.yaml>                   # 评测原始 COCO 模型
  python eval_detect.py --data <data.yaml> --model best.pt   # 评测微调后模型
"""

import argparse
import sys
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

IOU_THRESH  = 0.5
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default=str(Path(__file__).parent / "models/yolo26x.pt"), help="模型权重路径")
    p.add_argument("--data",   required=True,               help="数据集配置文件（data.yaml）")
    p.add_argument("--conf",   type=float, default=0.5,    help="置信度阈值")
    p.add_argument("--device", default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


# ── 标注加载 ──────────────────────────────────────────────────────────────────

def load_labels_detect(label_dir: Path, idx_to_name: dict) -> dict:
    """{stem: [(class_name, cx, cy, w, h), ...]}"""
    labels = {}
    for f in label_dir.glob("*.txt"):
        items = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls_idx = int(parts[0])
                name = idx_to_name.get(cls_idx)
                if name:
                    items.append((name, *map(float, parts[1:])))
        labels[f.stem] = items
    return labels


# ── IoU 工具 ──────────────────────────────────────────────────────────────────

def iou_box(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


# ── 评测主逻辑 ────────────────────────────────────────────────────────────────

def xywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    return (cx - w/2)*img_w, (cy - h/2)*img_h, (cx + w/2)*img_w, (cy + h/2)*img_h


def compute_ap(precisions, recalls):
    """AP（11-point interpolation）"""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0)
        ap += p / 11
    return ap


def evaluate_detect(model, val_img_dir, val_label_dir, args, gt_names):
    idx_to_name  = {i: name for i, name in enumerate(gt_names)}
    gt_labels    = load_labels_detect(val_label_dir, idx_to_name)
    model_names  = model.names
    eval_names   = sorted(gt_names)

    detections = defaultdict(list)
    gt_counts  = defaultdict(int)

    img_paths = sorted(p for p in val_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"标注文件数: {len(gt_labels)}  评测图片数: {len(img_paths)}")

    for img_path in tqdm(img_paths, unit="img"):
        stem = img_path.stem
        gts  = gt_labels.get(stem, [])

        gt_by_name = defaultdict(list)
        for (name, cx, cy, w, h) in gts:
            gt_counts[name] += 1
            gt_by_name[name].append((cx, cy, w, h))

        result  = model.predict(str(img_path), imgsz=1920,
                                conf=args.conf, device=args.device or None,
                                verbose=False, save=False)[0]
        img_h, img_w = result.orig_shape

        preds_by_name = defaultdict(list)
        for box in result.boxes:
            name = model_names.get(int(box.cls.item()))
            if name not in eval_names:
                continue
            preds_by_name[name].append((float(box.conf.item()), box.xyxy[0].tolist()))

        for name in eval_names:
            gt_boxes = [xywh_to_xyxy(cx, cy, w, h, img_w, img_h)
                        for (cx, cy, w, h) in gt_by_name[name]]
            matched  = [False] * len(gt_boxes)
            for conf, pred_xyxy in sorted(preds_by_name[name], key=lambda x: -x[0]):
                best_iou, best_j = 0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    v = iou_box(pred_xyxy, gt_box)
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_iou >= IOU_THRESH and best_j >= 0
                if is_tp:
                    matched[best_j] = True
                detections[name].append((conf, int(is_tp)))

    _print_results(eval_names, detections, gt_counts)


def _print_results(eval_names, detections, gt_counts):
    print(f"\n{'类别':<18} {'GT':>6} {'Pred':>6} {'P':>8} {'R':>8} {'mAP50':>8}")
    print("-" * 60)
    aps = []
    for name in eval_names:
        dets = sorted(detections[name], key=lambda x: -x[0])
        n_gt = gt_counts[name]
        if n_gt == 0:
            print(f"{name:<18} {n_gt:>6} {len(dets):>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        if not dets:
            aps.append(0.0)
            print(f"{name:<18} {n_gt:>6} {0:>6} {'N/A':>8} {'0.0000':>8} {'0.0000':>8}")
            continue
        tp_cum = np.cumsum([d[1] for d in dets])
        fp_cum = np.cumsum([1 - d[1] for d in dets])
        prec   = tp_cum / (tp_cum + fp_cum)
        rec    = tp_cum / n_gt
        ap     = compute_ap(prec.tolist(), rec.tolist())
        aps.append(ap)
        print(f"{name:<18} {n_gt:>6} {len(dets):>6} "
              f"{float(prec[-1]):>8.4f} {float(rec[-1]):>8.4f} {ap:>8.4f}")
    mean_ap = np.mean(aps) if aps else 0.0
    print("-" * 60)
    print(f"{'all':<18} {sum(gt_counts.values()):>6} {'':>6} {'':>8} {'':>8} {mean_ap:>8.4f}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def find_label_dir(data_root: Path, val_rel: str) -> Path:
    sub = Path(val_rel).name
    candidate = data_root / "labels" / sub
    if candidate.exists():
        return candidate
    fallback = data_root / "labels"
    if not fallback.exists():
        raise FileNotFoundError(f"找不到标注目录: {candidate}  或  {fallback}")
    return fallback


def main():
    args = parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    data_root     = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() \
                    else (Path(args.data).parent / cfg["path"]).resolve()
    val_img_dir   = data_root / cfg["val"]
    val_label_dir = find_label_dir(data_root, cfg["val"])
    gt_names      = cfg["names"]
    if isinstance(gt_names, dict):
        gt_names = [gt_names[k] for k in sorted(gt_names)]

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}")
    print(f"  imgsz    {1920}")
    print(f"  conf     {args.conf}")
    print(f"  device   {args.device or 'auto'}")
    print("─" * 60, flush=True)

    model = YOLO(args.model)
    print(f"[   model] classes={list(model.names.values())}")
    print(f"[    data] gt_classes={gt_names}  label_dir={val_label_dir}")

    evaluate_detect(model, val_img_dir, val_label_dir, args, gt_names)


if __name__ == "__main__":
    main()
