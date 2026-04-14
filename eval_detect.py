"""
评测目标检测模型（使用 YOLO 内置 val，与训练过程一致）。

用法:
  python eval_detect.py --data <data.yaml>
  python eval_detect.py --data <data.yaml> --model best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default=str(Path(__file__).parent / "models/yolo26x.pt"), help="模型权重路径")
    p.add_argument("--data",   required=True,              help="数据集配置文件（data.yaml）")
    p.add_argument("--device", default="",                 help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def _print_class_metrics(validator):
    try:
        v  = validator
        m  = v.metrics
        if len(m.ap_class_index) == 0:
            return
        names = v.names
        seen  = v.seen
        nt    = getattr(v, "nt_per_class", None)
        if nt is None:
            nt = getattr(v, "nt", None)
        if nt is None:
            try:
                lbls    = v.dataloader.dataset.labels
                all_cls = np.concatenate([l[:, 0] for l in lbls if len(l)]).astype(int)
                nt      = np.bincount(all_cls, minlength=max(names.keys()) + 1)
            except Exception:
                pass

        hf  = "%22s%11s%11s%11s%11s%11s%11s"
        pf  = "%22s%11i%11i%11.3g%11.3g%11.3g%11.3g"
        rf  = "%22s%11i%11s%11.3g%11.3g%11.3g%11.3g"
        fmt = pf if nt is not None else rf

        print(hf % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"))
        n_all = int(nt.sum()) if nt is not None else "-"
        print(fmt % ("all", seen, n_all, m.box.mp, m.box.mr, m.box.map50, m.box.map))
        for i, cls_idx in enumerate(m.ap_class_index):
            name   = names.get(int(cls_idx), str(cls_idx))
            n_inst = int(nt[int(cls_idx)]) if nt is not None else "-"
            print(fmt % (name, seen, n_inst,
                         m.box.p[i], m.box.r[i], m.box.ap50[i], m.box.ap[i]))
    except Exception as e:
        print(f"  [per-class metrics] 打印失败: {e}")


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}")
    print(f"  imgsz    1920")
    print(f"  conf     0.001  (ultralytics default, sweeps thresholds for AP)")
    print(f"  iou      0.6    (ultralytics default NMS IoU threshold)")
    print(f"  device   {args.device or 'auto'}")
    print("─" * 60, flush=True)

    model = YOLO(args.model)

    _captured = []
    def _capture(v): _captured.append(v)
    model.add_callback("on_val_end", _capture)
    model.val(data=args.data, imgsz=1920, device=args.device or None, verbose=False)

    print()
    if _captured:
        _print_class_metrics(_captured[0])
    else:
        print("  验证未返回结果")


if __name__ == "__main__":
    main()
