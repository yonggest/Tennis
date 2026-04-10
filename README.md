# opentennis

网球比赛视频分析流水线：检测球员、球拍、网球，检测球场关键点，输出标注视频。

## 环境

GTX 1080 Ti (sm_61) 不支持 PyTorch 2.x，需用 Python 3.10 + torch 1.13.1+cu117：

```bash
python3.10 -m venv .venv
.venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv/bin/pip install ultralytics==8.4.35 opencv-python numpy==1.26.4 pandas scipy
```

## 运行流程

### 第一阶段：检测（球场 + 物体）

```bash
.venv/bin/python detect.py \
  -i <video> \
  -m models/yolo26x.pt \
  -s models/court_seg.pt \
  -z 1920
```

输出：`<video>.json`

### 第二阶段：渲染输出视频

```bash
.venv/bin/python render.py -i <video> -j <video>.json
```

输出：`<video>_out.mp4`

### 调试球场检测

```bash
.venv/bin/python debug_court.py -i <video>
```

输出：`<video>_court_debug.jpg`、`<video>_court_mask.jpg`

---

## 模型

| 文件 | 用途 |
|---|---|
| `models/yolo26x.pt` | 检测球员 / 球拍 / 网球（YOLOv8x fine-tuned） |
| `models/court_seg.pt` | 球场区域分割，用于初始化单应矩阵（YOLOv8n-seg） |

---

## 训练

### 物体检测模型（yolo26x）

```bash
.venv/bin/python train_yolo.py \
  --model models/yolo26x.pt \
  --data <dataset>/data.yaml \
  --epochs 50 --batch 2 --imgsz 1920 \
  --name finetune
```

### 球场分割模型（court_seg）

数据准备（COCO → YOLO seg 格式）：

```bash
# 在 annotation 目录下运行，分别转换 train/val
cd ../annotation
python coco2yolo.py \
  -i ../datasets/court-26 \
  -o ../opentennis/runs/datasets/court-26-yolo
```

> `coco2yolo.py` 只做格式转换，不拆分 train/val。train/val 需手动准备为独立目录后分别转换，或直接编辑生成的 `data.yaml` 指定路径。

训练：

```bash
.venv/bin/python train_yolo.py \
  --model models/yolov8n-seg.pt \
  --data <yolo_dir>/data.yaml \
  --epochs 100 --batch 4 --imgsz 640 \
  --lr0 0.001 --freeze 0 \
  --name court-seg-v1
```

> `--freeze 0`：训练数据量小（~15张）且与 COCO 域差异大，需放开全部层充分学习。

复制最佳权重：

```bash
cp runs/segment/exp/court-seg-v1/weights/best.pt models/court_seg.pt
```

### 评估

物体检测模型：

```bash
.venv/bin/python eval_yolo.py \
  --model models/yolo26x.pt \
  --data <dataset>/data.yaml
```

球场分割模型：

```bash
.venv/bin/python eval_yolo.py \
  --model runs/segment/exp/court-seg-v13/weights/best.pt \
  --data runs/datasets/court-26-yolo/data.yaml \
  --imgsz 640
```
