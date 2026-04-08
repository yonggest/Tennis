# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Computer vision pipeline that analyzes tennis match video: detects players, rackets, and ball (YOLOv26x), detects court keypoints (template homography), assigns stable player IDs, filters the real tennis ball by trajectory, and outputs an annotated video.

## Running the Pipeline

```bash
.venv/bin/python main.py -i <input_video> -m models/yolo26x.pt --imgsz 1920
```

- **Output:** same directory as input, filename `<input>_out.mp4` (H.264, crf=18)
- **Required model:** `models/yolo26x.pt` — single model for all three classes
- **Python env:** `.venv/bin/python` (Python 3.10 required, see Dependencies)

## Dependencies

GTX 1080 Ti (sm_61) is not supported by PyTorch 2.x. Must use Python 3.10 + torch 1.13.1+cu117:

```bash
python3.10 -m venv .venv
.venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv/bin/pip install ultralytics==8.4.35 opencv-python numpy==1.26.4 pandas scipy
```

## Training & Evaluation

```bash
# Fine-tune
.venv/bin/python train_yolo.py --data datasets/xxx-yolo/data.yaml

# Evaluate
.venv/bin/python eval_yolo.py --data datasets/xxx-yolo/data.yaml
```

- Training outputs to `finetune/<timestamp>/` under the project root (absolute path, not affected by ultralytics global `runs_dir` setting)

## Architecture

### Pipeline Flow (`main.py`)

```
read_video()
  → CourtDetector.predict(frame 0) — 14 court keypoints via template homography
  → ObjectsDetector.run() — single model.predict() for person/racket/ball
  → hull mask applied to all frames
  → draw: court hull → players → rackets → balls → frame number
  → save_video() — ffmpeg H.264 encoding
```

### Key Design Decisions

- **Single inference:** `model.predict()` (not `model.track()`) with `classes=[0, 38, 32]` detects all three classes at once. `track()` was abandoned because ByteTrack drops unconfirmed detections (box.id=None), causing ball loss.
- **Court keypoints** detected only on frame 0 via template homography — the court doesn't move.
- **Valid zone hull:** convex hull of court keypoints used to mask out-of-court regions before detection.
- **Output encoding:** ffmpeg libx264, crf=18, preset=fast.
- **Inference device:** auto-selects cuda > cpu. MPS tested but slower than CPU for yolo26x on M5.

### Module Responsibilities

| Module | Purpose |
|---|---|
| `main.py` | Pipeline entry point, CLI args, drawing, video I/O |
| `objects_detector.py` | ObjectsDetector: single predict() call for person/racket/ball |
| `court_detector.py` | CourtDetector: template homography → 14 court keypoints, valid zone hull |
| `utils.py` | read_video, save_video (ffmpeg H.264), save_coco, text_params |
| `train_yolo.py` | Fine-tune YOLO on tennis dataset |
| `eval_yolo.py` | Evaluate model with per-class AP metrics |
