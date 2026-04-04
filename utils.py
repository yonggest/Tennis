import cv2
import os
import time


def text_params(frame_height, base_height=1080):
    """根据帧高度返回 (font_scale, thickness)，基准为 1080p。"""
    scale = frame_height / base_height
    return scale * 0.6, max(1, round(scale))


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def save_video(frames, path, fps=24):
    h, w = frames[0].shape[:2]
    out_path = os.path.splitext(path)[0] + '.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    total = len(frames)
    fw = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(frames):
        out.write(frame)
        pct = (i + 1) * 100 // total
        print(f"[   video] {i+1:>{fw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    out.release()
    print(f"[   video] {total:>{fw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")
    print(f"[   video] saved → {out_path}", flush=True)
