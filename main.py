import argparse
import os
import time
import cv2
from utils import read_video, save_video, text_params
from objects_detector import ObjectsDetector
from court_detector import CourtDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='input_videos/input_video.mp4')
    parser.add_argument('--model',  default='models/yolo26x.pt')
    parser.add_argument('--output', default=None, help='输出文件名（不含路径），默认与输入同名')
    args = parser.parse_args()

    input_dir  = os.path.dirname(args.input) or '.'
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    output_name = args.output if args.output else input_name + '_out.mp4'
    output_path = os.path.join(input_dir, output_name)

    frames, fps = read_video(args.input)

    court = CourtDetector()
    court.predict(frames[0])
    valid_hull = court.get_valid_zone_hull(frames[0].shape)

    objects = ObjectsDetector(args.model, imgsz=1920)
    players, rackets, balls = objects.run(frames, valid_hull=valid_hull)

    total = len(frames)
    w = len(str(total))
    fh = frames[0].shape[0]
    scale, thick = text_params(fh)
    scale_large, thick_large = text_params(fh, base_height=1080)
    margin = int(fh * 0.028)
    t0 = time.time()
    for i, frame in enumerate(frames):
        cv2.polylines(frame, [valid_hull], True, (0, 255, 255), 2)
        for det in players[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if det.get('track_id') is not None:
                cv2.putText(frame, f"P{det['track_id']}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)
        for det in rackets[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        for det in balls[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thick)
            tid = det.get('track_id')
            label = (f"B{tid}" if tid is not None else "B?") + f" {det['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick)
        cv2.putText(frame, str(i), (margin, fh - margin),
                    cv2.FONT_HERSHEY_SIMPLEX, scale_large * 1.5, (0, 255, 0), thick_large)
        pct = (i + 1) * 100 // total
        print(f"[    draw] {i+1:>{w}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    print(f"[    draw] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")

    save_video(frames, output_path, fps=fps)


if __name__ == "__main__":
    main()
