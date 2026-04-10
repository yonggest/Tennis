"""
可视化球场检测每一步的中间结果。

用法：
    python debug_court.py -i <video>
输出：
    <video>_debug/  目录，包含各步骤可视化图
"""

import argparse
import os
import sys

import cv2
import numpy as np

from court_detector import CourtDetector, MODEL_KPS_M, COURT_LINES, COURT_W, COURT_L, NET_Y


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--input',  required=True, help='输入视频路径')
    p.add_argument('-s', '--court-model', default='models/court_seg.pt', help='球场分割模型路径')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def save(path, img, label):
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}  [{label}]")


def draw_court_lines(frame, H, color=(0, 220, 100), thickness=1):
    """把球场线条（含线宽两侧边缘）投影到 frame 上。"""
    vis = frame.copy()
    for (p1, p2, lw_m) in COURT_LINES:
        half = lw_m / 2
        if p1[1] == p2[1]:   # 水平线
            for dy in (-half, +half):
                pts = np.array([[[p1[0], p1[1]+dy], [p2[0], p2[1]+dy]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pts, H)[0].astype(int)
                cv2.line(vis, tuple(proj[0]), tuple(proj[1]), color, thickness)
        else:                 # 垂直线
            for dx in (-half, +half):
                pts = np.array([[[p1[0]+dx, p1[1]], [p2[0]+dx, p2[1]]], ], dtype=np.float32)
                proj = cv2.perspectiveTransform(pts, H)[0].astype(int)
                cv2.line(vis, tuple(proj[0]), tuple(proj[1]), color, thickness)
    # 网线
    net = np.array([[[0, NET_Y]], [[COURT_W, NET_Y]]], dtype=np.float32)
    net_px = cv2.perspectiveTransform(net, H)
    cv2.line(vis, tuple(net_px[0,0].astype(int)), tuple(net_px[1,0].astype(int)), color, thickness)
    return vis


def main():
    args = parse_args()

    # ── 读第一帧 ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("无法读取视频")
        sys.exit(1)

    out_dir = os.path.splitext(args.input)[0] + '_debug'
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n输出目录: {out_dir}\n")

    # ── 步骤 0：原始第一帧 ──────────────────────────────────────────
    save(f"{out_dir}/0_original.jpg", frame, "原始第一帧")

    # ── 步骤 1：球场颜色分割 ────────────────────────────────────────
    detector = CourtDetector(seg_model=args.court_model)
    court_mask = detector._get_court_mask(frame)

    vis_mask = frame.copy()
    vis_mask[court_mask == 0] = (vis_mask[court_mask == 0] * 0.35).astype(np.uint8)
    cv2.polylines(vis_mask,
                  [cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                   if cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else np.array([])],
                  True, (0, 255, 255), 2)
    save(f"{out_dir}/1_court_mask.jpg", vis_mask, "球场颜色分割（场外变暗）")

    # ── 步骤 2：球场内白色像素 ──────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    white_in = cv2.bitwise_and(white, court_mask)

    vis_white = frame.copy()
    vis_white[white_in > 0] = (0, 200, 255)   # 橙色标出白线像素
    save(f"{out_dir}/2_white_pixels.jpg", vis_white, "球场内白色像素（橙色）")

    # ── 步骤 3：距离场 ──────────────────────────────────────────────
    dist_map = detector._build_dist_map(frame, court_mask)
    cap_val   = float(dist_map.max())
    dist_vis  = (dist_map / cap_val * 255).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_INFERNO)
    save(f"{out_dir}/3_dist_map.jpg", dist_color, "距离场（亮=远离白线，暗=靠近白线）")

    # ── 步骤 4：YOLO seg 初始化 ─────────────────────────────────────
    H_seg, c_seg = detector._yolo_seg_init(frame, dist_map)
    print(f"  YOLO seg init cost: {c_seg:.3f}")

    if H_seg is not None:
        vis_seg = draw_court_lines(frame, H_seg, color=(0, 100, 255), thickness=1)
        # 画4个初始角点
        corners_m = np.array([[[0,0],[COURT_W,0],[COURT_W,COURT_L],[0,COURT_L]]], dtype=np.float32).reshape(-1,1,2)
        corners_px = cv2.perspectiveTransform(corners_m, H_seg).reshape(-1,2).astype(int)
        for pt in corners_px:
            cv2.circle(vis_seg, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(vis_seg, f"seg init cost={c_seg:.3f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
        save(f"{out_dir}/4_seg_init.jpg", vis_seg, f"YOLO seg 初始 H（cost={c_seg:.3f}）")
    else:
        print("  YOLO seg 初始化失败，跳过步骤 4")

    # ── 步骤 5：优化后结果 ──────────────────────────────────────────
    H_opt = detector._optimize(H_seg, dist_map, frame.shape)
    c_opt = detector._cost(H_opt, dist_map)
    print(f"  optimized cost: {c_opt:.3f}")

    vis_opt = draw_court_lines(frame, H_opt, color=(80, 200, 255), thickness=1)
    kps = cv2.perspectiveTransform(MODEL_KPS_M.reshape(-1,1,2), H_opt).reshape(-1,2).astype(int)
    for pt in kps:
        cv2.circle(vis_opt, tuple(pt), 4, (80, 200, 255), -1)
    cv2.putText(vis_opt, f"optimized cost={c_opt:.3f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 200, 255), 2)
    save(f"{out_dir}/5_optimized.jpg", vis_opt, f"优化后 H（cost={c_opt:.3f}）")

    # ── 步骤 6：模板像素投影（散点图）──────────────────────────────
    vis_tmpl = frame.copy()
    pts = detector._template_pts_m.reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H_opt).reshape(-1, 2)
    h_img, w_img = frame.shape[:2]
    valid = ((proj[:,0] >= 0) & (proj[:,0] < w_img) &
             (proj[:,1] >= 0) & (proj[:,1] < h_img))
    for x, y in proj[valid].astype(int):
        vis_tmpl[y, x] = (0, 255, 0)
    save(f"{out_dir}/6_template_proj.jpg", vis_tmpl, "模板白线像素投影（绿点）")

    print("\n完成。")


if __name__ == '__main__':
    main()
