"""
第三阶段：读取 track.py 输出的 JSON（含 track_id），
进行缓冲区过滤，输出供 render.py 使用的处理后 JSON。

也可直接读取 detect.py 输出（跳过追踪阶段），此时球均视为无轨迹检测处理。

用法：
    python parse.py -i <video>.tracked.json
    python parse.py -i <video>.tracked.json -o <video>.parsed.json
输出：
    <video>.parsed.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from utils import load_detections, save_coco, propagate_video


_STATIC_BBOX_DIAG_PX = 20.0  # 静止球判定阈值：轨迹全局包围盒对角线（像素）



def _in_hull(hull, x, y):
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def _bbox_overlaps_hull(hull, x1, y1, x2, y2):
    """bbox 的中心或任一角点在凸包内则返回 True。"""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for pt in [(cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        if _in_hull(hull, *pt):
            return True
    return False


def _filter_players(players, ground_hull):
    """返回 (kept, removed)，底部中心在地面缓冲区内为有效球员。"""
    kept, removed = [], []
    for frame in players:
        k, r = [], []
        for d in frame:
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            (k if _in_hull(ground_hull, cx, d['bbox'][3]) else r).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _filter_rackets(rackets, volume_hull):
    """返回 (kept, removed)，bbox 与缓冲区立方体凸包有重叠为有效球拍。"""
    kept, removed = [], []
    for frame in rackets:
        k, r = [], []
        for d in frame:
            (k if _bbox_overlaps_hull(volume_hull, *d['bbox']) else r).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _make_wall_quads(vol_bottom_pts, vol_top_pts, img_height):
    """构造左右侧边墙延伸到天空的四边形 (4,1,2) float32。
    球起点落在墙面四边形内 → 出界。
    vol_bottom/top 顺序: [远左, 远右, 近右, 近左]
    """
    bpts = np.array(vol_bottom_pts, dtype=np.float64)
    tpts = np.array(vol_top_pts,    dtype=np.float64)
    fl_b, fr_b, nr_b, nl_b = bpts
    fl_t, fr_t, nr_t, nl_t = tpts
    sky_y = float(-img_height)

    def to_sky(p_b, p_t):
        dy = p_t[1] - p_b[1]
        if abs(dy) < 1e-6:
            return p_t.copy()
        t = (sky_y - p_t[1]) / dy
        return p_t + t * (p_t - p_b)

    def quad(a, b, c, d):
        return np.array([a[:2], b[:2], c[:2], d[:2]],
                        dtype=np.float32).reshape(-1, 1, 2)

    # 左墙四边形：远左底 → 近左底 → 近左天 → 远左天
    left_q  = quad(fl_b, nl_b, to_sky(nl_b, nl_t), to_sky(fl_b, fl_t))
    # 右墙四边形：远右底 → 近右底 → 近右天 → 远右天
    right_q = quad(fr_b, nr_b, to_sky(nr_b, nr_t), to_sky(fr_b, fr_t))
    return left_q, right_q


def _filter_balls(balls, left_wall_q, right_wall_q, volume_hull):
    """
    返回 (kept, removed)。
    静止球（轨迹全局包围盒对角线 < _STATIC_BBOX_DIAG_PX）必须落在立方体凸包内；
    运动球起点落在左墙或右墙四边形内（出界）→ 整条轨迹无效。

    track_id 处理：
    - 若输入来自 track.py（有任意非 None 的 track_id）：track_id=None 的检测直接移除。
    - 若输入来自 detect.py（所有 track_id 均为 None）：按静止球处理（兼容跳过 track 的用法）。
    """
    # 判断是否来自 track 阶段
    has_tracked = any(d.get('track_id') is not None
                      for frame in balls for d in frame)

    track_pts = defaultdict(list)
    for frame in balls:
        for d in frame:
            tid = d.get('track_id')
            if tid is not None:
                cx = (d['bbox'][0] + d['bbox'][2]) / 2
                cy = (d['bbox'][1] + d['bbox'][3]) / 2
                track_pts[tid].append((cx, cy))

    static_tracks  = set()
    invalid_tracks = set()
    for tid, pts in track_pts.items():
        if len(pts) < 2:
            static_tracks.add(tid)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox_diag = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if bbox_diag < _STATIC_BBOX_DIAG_PX:
            static_tracks.add(tid)
        elif (_in_hull(left_wall_q,  pts[0][0], pts[0][1]) or
              _in_hull(right_wall_q, pts[0][0], pts[0][1])):
            invalid_tracks.add(tid)

    kept, removed = [], []
    for frame in balls:
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            cx  = (d['bbox'][0] + d['bbox'][2]) / 2
            cy  = (d['bbox'][1] + d['bbox'][3]) / 2
            if tid in invalid_tracks:
                r.append(d)
            elif tid is None:
                # track 阶段已明确拒绝的检测 → 直接移除；未经 track 阶段则按静止球处理
                if has_tracked:
                    r.append(d)
                else:
                    (k if _in_hull(volume_hull, cx, cy) else r).append(d)
            elif tid in static_tracks:
                (k if _in_hull(volume_hull, cx, cy) else r).append(d)
            else:
                k.append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True, help='track.py（或 detect.py）输出的 JSON 路径')
    p.add_argument('-o', '--output', default=None,  help='输出 JSON 路径（默认：输入同名加 _parsed）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    if stem.endswith('.tracked'):
        stem = stem[:-len('.tracked')]
    output_path = args.output or stem + '.parsed.json'

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  output  {output_path}")
    print("─" * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)

    # 缓冲区过滤
    ground_hull = court['ground_hull']
    volume_hull = court['volume_hull']
    left_wall_q, right_wall_q = _make_wall_quads(
        court['vol_bottom_pts'], court['vol_top_pts'], height)

    n_players_before = sum(len(f) for f in players)
    n_rackets_before = sum(len(f) for f in rackets)
    n_balls_before   = sum(len(f) for f in balls)
    players, players_inv = _filter_players(players, ground_hull)
    rackets, rackets_inv = _filter_rackets(rackets, volume_hull)
    balls,   balls_inv   = _filter_balls(balls, left_wall_q, right_wall_q, volume_hull)
    print(f"[  filter] players: {n_players_before} → {sum(len(f) for f in players)}")
    print(f"[  filter] rackets: {n_rackets_before} → {sum(len(f) for f in rackets)}")
    print(f"[  filter] balls:   {n_balls_before} → {sum(len(f) for f in balls)}")

    # 合并 valid/invalid，写入 valid 标记
    n = len(players)
    players_out = [[dict(d, valid=True)  for d in players[fi]] +
                   [dict(d, valid=False) for d in players_inv[fi]] for fi in range(n)]
    rackets_out = [[dict(d, valid=True)  for d in rackets[fi]] +
                   [dict(d, valid=False) for d in rackets_inv[fi]] for fi in range(n)]
    balls_out   = [[dict(d, valid=True)  for d in balls[fi]] +
                   [dict(d, valid=False) for d in balls_inv[fi]] for fi in range(n)]

    save_coco(width, height, players_out, rackets_out, balls_out,
              output_path, fps=fps, court=court,
              video=propagate_video(args.input, output_path))


if __name__ == '__main__':
    main()
