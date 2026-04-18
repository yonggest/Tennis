"""
第五阶段：读取 pose.py 输出的 JSON，检测有效回合（rally）及回合后球员的庆祝动作（hype）。

用法：
    python rally.py -i <video>.posed.json
    python rally.py -i <video>.posed.json -o <video>.rallies.json
输出：
    <video>.rallies.json（默认：将 _posed 后缀替换为 _rallies）

回合检测算法：
    以每帧是否存在有效追踪球（valid=True, track_id!=None）为信号，
    允许最多 --gap-seconds 的无球间隙，过滤掉短于 --min-seconds 的片段。

Hype 检测算法：
    回合结束后 --hype-window-seconds 秒内，
    检测有效球员的手腕是否明显高于肩膀（举手/挥拳动作）。
"""

import argparse
import json
import os
import sys

from utils import load_detections, propagate_video


# COCO 17点关键点索引
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ELBOW     = 7
_KP_RIGHT_ELBOW    = 8
_KP_LEFT_WRIST     = 9
_KP_RIGHT_WRIST    = 10
_KP_LEFT_HIP       = 11
_KP_RIGHT_HIP      = 12

_KP_CONF_THRESH    = 0.3    # 关键点可信阈值


def _is_arm_raised(kps):
    """判断球员是否举手：手腕明显高于肩膀（图像坐标中 y 更小）。
    阈值自适应躯干高度（肩到髋距离的 20%），无法估算时用固定 20px。
    """
    if not kps or len(kps) < 17:
        return False

    ls, rs = kps[_KP_LEFT_SHOULDER],  kps[_KP_RIGHT_SHOULDER]
    lw, rw = kps[_KP_LEFT_WRIST],     kps[_KP_RIGHT_WRIST]
    lh, rh = kps[_KP_LEFT_HIP],       kps[_KP_RIGHT_HIP]

    # 估算躯干高度（肩→髋），用于自适应阈值
    torso_h = 0.0
    if ls[2] > _KP_CONF_THRESH and lh[2] > _KP_CONF_THRESH:
        torso_h = max(torso_h, abs(float(ls[1]) - float(lh[1])))
    if rs[2] > _KP_CONF_THRESH and rh[2] > _KP_CONF_THRESH:
        torso_h = max(torso_h, abs(float(rs[1]) - float(rh[1])))
    threshold = torso_h * 0.2 if torso_h > 10 else 20.0

    left_raised  = (lw[2] > _KP_CONF_THRESH and ls[2] > _KP_CONF_THRESH and
                    float(lw[1]) < float(ls[1]) - threshold)
    right_raised = (rw[2] > _KP_CONF_THRESH and rs[2] > _KP_CONF_THRESH and
                    float(rw[1]) < float(rs[1]) - threshold)
    return left_raised or right_raised


def detect_rallies(balls, fps, gap_seconds=1.0, min_seconds=0.5):
    """检测回合：以有效追踪球的存在为信号，合并短暂间隙，过滤过短片段。

    返回：[(start_frame, end_frame), ...]
    """
    gap_frames = int(fps * gap_seconds)
    min_frames = int(fps * min_seconds)
    n = len(balls)

    # 每帧是否有有效追踪球
    live = [False] * n
    for fi, frame_balls in enumerate(balls):
        for det in frame_balls:
            if det.get('valid', True) and det.get('track_id') is not None:
                live[fi] = True
                break

    # 合并间隙，提取片段
    rallies = []
    start = None
    gap = 0
    for fi in range(n):
        if live[fi]:
            if start is None:
                start = fi
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > gap_frames:
                    end = fi - gap
                    if end - start >= min_frames:
                        rallies.append((start, end))
                    start = None
                    gap = 0

    # 处理末尾片段
    if start is not None:
        end = n - 1 - gap
        if end - start >= min_frames:
            rallies.append((start, end))

    return rallies


def detect_hypes(players, rallies, fps):
    """检测每个回合结束后紧接的下一帧中球员的举手/挥拳动作。

    返回：{rally_idx: [{"frame": int, "time": float, "track_id": int|None, "gesture": str}, ...]}
    """
    n = len(players)
    result = {}

    for ri, (_, end) in enumerate(rallies):
        hypes = []
        fi = end + 1
        if fi >= n:
            result[ri] = hypes
            continue
        for det in players[fi]:
            if not det.get('valid', True):
                continue
            kps = det.get('keypoints')
            if not _is_arm_raised(kps):
                continue
            tid = det.get('track_id')
            hypes.append({
                'frame':    fi,
                'time':     round(fi / fps, 3),
                'track_id': tid,
                'gesture':  'arm_raise',
            })
        result[ri] = hypes

    return result


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',          required=True,      help='pose.py（或 parse.py）输出的 JSON 路径')
    p.add_argument('-o', '--output',         default=None,       help='输出 JSON 路径（默认：输入同名加 _rallies）')
    p.add_argument('--gap-seconds',          type=float, default=1.0,  help='允许合并的无球间隙时长（秒）')
    p.add_argument('--min-seconds',          type=float, default=0.5,  help='最短回合时长（秒）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    for suffix in ('.posed', '.parsed', '.tracked', '.detected'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    output_path = args.output or stem + '.rallies.json'

    print('─' * 60)
    print(f'  input           {args.input}')
    print(f'  output          {output_path}')
    print(f'  gap             {args.gap_seconds}s')
    print(f'  min rally       {args.min_seconds}s')
    print('─' * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)

    rallies_raw = detect_rallies(balls, fps,
                                 gap_seconds=args.gap_seconds,
                                 min_seconds=args.min_seconds)
    hypes_by_rally = detect_hypes(players, rallies_raw, fps)

    events = []
    for ri, (start, end) in enumerate(rallies_raw):
        events.append({
            'type':        'rally',
            'id':          ri,
            'start_frame': start,
            'end_frame':   end,
            'start_time':  round(start / fps, 3),
            'end_time':    round(end   / fps, 3),
            'n_frames':    end - start + 1,
        })
        for hype in hypes_by_rally.get(ri, []):
            events.append(dict({'type': 'hype'}, **hype))

    result = {
        'fps':    fps,
        'width':  width,
        'height': height,
        'events': events,
    }
    video_rel = propagate_video(args.input, output_path)
    if video_rel is not None:
        result['video'] = video_rel
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    rallies = [e for e in events if e['type'] == 'rally']
    hypes   = [e for e in events if e['type'] == 'hype']
    print(f'[ rally] {len(rallies)} rallies,  {len(hypes)} hype events')
    for e in events:
        if e['type'] == 'rally':
            print(f"         rally #{e['id']:>2}  frame {e['start_frame']:>4}–{e['end_frame']:>4}"
                  f"  ({e['start_time']:.1f}s–{e['end_time']:.1f}s)  {e['n_frames']} frames")
        else:
            print(f"           hype       frame {e['frame']:>4}"
                  f"  ({e['time']:.1f}s)  track_id={e['track_id']}  {e['gesture']}")
    print(f'[ rally] saved → {output_path}', flush=True)


if __name__ == '__main__':
    main()
