"""生成合成测试视频。

为什么需要它：这个项目要能在「没有任何素材」的机器上一条命令跑通。
没有真实行车记录仪素材时，用程序合成一段「后方来车从远处冲向车门」的画面，
至少能验证 检测 -> 跟踪 -> 风险分级 -> 渲染 这条链路是通的。

注意：合成视频只用于验证 pipeline，不代表真实场景下的精度。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
FPS = 30
DURATION = 6.0


def build_background(seed: int = 7) -> np.ndarray:
    """静态背景。所有元素每帧完全一致，否则会被背景建模当成运动目标。"""
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        v = int(46 + 34 * (y / H))
        bg[y, :] = (v, v, max(0, v - 6))

    noise = rng.integers(-7, 8, size=(H, W, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 远处墙面
    cv2.rectangle(bg, (0, 0), (W, int(0.10 * H)), (98, 98, 94), -1)
    cv2.line(bg, (0, int(0.10 * H)), (W, int(0.10 * H)), (130, 130, 126), 2)

    # 车位线
    cv2.line(bg, (int(0.20 * W), int(0.10 * H)), (int(0.24 * W), H), (196, 196, 192), 4)

    # 路面虚线（车道分界）
    y = int(0.16 * H)
    while y < H:
        cv2.line(bg, (int(0.52 * W), y), (int(0.56 * W), y + int(0.05 * H)), (168, 168, 164), 3)
        y += int(0.11 * H)
    return bg


def draw_vehicle(img: np.ndarray, cx: float, cy: float, bh: float) -> None:
    """画一台从后方驶来的车（后视剪影）。bh 是包围盒高度，用作距离代理量。"""
    bw = bh * 0.80
    x1, x2 = cx - bw / 2, cx + bw / 2
    y1, y2 = cy - bh / 2, cy + bh / 2
    r = max(2, int(bh * 0.13))

    body = (196, 196, 192)
    cv2.rectangle(img, (int(x1 + r), int(y1)), (int(x2 - r), int(y2)), body, -1)
    cv2.rectangle(img, (int(x1), int(y1 + r)), (int(x2), int(y2 - r)), body, -1)
    for ccx, ccy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (int(ccx), int(ccy)), r, body, -1, cv2.LINE_AA)

    # 后窗
    win = (118, 120, 124)
    wx1, wx2 = x1 + bw * 0.12, x2 - bw * 0.12
    wy1, wy2 = y1 + bh * 0.10, y1 + bh * 0.42
    cv2.rectangle(img, (int(wx1), int(wy1)), (int(wx2), int(wy2)), win, -1)

    # 尾灯
    lamp = (58, 58, 214)
    lw, lh = bw * 0.20, bh * 0.075
    for lx in (x1 + bw * 0.06, x2 - bw * 0.06 - lw):
        cv2.rectangle(img, (int(lx), int(y2 - bh * 0.26)),
                      (int(lx + lw), int(y2 - bh * 0.26 + lh)), lamp, -1)

    # 车牌
    pw, ph = bw * 0.26, bh * 0.06
    cv2.rectangle(img, (int(cx - pw / 2), int(y2 - bh * 0.15)),
                  (int(cx + pw / 2), int(y2 - bh * 0.15 + ph)), (238, 238, 234), -1)


def make_demo(path: Path, duration: float = DURATION) -> Path:
    bg = build_background()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"无法创建视频文件：{path}")

    total = int(duration * FPS)
    start = (int(0.45 * W), int(0.42 * H), 30.0)   # cx, cy, 包围盒高度
    end = (int(0.78 * W), int(0.66 * H), 380.0)    # 冲到车门附近

    for i in range(total):
        # 目标靠近是非线性的：越近越快，这样 TTC 才会收敛
        p = i / max(1, total - 1)
        e = p ** 1.6
        cx = start[0] + (end[0] - start[0]) * e
        cy = start[1] + (end[1] - start[1]) * e
        bh = start[2] + (end[2] - start[2]) * e
        frame = bg.copy()
        draw_vehicle(frame, cx, cy, bh)
        writer.write(frame)

    writer.release()
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="生成 DooringGuard 合成测试视频")
    p.add_argument("--out", default="assets/demo.mp4")
    p.add_argument("--duration", type=float, default=DURATION)
    args = p.parse_args()
    out = make_demo(Path(args.out), args.duration)
    print(f"已生成 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
