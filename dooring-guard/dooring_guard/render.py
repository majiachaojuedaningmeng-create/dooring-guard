"""画面渲染。

注意：画面上的文字一律用英文。OpenCV 的 putText 不支持中文，
在 RDK（Ubuntu）上通常也没有中文字体，硬写中文会变成方块。
中文留给 README、CSV 和终端输出。
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Track
from .risk import Level, RiskState

COLORS = {
    Level.SAFE: (86, 178, 66),
    Level.WARN: (40, 170, 245),
    Level.DANGER: (60, 60, 225),
}
LABELS = {Level.SAFE: "SAFE", Level.WARN: "CAUTION", Level.DANGER: "DANGER"}


def _dashed_polyline(frame, pts, color, thickness=2, dash=18, gap=12) -> None:
    pts = [tuple(map(int, p)) for p in pts]
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        total = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if total == 0:
            continue
        step = 0
        while step < total:
            a = step / total
            b = min(1.0, (step + dash) / total)
            q0 = (int(p0[0] + (p1[0] - p0[0]) * a), int(p0[1] + (p1[1] - p0[1]) * a))
            q1 = (int(p0[0] + (p1[0] - p0[0]) * b), int(p0[1] + (p1[1] - p0[1]) * b))
            cv2.line(frame, q0, q1, color, thickness, cv2.LINE_AA)
            step += dash + gap


def draw_roi(frame, roi) -> None:
    if not roi:
        return
    h, w = frame.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in roi], np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (150, 120, 70))
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)
    _dashed_polyline(frame, pts, (150, 120, 70), thickness=1)


def _pill(frame, text, org, color, scale=0.62, thickness=2) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + base + 4), color, -1)
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def draw_tracks(frame, tracks: list[Track], risks: dict[int, RiskState]) -> None:
    for t in tracks:
        r = risks.get(t.tid)
        if r is None:
            continue
        color = COLORS[r.level]
        x, y, w, h = t.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        if r.ttc != float("inf"):
            tail = f"#{t.tid} {r.ttc:.1f}s"
        else:
            tail = f"#{t.tid} --"
        _pill(frame, tail, (x, max(22, y - 6)), color, scale=0.5, thickness=1)


def draw_banner(frame, level: Level, ttc: float, d: float) -> None:
    h, w = frame.shape[:2]
    color = COLORS[level]
    bar_h = 58
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    text = f"DOOR CHECK: {LABELS[level]}"
    cv2.putText(frame, text, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    right = f"TTC {ttc:.1f}s" if ttc != float("inf") else "TTC --"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame, right, (w - tw - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    # 右上角距离条：越满越近
    bx, bw = w - 96, 18
    by, bh = 14, bar_h - 28
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1, cv2.LINE_AA)
    fill = int(np.clip(d, 0.0, 1.0) * bh)
    cv2.rectangle(frame, (bx, by + bh - fill), (bx + bw, by + bh), (255, 255, 255), -1)


def draw_hud(frame, n_tracks: int, fps_text: str) -> None:
    h = frame.shape[0]
    cv2.putText(
        frame,
        f"tracks {n_tracks}  |  {fps_text}",
        (22, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
