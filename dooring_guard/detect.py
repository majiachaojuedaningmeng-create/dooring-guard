"""运动目标检测与轻量跟踪。

设计取舍：默认不加载任何预训练模型。
- 优点：pip install 完就能跑，不需要 GPU、不需要下载权重，符合「最小可运行」。
- 代价：只能感知「画面里在动的东西」，分不清自行车和汽车。
需要更准的时候用 `--engine yolo`（见 yolo_engine.py）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """一条轨迹。history 存 (ts, cx, cy, w, h)，按时间戳而非帧号，换帧率也不坏。"""

    tid: int
    box: Box
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    missed: int = 0

    @property
    def age(self) -> int:
        return len(self.history)

    def sample(self, max_age: float | None = None):
        """取最近一帧 (ts, cx, cy, w, h)，可限定最大时间跨度。"""
        ts, cx, cy, w, h = self.history[-1]
        if max_age is None:
            return ts, cx, cy, w, h
        for item in reversed(self.history):
            if ts - item[0] >= max_age:
                return item
        return self.history[0]


class MotionDetector:
    """MOG2 背景建模 + 形态学 + 轮廓筛选。

    适用场景：机位固定（车载侧后视、门柱摄像头、停车位立杆）。
    不适用：手持抖动画面 —— 那需要先做稳像，本项目不处理。
    """

    def __init__(
        self,
        min_area_ratio: float = 0.0006,
        max_area_ratio: float = 0.60,
        history: int = 300,
        var_threshold: float = 24.0,
        roi: Sequence[Sequence[float]] | None = None,
        warmup: int = 20,
    ) -> None:
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.roi = [(float(p[0]), float(p[1])) for p in roi] if roi else None
        self.warmup = warmup
        self._seen = 0

    def roi_mask(self, shape) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        if self.roi is None:
            mask[:] = 255
        else:
            pts = np.array([[int(x * w), int(y * h)] for x, y in self.roi], np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def warm(self, frame) -> None:
        """喂前几帧建立背景模型，避免刚开播就整屏误检。"""
        self._sub.apply(frame)
        self._seen += 1

    def __call__(self, frame) -> list[Box]:
        fg = self._sub.apply(frame)
        if self._seen < self.warmup:
            self._seen += 1
            return []

        fg[fg == 127] = 0  # 阴影
        fg = cv2.medianBlur(fg, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.dilate(fg, kernel, iterations=1)
        fg = cv2.bitwise_and(fg, self.roi_mask(fg.shape))

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        boxes: list[Box] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_ratio * frame_area:
                continue
            if area > self.max_area_ratio * frame_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < 14 or w < 8:
                continue
            boxes.append((x, y, w, h))
        return self._merge(boxes)

    @staticmethod
    def _should_merge(a: Box, b: Box) -> bool:
        """同一个物体的碎片要合起来。

        真实场景里常见：车轮和车身被拆成两块、贴地阴影单独成块。
        不合并的话，跟踪器会当成两个目标，凭空多出一条假轨迹。
        """
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ox = min(ax + aw, bx + bw) - max(ax, bx)
        if ox <= 0:
            return False
        if ox / max(1, min(aw, bw)) < 0.5:  # 横向重叠不足，大概率不是同一个东西
            return False
        gap = max(ay, by) - min(ay + ah, by + bh)
        return gap < 0.5 * min(ah, bh)

    @classmethod
    def _merge(cls, boxes: list[Box]) -> list[Box]:
        boxes = list(boxes)
        merged_any = True
        while merged_any:
            merged_any = False
            out: list[Box] = []
            for box in boxes:
                for i, kept in enumerate(out):
                    if cls._should_merge(box, kept):
                        ax, ay = min(box[0], kept[0]), min(box[1], kept[1])
                        ax2 = max(box[0] + box[2], kept[0] + kept[2])
                        ay2 = max(box[1] + box[3], kept[1] + kept[3])
                        out[i] = (ax, ay, ax2 - ax, ay2 - ay)
                        merged_any = True
                        break
                else:
                    out.append(box)
            boxes = out
        return boxes


class Tracker:
    """贪心 IoU 匹配。够用就好 —— 这个场景目标数很少，上卡尔曼是过度设计。"""

    def __init__(self, iou_threshold: float = 0.15, max_missed: int = 12) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _cxcywh(b: Box):
        x, y, w, h = b
        return (x + w / 2.0, y + h / 2.0, w, h)

    def update(self, boxes: list[Box], ts: float) -> list[Track]:
        for t in self.tracks:
            t.missed += 1

        pairs: list[tuple[float, int, int]] = []
        for i, b in enumerate(boxes):
            for t in self.tracks:
                v = iou(b, t.box)
                if v >= self.iou_threshold:
                    pairs.append((v, i, t.tid))
        pairs.sort(reverse=True)

        by_id = {t.tid: t for t in self.tracks}
        used_b: set[int] = set()
        used_t: set[int] = set()
        for _, i, tid in pairs:
            if i in used_b or tid in used_t:
                continue
            t = by_id[tid]
            t.box = boxes[i]
            t.missed = 0
            t.history.append((ts, *self._cxcywh(boxes[i])))
            used_b.add(i)
            used_t.add(tid)

        for i, b in enumerate(boxes):
            if i in used_b:
                continue
            t = Track(tid=self._next_id, box=b)
            t.history.append((ts, *self._cxcywh(b)))
            self._next_id += 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks
