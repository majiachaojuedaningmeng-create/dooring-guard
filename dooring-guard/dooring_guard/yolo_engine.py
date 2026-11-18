"""可选的 YOLO 引擎。

只有在装了 ultralytics 时才可用：
    pip install ultralytics

它比运动检测多做两件事：
1. 区分 人是人、车是车（运动检测只会说「有东西在动」）
2. 静态画面里也能检出目标（运动检测做不到）

代价是首次运行要下载权重、推理更吃算力 —— 迁移到 RDK X5 时应该走
官方的模型转换工具链（.pt -> .bin）跑 BPU，而不是在 CPU 上硬扛。
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

VEHICLE_AND_PERSON = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


class YoloDetector:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        conf: float = 0.30,
        classes: Sequence[int] | None = None,
        roi: Sequence[Sequence[float]] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "未安装 ultralytics。执行 `pip install ultralytics`，"
                "或改用默认的 --engine motion。"
            ) from exc

        self.model = YOLO(model)
        self.conf = conf
        self.classes = list(classes or VEHICLE_AND_PERSON)
        self.roi = [(float(p[0]), float(p[1])) for p in roi] if roi else None

    def _inside(self, cx: float, cy: float, shape) -> bool:
        if self.roi is None:
            return True
        h, w = shape[:2]
        pts = np.array([[x * w, y * h] for x, y in self.roi], np.float32)
        return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0

    def __call__(self, frame) -> list[tuple[int, int, int, int]]:
        result = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)[0]
        boxes: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy().astype(int):
            w, h = int(x2 - x1), int(y2 - y1)
            if h < 12:
                continue
            if not self._inside(x1 + w / 2, y1 + h / 2, frame.shape):
                continue
            boxes.append((int(x1), int(y1), w, h))
        return boxes
