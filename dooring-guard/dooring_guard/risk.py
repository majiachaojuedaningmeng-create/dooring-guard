"""风险分级。

核心假设：在这个机位下，物体在画面里的「高度」可以当作「距离」的代理量。
远处小、近处大 —— 不精确，但对于「有没有东西正在冲过来」这个问题足够了。

真正上车的版本应该换掉这一层：用单目测距（已知车牌/车轮尺寸）或双目深度，
这里刻意保持可解释、可手算，方便在答辩时讲清楚。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Level(str, Enum):
    SAFE = "SAFE"
    WARN = "WARN"
    DANGER = "DANGER"


LEVEL_RANK = {Level.SAFE: 0, Level.WARN: 1, Level.DANGER: 2}


@dataclass
class RiskConfig:
    far_height_ratio: float = 0.06   # 占画面高度的比例，视为「远点」
    near_height_ratio: float = 0.50  # 视为「贴近车门」
    danger_ttc: float = 1.5          # 秒
    warn_ttc: float = 4.0            # 秒
    alpha: float = 0.30              # 距离平滑系数
    min_samples: int = 6             # 轨迹稳定多少帧后才开始判风险
    max_speed: float = 3.0           # 逼近速度上限（单位：距离比例/秒），防抖动炸出假 TTC

    # 滞回阈值：已经 WARN 了要更宽松的条件才会掉回 SAFE，否则告警会疯狂闪烁
    stay_warn_distance: float = 0.40
    stay_warn_ttc: float = 6.0
    stay_danger_distance: float = 0.78
    stay_danger_ttc: float = 2.5
    danger_distance: float = 0.92


@dataclass
class RiskState:
    tid: int
    level: Level
    distance_ratio: float
    approach_speed: float
    ttc: float
    changed: bool = False


class RiskAssessor:
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()
        self._state: dict[int, dict] = {}

    def update(self, tid: int, ts: float, box_h: float, frame_h: float) -> RiskState:
        c = self.cfg
        near = max(1.0, c.near_height_ratio * frame_h)
        far = max(0.0, c.far_height_ratio * frame_h)
        raw = float(np.clip((box_h - far) / max(1e-6, near - far), 0.0, 1.15))

        st = self._state.get(tid)
        if st is None:
            self._state[tid] = {
                "d": raw,
                "ts": ts,
                "speed": 0.0,
                "level": Level.SAFE,
                "n": 1,
            }
            return RiskState(tid, Level.SAFE, raw, 0.0, math.inf, False)

        dt = ts - st["ts"]
        d = st["d"] * (1.0 - c.alpha) + raw * c.alpha
        inst = (d - st["d"]) / dt if dt > 1e-3 else st["speed"]
        speed = 0.5 * inst + 0.5 * st["speed"] if st["speed"] else inst
        speed = min(speed, c.max_speed)

        # 刚出现的轨迹先不给结论：头几帧的距离变化率全是噪声
        if st["n"] < c.min_samples:
            st.update(d=d, ts=ts, speed=0.0, n=st["n"] + 1)
            return RiskState(tid, Level.SAFE, d, 0.0, math.inf, False)

        ttc = max(0.0, (1.0 - d) / speed) if speed > 1e-4 else math.inf
        level = self._classify(level=st["level"], d=d, ttc=ttc, speed=speed)

        changed = level is not st["level"]
        st.update(d=d, ts=ts, speed=speed, level=level, n=st["n"] + 1)
        return RiskState(tid, level, d, speed, ttc, changed)

    def _classify(self, level: Level, d: float, ttc: float, speed: float) -> Level:
        c = self.cfg
        if speed <= 1e-4:  # 没有在靠近，谈不上风险
            return Level.SAFE

        if level is Level.DANGER:
            if d >= c.stay_danger_distance or ttc <= c.stay_danger_ttc:
                return Level.DANGER
        elif level is Level.WARN:
            if d >= c.danger_distance or ttc <= c.danger_ttc:
                return Level.DANGER
            if d >= c.stay_warn_distance or ttc <= c.stay_warn_ttc:
                return Level.WARN
        else:
            if d >= c.danger_distance or ttc <= c.danger_ttc:
                return Level.DANGER
            if d >= 0.55 or ttc <= c.warn_ttc:
                return Level.WARN
        return Level.SAFE

    def prune(self, active_ids: set[int]) -> None:
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]
