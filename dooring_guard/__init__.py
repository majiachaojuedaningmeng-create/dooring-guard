"""DooringGuard — 开门杀风险预警。

纯视觉、无需硬件、无需训练模型即可跑通的最小实现。
术语：dooring = 车门开启瞬间撞击后方来车/骑行者，中文俗称「开门杀」。
"""

__version__ = "0.1.0"
__all__ = ["detect", "risk", "render", "cli"]
