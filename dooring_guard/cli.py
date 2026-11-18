"""命令行入口。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2

from . import render
from .detect import MotionDetector, Tracker
from .risk import LEVEL_RANK, Level, RiskAssessor, RiskConfig

# 默认关注区域：画面右侧（模拟从车内看向右后方来车方向）。
# 归一化坐标，顺序无所谓，会自动闭合成多边形。
DEFAULT_ROI = [(0.20, 0.10), (1.0, 0.10), (1.0, 1.0), (0.20, 1.0)]

ROOT = Path(__file__).resolve().parent.parent


def parse_roi(text: str | None):
    if text is None:
        return DEFAULT_ROI
    if text.strip().lower() in {"none", "off", ""}:
        return None
    pts = []
    for token in text.replace(";", " ").split():
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise SystemExit("--roi 至少需要 3 个点，例如 '0.2,0.1 1,0.1 1,1 0.2,1'")
    return pts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dooring-guard",
        description="开门杀风险预警：从视频里判断后方来车是否会撞上正在打开的车门。",
    )
    p.add_argument("--source", default=None, help="视频文件路径，或摄像头编号（如 0）。留空则用合成测试视频。")
    p.add_argument("--out", default="out", help="输出目录，默认 out/")
    p.add_argument("--engine", choices=["motion", "yolo"], default="motion",
                   help="motion=零依赖运动检测（默认）；yolo=需安装 ultralytics")
    p.add_argument("--roi", default=None,
                   help="归一化关注区域，如 '0.2,0.1 1,0.1 1,1 0.2,1'；传 none 关闭")
    p.add_argument("--near", type=float, default=0.50, help="「贴近车门」对应的目标高度占画面比例")
    p.add_argument("--far", type=float, default=0.06, help="「远处」对应的目标高度占画面比例")
    p.add_argument("--danger-ttc", type=float, default=1.5, help="触发 DANGER 的碰撞时间阈值（秒）")
    p.add_argument("--warn-ttc", type=float, default=4.0, help="触发 CAUTION 的碰撞时间阈值（秒）")
    p.add_argument("--no-video", action="store_true", help="不写标注视频（跑得更快）")
    p.add_argument("--demo", action="store_true", help="用合成测试视频跑一遍，不需要任何素材")
    p.add_argument("--display", action="store_true", help="实时显示窗口，按 q 退出")
    return p


def ensure_demo(path: Path) -> Path:
    if not path.exists():
        print(f"[demo] 未找到 {path}，正在生成合成测试视频 ...")
        sys.path.insert(0, str(ROOT))
        from tools.make_demo_video import make_demo

        path.parent.mkdir(parents=True, exist_ok=True)
        make_demo(path)
    return path


def resolve_source(src: str | None, demo_dir: Path):
    if src is None or src.lower() == "demo":
        return str(ensure_demo(demo_dir / "assets" / "demo.mp4")), True
    if src.isdigit():
        return int(src), False
    path = Path(src)
    if not path.exists():
        raise SystemExit(f"找不到视频文件：{src}")
    return str(path), False


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roi = parse_roi(args.roi)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    source, is_demo = resolve_source("demo" if args.demo else args.source, ROOT)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频源：{source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3 or fps > 240:
        fps = 30.0

    if args.engine == "yolo":
        from .yolo_engine import YoloDetector

        detector = YoloDetector(roi=roi)
    else:
        detector = MotionDetector(roi=roi)

    tracker = Tracker()
    assessor = RiskAssessor(RiskConfig(
        near_height_ratio=args.near,
        far_height_ratio=args.far,
        danger_ttc=args.danger_ttc,
        warn_ttc=args.warn_ttc,
    ))

    writer = None
    events: list[dict] = []
    peak = Level.SAFE
    min_ttc = math.inf
    best_frame = None
    frames = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        ts = idx / fps
        idx += 1
        frames += 1

        if writer is None and not args.no_video:
            writer = cv2.VideoWriter(
                str(outdir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        boxes = detector(frame)
        tracks = tracker.update(boxes, ts)

        risks = {}
        for t in tracks:
            r = assessor.update(t.tid, ts, t.box[3], h)
            risks[t.tid] = r
            if r.ttc < min_ttc:
                min_ttc = r.ttc
            if LEVEL_RANK[r.level] > LEVEL_RANK[peak]:
                peak = r.level
            if r.changed and r.level is not Level.SAFE:
                events.append({
                    "t": round(ts, 2), "track": t.tid, "level": r.level.value,
                    "distance_ratio": round(r.distance_ratio, 3),
                    "ttc": "" if r.ttc == math.inf else round(r.ttc, 2),
                })
        assessor.prune({t.tid for t in tracks})

        render.draw_roi(frame, roi)
        render.draw_tracks(frame, tracks, risks)
        top = max(risks.values(), key=lambda r: LEVEL_RANK[r.level], default=None)
        render.draw_banner(
            frame,
            top.level if top else Level.SAFE,
            top.ttc if top else math.inf,
            top.distance_ratio if top else 0.0,
        )
        render.draw_hud(frame, len(tracks), f"{fps:.0f}fps")

        if top and top.level is Level.DANGER and best_frame is None:
            best_frame = frame.copy()

        if writer is not None:
            writer.write(frame)
        if args.display:
            cv2.imshow("DooringGuard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[out] 标注视频 -> {outdir / 'annotated.mp4'}")
    if args.display:
        cv2.destroyAllWindows()

    csv_path = outdir / "events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["t", "track", "level", "distance_ratio", "ttc"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(events)
    print(f"[out] 事件记录 -> {csv_path}")

    if best_frame is not None:
        cv2.imwrite(str(outdir / "keyframe.jpg"), best_frame)
        print(f"[out] 危险关键帧 -> {outdir / 'keyframe.jpg'}")

    n_danger = sum(1 for e in events if e["level"] == "DANGER")
    n_warn = sum(1 for e in events if e["level"] == "WARN")
    print("-" * 52)
    print(f"处理帧数        {frames}（约 {frames / fps:.1f} 秒）")
    print(f"告警事件        {len(events)} 次（CAUTION {n_warn} / DANGER {n_danger}）")
    print(f"最高风险等级    {peak.value}")
    if min_ttc != math.inf:
        print(f"最小碰撞时间    {min_ttc:.2f} 秒")
    if is_demo:
        print("注：当前跑的是合成测试视频，只验证 pipeline 是否连通，不代表真实场景精度。")
    return 0


def main() -> int:
    return run(sys.argv[1:])
