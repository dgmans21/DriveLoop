"""결과 웹페이지(docs/, GitHub Pages용) 에셋 만들기 — CARLA 서버 없이 동작 (GPU 불필요).

    python scripts/40_build_site.py                       # outputs/web/ 의 조건별 최신 주행으로 docs/assets 갱신
    python scripts/40_build_site.py --highlight clear_noon:122:137

입력 : outputs/web/<tag>/ (04_model_drive.py --web), outputs/drive_<tag>.csv, outputs/videos/drive_<tag>.mp4
출력 : docs/assets/drive/<조건>/{front.mp4, chase.mp4, poster.jpg, frames.json}
       docs/assets/drive/index.json   조건 목록 + 요약 수치
       docs/assets/maps/<맵>.json      2D 지도용 차선 중심선 (OpenDRIVE를 서버 없이 파싱)
       docs/assets/highlight.mp4       첫 화면 하이라이트 (오버레이가 그려진 녹화에서 자름)
       docs/assets/compare.json        3단계 정답값 vs 모델 주행 비교 (51_compare_report.py 결과)
웹 용량: 전방 960px / 3인칭 480px 로 다시 압축 (git·Pages에 올릴 크기)
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from driveloop.config import PROJECT_ROOT

CONDITIONS = ["clear_noon", "rain_noon", "clear_night", "rain_night"]
DOCS = PROJECT_ROOT / "docs"
ASSETS = DOCS / "assets"


def ffmpeg(*args: str) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg가 필요합니다 (PATH)")
    subprocess.run([exe, "-y", "-loglevel", "error", *args], check=True)


def encode(src: Path, dst: Path, width: int, crf: int, start: float | None = None, end: float | None = None) -> None:
    cut = ["-ss", f"{start}", "-to", f"{end}"] if start is not None else []
    ffmpeg(*cut, "-i", str(src), "-vf", f"scale={width}:-2", "-c:v", "libx264", "-crf", str(crf),
           "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst))


def latest_runs(web_root: Path) -> dict[str, Path]:
    """조건별 가장 최근 웹 출력 폴더 (이름: model_<맵>_<조건>_s<seed>_<시각>)."""
    runs = {}
    for d in sorted(web_root.iterdir()):
        for c in CONDITIONS:
            if f"_{c}_" in d.name and (d / "frames.json").exists():
                runs[c] = d   # 정렬 순서상 마지막 = 최신
    return runs


def headings(xs: list[float], ys: list[float], min_step: float = 0.3) -> list[float]:
    """연속 위치로 진행 방향(도). 정지 중에는 직전 방향 유지."""
    out, last = [], 0.0
    for i in range(len(xs)):
        j = min(i + 5, len(xs) - 1)
        dx, dy = xs[j] - xs[i], ys[j] - ys[i]
        if math.hypot(dx, dy) > min_step:
            last = math.degrees(math.atan2(dy, dx))
        out.append(round(last, 1))
    return out


def stop_lines(frames: list[dict]) -> list[dict]:
    """정지선 위치 추정: 신호 접근 구간마다 정지선 거리가 가장 작은 프레임에서 진행 방향으로 그 거리만큼 앞."""
    lines, seg = [], []

    def flush():
        if seg:
            f = min(seg, key=lambda r: abs(r["d"]))
            yaw = math.radians(f["yaw"])
            lines.append({"x": round(f["x"] + f["d"] * math.cos(yaw), 2), "y": round(f["y"] + f["d"] * math.sin(yaw), 2),
                          "yaw": f["yaw"], "t0": seg[0]["t"], "t1": seg[-1]["t"]})

    prev = None
    for f in frames:
        d = f.get("d")
        if d is None or (prev is not None and d > prev + 5):   # 신호 없음 / 다음 신호로 바뀜
            flush()
            seg = []
        if d is not None:
            seg.append(f)
        prev = d
    flush()
    return lines


def build_map(map_name: str, frames_xy: list[tuple[float, float]], margin: float = 60.0) -> dict:
    """주행 범위(+여유) 안의 차선 중심선. CARLA 클라이언트가 OpenDRIVE를 오프라인으로 파싱한다."""
    import carla

    xodr = Path(os.environ.get("CARLA_ROOT", r"D:\CARLA_0.9.16")) / "CarlaUE4/Content/Carla/Maps/OpenDrive" / f"{map_name}.xodr"
    m = carla.Map(map_name, xodr.read_text(encoding="utf-8"))
    xs, ys = zip(*frames_xy)
    x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    lanes = defaultdict(list)
    for w in m.generate_waypoints(1.0):
        p = w.transform.location
        if x0 <= p.x <= x1 and y0 <= p.y <= y1:
            lanes[(w.road_id, w.section_id, w.lane_id)].append((w.s, round(p.x, 1), round(p.y, 1), w.lane_width, w.is_junction))
    out = []
    for pts in lanes.values():
        pts.sort()
        if len(pts) >= 2:
            out.append({"w": round(pts[0][3], 2), "j": pts[0][4], "p": [[x, y] for _, x, y, _, _ in pts]})
    return {"map": map_name, "bbox": [x0, y0, x1, y1], "lanes": out}


def summarize(csv: pd.DataFrame) -> dict:
    s = csv[csv.gt_state.notna()]
    r = s[s.raw_state.notna()]
    return {
        "agree": round(float((s.tl_state == s.gt_state).mean()), 3),
        "found": round(len(r) / max(len(s), 1), 3),
        "color": round(float((r.raw_state == r.gt_state).mean()), 3),
        "red_run": int(((csv.gt_state == "RED") & (csv.stop_dist < 0) & (csv.speed > 1)).sum()),
        "stops": int(((csv.state == "STOPPED") & (csv.state.shift() != "STOPPED")).sum()),
        "infer_ms": round(float(csv.infer_ms.median()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-root", default=str(PROJECT_ROOT / "outputs" / "web"))
    ap.add_argument("--highlight", default="clear_noon:122:137", help="조건:시작초:끝초 (오버레이 녹화에서 자름)")
    ap.add_argument("--skip-video", action="store_true", help="JSON만 갱신 (영상 재압축 생략)")
    ap.add_argument("--compare", default="compare_v1", help="outputs/compare/<이름>/summary.json 을 웹에 넣는다")
    args = ap.parse_args()

    runs = latest_runs(Path(args.web_root))
    missing = [c for c in CONDITIONS if c not in runs]
    if missing:
        print(f"[site] 없는 조건: {missing} (04_model_drive.py --web --weather .. --time .. 로 녹화)")
    (ASSETS / "drive").mkdir(parents=True, exist_ok=True)
    (ASSETS / "maps").mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").touch()

    index, maps_done = [], set()
    for cond in CONDITIONS:
        if cond not in runs:
            continue
        run = runs[cond]
        data = json.loads((run / "frames.json").read_text(encoding="utf-8"))
        csv = pd.read_csv(PROJECT_ROOT / "outputs" / f"drive_{run.name}.csv")
        frames = data["frames"]
        assert len(frames) == len(csv), f"{run.name}: frames {len(frames)} != csv {len(csv)}"
        if frames and frames[0].get("ms", 0) > 500:
            frames[0]["ms"] = None   # 첫 추론은 모델 워밍업(CUDA 초기화) 포함 → 화면 표시에서 제외
        yaws = headings(csv.x.tolist(), csv.y.tolist())
        for f, x, y, yaw in zip(frames, csv.x, csv.y, yaws):
            f.update({"x": round(float(x), 2), "y": round(float(y), 2), "yaw": yaw})
        meta = {**data["meta"], "weights": Path(data["meta"]["weights"]).parent.parent.name if data["meta"]["weights"] else None}
        out = ASSETS / "drive" / cond
        out.mkdir(exist_ok=True)
        (out / "frames.json").write_text(json.dumps({"meta": meta, "stops": stop_lines(frames), "frames": frames},
                                                    ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if not args.skip_video:
            encode(run / "front.mp4", out / "front.mp4", 960, 28)
            encode(run / "chase.mp4", out / "chase.mp4", 480, 30)
            ffmpeg("-ss", "10", "-i", str(run / "front.mp4"), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "4",
                   str(out / "poster.jpg"))
        if meta["map"] not in maps_done:
            mp = build_map(meta["map"], [(f["x"], f["y"]) for f in frames])
            (ASSETS / "maps" / f"{meta['map']}.json").write_text(json.dumps(mp, separators=(",", ":")), encoding="utf-8")
            maps_done.add(meta["map"])
        index.append({"id": cond, "map": meta["map"], "seconds": round(frames[-1]["t"]), **summarize(csv)})
        print(f"[site] {cond:12} ← {run.name}  {index[-1]}")

    (ASSETS / "drive" / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    # 3단계 비교 (51_compare_report.py의 summary.json) → 웹에 필요한 부분만
    cmp_src = PROJECT_ROOT / "outputs" / "compare" / args.compare / "summary.json"
    if cmp_src.exists():
        s = json.loads(cmp_src.read_text(encoding="utf-8"))
        keep = ["condition", "perception", "runs_n", "red_violations", "stops", "proper_stops", "false_stops",
                "false_brakes", "stop_over_line", "stop_err_mean", "start_delay_mean", "decision_agree", "collisions"]
        web = {k: s[k] for k in ("name", "map", "seconds", "seeds", "runs", "total")}
        web["by_condition"] = [{k: r[k] for k in keep} for r in s["by_condition"]]
        (ASSETS / "compare.json").write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[site] compare ← {cmp_src}")
    else:
        print(f"[site] 비교 결과 없음: {cmp_src} (51_compare_report.py 실행)")

    if not args.skip_video and args.highlight:
        cond, a, b = args.highlight.split(":")
        if cond in runs:
            src = PROJECT_ROOT / "outputs" / "videos" / f"drive_{runs[cond].name}.mp4"
            encode(src, ASSETS / "highlight.mp4", 960, 27, float(a), float(b))
            print(f"[site] highlight ← {src.name} {a}–{b}s")

    total = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    print(f"[site] docs/assets 총 {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
