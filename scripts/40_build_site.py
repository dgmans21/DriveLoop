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


def build_acc_clips(src: Path, skip_video: bool) -> list[dict]:
    """52_acc_scenarios.py --record 영상 → docs/assets/acc/<이름>.mp4 (960px) + 포스터. 이름: <날씨>_<시나리오>_s<시드>_model"""
    order = {"brake": 0, "stopped": 1, "cutin": 2, "follow": 3}
    out_dir = ASSETS / "acc"
    clips = []
    for p in sorted(src.glob("*_model.mp4")) if src.exists() else []:
        stem = p.stem
        parts = stem.split("_")
        cond, scen = "_".join(parts[:2]), parts[2]
        if not skip_video:
            out_dir.mkdir(parents=True, exist_ok=True)
            encode(p, out_dir / f"{stem}.mp4", 960, 28)
            ffmpeg("-ss", "12", "-i", str(p), "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "4",
                   str(out_dir / f"{stem}.jpg"))
        if (out_dir / f"{stem}.mp4").exists():
            clips.append({"scenario": scen, "condition": cond, "src": f"assets/acc/{stem}.mp4",
                          "poster": f"assets/acc/{stem}.jpg"})
    return sorted(clips, key=lambda c: order.get(c["scenario"], 9))


def build_acc(versions: list[str]) -> dict | None:
    """앞차 시나리오(52_acc_scenarios.py) 결과 → 웹용: 최종 버전의 시나리오별 정답값 vs 모델 + 버전별 개선 과정."""
    root = PROJECT_ROOT / "outputs" / "compare"
    out = {"versions": [], "final": []}
    for v in versions:
        runs = sorted((root / v / "runs").glob("*.json"))
        if not runs:
            continue
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
        df = pd.DataFrame(rows)
        m = df[df.perception == "model"]
        speed_err = []
        for p in (root / v / "runs").glob("*_model.csv"):
            log = pd.read_csv(p)
            both = log.lead_dist.notna() & log.p_lead_dist.notna()
            speed_err.append((log.p_lead_speed - log.lead_speed)[both])
        se = pd.concat(speed_err) if speed_err else pd.Series(dtype=float)
        stopped = m[m.scenario == "stopped"]
        out["versions"].append({
            "name": v, "runs": int(len(df)), "collisions": int(df.collisions.sum()),
            "hard_model": int(m.hard_brakes.sum()), "hard_gt": int(df[df.perception == "gt"].hard_brakes.sum()),
            "dist_err": round(float(m.lead_err_med.mean()), 2),
            "speed_err": round(float(se.abs().median()), 2) if len(se) else None,
            "speed_over": round(float((se > 2).mean()), 3) if len(se) else None,
            "stopped_ttc": round(float(stopped.min_ttc.min()), 2) if len(stopped) else None,
        })
        last = df
    agg = last.groupby(["scenario", "perception"]).agg(
        runs=("seed", "size"), collisions=("collisions", "sum"), min_gap=("min_lead_gap", "min"),
        min_ttc=("min_ttc", "min"), hard=("hard_brakes", "sum"), final_gap=("final_gap", "mean"),
        dist_err=("lead_err_med", "mean"), miss=("lead_miss_rate", "mean")).reset_index()
    order = {"follow": 0, "brake": 1, "stopped": 2, "cutin": 3}
    agg = agg.sort_values(["scenario", "perception"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    out["final"] = [{k: (None if pd.isna(x) else (round(float(x), 2) if isinstance(x, float) else x))
                     for k, x in r.items()} for r in agg.to_dict(orient="records")]
    lead_rep = PROJECT_ROOT / "outputs" / "lead_distance" / "report.json"
    if lead_rep.exists():
        rep = json.loads(lead_rep.read_text(encoding="utf-8"))
        out["mono"] = [r for r in rep["table"] if r["method"] == "z_ground"]
    return out if out["versions"] else None


def build_ped_clips(src: Path) -> list[dict]:
    """53_ped_scenarios.py --record 영상 → docs/assets/ped/<이름>.mp4 (960px) + 포스터.
    없는 것만 인코딩 (--skip-video와 무관: 새 영상만 추가되므로 기존 주행 영상 재인코딩 없이 갱신 가능)."""
    order = {"dartout": 0, "stop": 1, "cross": 2, "curb": 3, "sidewalk": 4}
    poster_at = {"dartout": 7.0, "stop": 9.0, "cross": 5.5, "curb": 6.4}
    out_dir = ASSETS / "ped"
    clips = []
    for p in sorted(src.glob("*_model.mp4")) if src.exists() else []:
        stem = p.stem
        parts = stem.split("_")
        cond, scen = "_".join(parts[:2]), parts[2]
        if not (out_dir / f"{stem}.mp4").exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            encode(p, out_dir / f"{stem}.mp4", 960, 28)
            ffmpeg("-ss", str(poster_at.get(scen, 6)), "-i", str(p), "-frames:v", "1", "-vf", "scale=960:-2",
                   "-q:v", "4", str(out_dir / f"{stem}.jpg"))
        if (out_dir / f"{stem}.mp4").exists():
            clips.append({"scenario": scen, "condition": cond, "src": f"assets/ped/{stem}.mp4",
                          "poster": f"assets/ped/{stem}.jpg"})
    return sorted(clips, key=lambda c: order.get(c["scenario"], 9))


def build_ped(versions: list[str], detect_runs: tuple[str, str]) -> dict | None:
    """보행자 시나리오(53_ped_scenarios.py) → 웹용: 최종 버전 시나리오별 정답값 vs 모델 + 버전별 개선 + 모델 성적표."""
    root = PROJECT_ROOT / "outputs" / "compare"
    out = {"versions": [], "final": [], "detect": None}
    last = None
    gt_by_version = {}
    for v in versions:
        runs = sorted((root / v / "runs").glob("*.json"))
        if not runs:
            continue
        df = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in runs])
        m, g = df[df.perception == "model"], df[df.perception == "gt"]
        if len(g):
            gt_by_version[v] = g
        out["versions"].append({
            "name": v, "runs": int(len(df)), "collisions": int(df.collisions.sum()),
            "hard_model": int(m.hard_brakes.sum()), "hard_gt": int(g.hard_brakes.sum()) if len(g) else None,
            "max_decel_model": round(float(m.max_decel.max()), 1),
            "max_decel_gt": round(float(g.max_decel.max()), 1) if len(g) else None,
            "dart_gap": round(float(m[m.scenario == "dartout"].min_ped_gap.min()), 1),
        })
        last = df
    if last is None:
        return None
    agg = last.groupby(["scenario", "perception"]).agg(
        runs=("seed", "size"), collisions=("collisions", "sum"), min_gap=("min_ped_gap", "min"),
        hard=("hard_brakes", "sum"), max_decel=("max_decel", "max"), min_speed=("min_speed_after_cruise", "min"),
        yield_s=("yield_s", "mean"), miss=("ped_miss_rate", "mean"), first_seen=("first_seen_gap", "mean")).reset_index()
    order = {"cross": 0, "stop": 1, "dartout": 2, "curb": 3, "sidewalk": 4}
    agg = agg.sort_values(["scenario", "perception"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    out["final"] = [{k: (None if pd.isna(x) else (round(float(x), 2) if isinstance(x, float) else x))
                     for k, x in r.items()} for r in agg.to_dict(orient="records")]
    # 모델 성적표: 같은 시험지(원본 v2 = v3 테스트와 같은 Town05 814장)에서 v3 vs v4 + v4 전체 테스트의 보행자 거리별
    old, new = (PROJECT_ROOT / "runs" / "detect" / r / "eval" for r in detect_runs)
    try:
        a = json.loads((old / "test_v2src.json").read_text(encoding="utf-8"))
        b = json.loads((new / "test_v2src.json").read_text(encoding="utf-8"))
        full = json.loads((new / "test.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        a = None
    if a:
        cls = ["tl_red", "tl_yellow", "tl_green", "vehicle"]
        out["detect"] = {
            "images": b["images"],
            "same": [{"cls": c, "old": a["recall_by_class"][c]["recall"], "new": b["recall_by_class"][c]["recall"],
                      "old_fp": a["false_positives"].get(c, 0), "new_fp": b["false_positives"].get(c, 0),
                      "n": a["recall_by_class"][c]["n"]} for c in cls],
            "ped_size": [{"bin": k.split("|", 1)[1], **v} for k, v in full["recall_by_size"].items()
                         if k.startswith("pedestrian|")],
            "ped_recall": full["recall_by_class"]["pedestrian"]["recall"],
            "ped_phantom_v2src": b["false_positives"].get("pedestrian", 0),
        }
        far = {"<24 (45m+)": 0, "24–40 (27–45m)": 1, "40–80 (14–27m)": 2, "80–160 (7–14m)": 3, "160+ (<7m)": 4}
        out["detect"]["ped_size"].sort(key=lambda r: -far.get(r["bin"], 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-root", default=str(PROJECT_ROOT / "outputs" / "web"))
    ap.add_argument("--highlight", default="clear_noon:122:137", help="조건:시작초:끝초 (오버레이 녹화에서 자름)")
    ap.add_argument("--skip-video", action="store_true", help="JSON만 갱신 (영상 재압축 생략)")
    ap.add_argument("--acc-versions", default="acc_scen_v2,acc_scen_v3,acc_scen_v4,acc_scen_v5",
                    help="앞차 시나리오 결과 (개선 과정 순서, 마지막이 최종)")
    ap.add_argument("--ped-versions", default="ped_scen_model_v1,ped_scen_model_v2,ped_scen_model_v3,ped_scen_model_v4",
                    help="보행자 시나리오 결과 (개선 과정 순서, 마지막이 최종)")
    ap.add_argument("--ped-clips", default="ped_clips_v4", help="outputs/compare/<이름>/clips 의 녹화 영상")
    ap.add_argument("--compare", default="compare_v2",
                    help="outputs/compare/<이름>/summary.json 을 웹에 넣는다 (--baseline으로 만든 전/후 비교 포함)")
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
        web = {k: s.get(k) for k in ("name", "map", "seconds", "seeds", "runs", "total", "baseline")}
        web["by_condition"] = [{k: r[k] for k in keep} for r in s["by_condition"]]
        (ASSETS / "compare.json").write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[site] compare ← {cmp_src}")
    else:
        print(f"[site] 비교 결과 없음: {cmp_src} (51_compare_report.py 실행)")

    acc = build_acc(args.acc_versions.split(","))
    if acc:
        acc["clips"] = build_acc_clips(PROJECT_ROOT / "outputs" / "compare" / "acc_clips" / "clips", args.skip_video)
        (ASSETS / "acc.json").write_text(json.dumps(acc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[site] acc ← {', '.join(v['name'] for v in acc['versions'])}")

    ped = build_ped(args.ped_versions.split(","), ("v3_base", "v4_base"))
    if ped:
        ped["clips"] = build_ped_clips(PROJECT_ROOT / "outputs" / "compare" / args.ped_clips / "clips")
        (ASSETS / "ped.json").write_text(json.dumps(ped, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[site] ped ← {', '.join(v['name'] for v in ped['versions'])}, 영상 {len(ped['clips'])}개")

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
