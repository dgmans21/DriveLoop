"""3단계 분석: 50_compare_drive.py의 실행 로그 → 조건 × 인지별 비교 표 + 모델만의 사건 목록 (CARLA·GPU 불필요).

    python scripts/51_compare_report.py                  # configs/eval/compare_v1.yaml
    python scripts/51_compare_report.py --events 10      # 사건 목록 길이

요약은 실행 시점 JSON이 아니라 CSV에서 다시 계산한다 (지표 코드를 고치면 바로 반영).
출력: <output_root>/<name>/summary.json, runs.csv  (웹 페이지는 40_build_site.py가 summary.json을 가져간다)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from driveloop.config import PROJECT_ROOT, load_driving_config, load_sim_config
from driveloop.eval.driving import nan_to_none, summarize_run

COND_ORDER = ["clear_noon", "rain_noon", "clear_night", "rain_night"]


def load_runs(run_dir: Path, dt: float, stop_margin: float) -> tuple[pd.DataFrame, dict]:
    rows, logs = [], {}
    for csv in sorted(run_dir.glob("*.csv")):
        cond, seed, perc = csv.stem.rsplit("_", 2)
        log = pd.read_csv(csv)
        meta = json.loads(csv.with_suffix(".json").read_text(encoding="utf-8")) if csv.with_suffix(".json").exists() else {}
        s = nan_to_none(summarize_run(log, dt, stop_margin))
        s.update({"condition": cond, "seed": int(seed[1:]), "perception": perc,
                  "collisions": meta.get("collisions"), "wall_s": meta.get("wall_s"),
                  "infer_ms_median": meta.get("infer_ms_median")})
        rows.append(s)
        logs[(cond, int(seed[1:]), perc)] = log
    return pd.DataFrame(rows), logs


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["condition", "perception"])
    t = g.agg(runs_n=("seed", "size"), red_violations=("red_violations", "sum"), stops=("stops", "sum"),
              proper_stops=("proper_stops", "sum"), false_stops=("false_stops", "sum"),
              false_brakes=("false_brakes", "sum"), stop_over_line=("stop_over_line", "sum"),
              caution_s=("caution_s", "sum"),
              stop_err_mean=("stop_err_mean", "mean"), start_delay_mean=("start_delay_mean", "mean"),
              decision_agree=("decision_agree", "mean"), collisions=("collisions", "sum"),
              distance_m=("distance_m", "sum")).reset_index()
    t["condition"] = pd.Categorical(t.condition, [c for c in COND_ORDER if c in set(t.condition)] +
                                    sorted(set(t.condition) - set(COND_ORDER)))
    return t.sort_values(["condition", "perception"]).round(3)


def model_events(logs: dict, limit: int) -> list[dict]:
    """모델 실행에서 정답값과 다르게 행동한 순간: 불필요한 감속 / 판단 불일치 구간 (영상·로그 확인용)."""
    out = []
    for (cond, seed, perc), log in logs.items():
        if perc != "model":
            continue
        s = log.state
        entering = (s == "STOPPING") & (s.shift() != "STOPPING") & ~log.gt_state.isin(["RED", "YELLOW"])
        for i in log.index[entering]:
            out.append({"kind": "false_brake", "condition": cond, "seed": seed, "t": float(log.t[i]),
                        "model": log.tl_state[i], "truth": log.gt_state[i], "raw": log.raw_state[i],
                        "dist": None if pd.isna(log.gt_dist[i]) else float(log.gt_dist[i])})
        # 정답 신호가 있는데 판단이 다른 연속 구간 (UNKNOWN 포함) 중 긴 것
        bad = log.gt_state.notna() & (log.tl_state != log.gt_state)
        grp = (bad != bad.shift()).cumsum()
        for _, seg in log[bad].groupby(grp[bad]):
            if len(seg) >= 10:            # 0.5초 이상
                out.append({"kind": "mismatch", "condition": cond, "seed": seed, "t": float(seg.t.iloc[0]),
                            "seconds": round(len(seg) * 0.05, 2), "model": seg.tl_state.mode().iloc[0]
                            if seg.tl_state.notna().any() else None, "truth": seg.gt_state.iloc[0],
                            "dist": None if pd.isna(seg.gt_dist.iloc[0]) else float(seg.gt_dist.iloc[0])})
    out.sort(key=lambda e: (e["kind"] != "false_brake", -e.get("seconds", 0)))
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval" / "compare_v1.yaml"))
    ap.add_argument("--events", type=int, default=15)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = PROJECT_ROOT / cfg["output_root"] / cfg["name"]
    dt = load_sim_config().fixed_delta_seconds
    drv = load_driving_config()
    runs, logs = load_runs(base / "runs", dt, drv.stop_margin)
    if runs.empty:
        raise SystemExit(f"실행 로그 없음: {base / 'runs'}")
    table = aggregate(runs)
    total = runs.groupby("perception")[["red_violations", "stops", "proper_stops", "false_stops", "false_brakes",
                                         "stop_over_line", "caution_s", "collisions"]].sum()
    total["start_delay_mean"] = runs.groupby("perception").start_delay_mean.mean().round(3)
    total["decision_agree"] = runs.groupby("perception").decision_agree.mean().round(3)
    events = model_events(logs, args.events)

    runs.to_csv(base / "runs.csv", index=False)
    summary = {"name": cfg["name"], "map": cfg["map"], "seconds": cfg["seconds"], "seeds": cfg["seeds"],
               "runs": len(runs), "by_condition": table.astype({"condition": str}).to_dict(orient="records"),
               "total": {p: nan_to_none(r.to_dict()) for p, r in total.iterrows()}, "events": events}
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"[compare] {cfg['name']}: {len(runs)}회 ({cfg['map']}, 각 {cfg['seconds']}초)\n")
    print(table.to_string(index=False))
    print("\n-- 전체\n" + total.to_string())
    print(f"\n-- 모델만의 사건 (상위 {len(events)})")
    for e in events:
        print("  ", e)
    print(f"\n[compare] → {base / 'summary.json'}")


if __name__ == "__main__":
    main()
