"""3단계: 정답값 인지(gt) vs 내 모델 인지(model) 일괄 주행 → 실행마다 tick 로그 CSV + 요약 JSON.

    python scripts/50_compare_drive.py --dry-run                          # 실행 계획만
    python scripts/50_compare_drive.py --only clear_noon --limit 2        # 파일럿
    python scripts/50_compare_drive.py --only clear_noon,rain_noon        # 조건 일부 (끝난 실행은 건너뜀)

공정한 비교를 위해 실행마다
  - 같은 시드 → 같은 출발 지점·같은 경로 선택 (RoutePlanner의 rng)
  - world.reset_all_traffic_lights() → 같은 신호 주기에서 출발
  - 판단·제어는 DrivingAgent (주행 데모 04와 같은 코드)
화면·녹화 없음 (로그만). 정답값 실행은 카메라를 붙이지 않아 빠르다.
결과: <output_root>/<name>/runs/<조건>_s<시드>_<인지>.csv / .json  (끝난 실행은 다시 돌리지 않음)
"""
import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import carla
import pandas as pd

from driveloop.agent import DrivingAgent
from driveloop.config import PROJECT_ROOT, CameraConfig, load_config, load_driving_config, load_sim_config
from driveloop.data.collection_config import CaptureCamera, load_collection_config
from driveloop.eval.driving import nan_to_none, summarize_run
from driveloop.sim.client import ActorPool, connect, spawn_ego, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame
from driveloop.sim.weather import apply_weather

LOG_COLS = ["t", "x", "y", "speed", "state", "target_speed", "tl_state", "raw_state", "gt_state", "gt_dist",
            "gt_tl", "infer_ms"]


@dataclass
class CompareConfig:
    name: str
    map: str
    conditions: list[list[str]]
    seeds: list[int]
    seconds: float
    perceptions: list[str] = field(default_factory=lambda: ["gt", "model"])
    conditions_source: str = "configs/collection/v3.yaml"
    weights: str = "runs/detect/v3_base/weights/best.pt"
    conf: float = 0.10
    output_root: str = "outputs/compare"


def plan(cfg: CompareConfig, only: set[str] | None) -> list[tuple[str, list[str], int, str]]:
    """(조건 id, [날씨, 시간], 시드, 인지). 조건 → 시드 → 인지 순: 같은 시드의 gt/model이 연달아 돈다."""
    runs = []
    for w, tod in cfg.conditions:
        cid = f"{w}_{tod}"
        if only and cid not in only:
            continue
        for s in cfg.seeds:
            for p in cfg.perceptions:
                runs.append((cid, [w, tod], s, p))
    return runs


def run_one(client, world, sim_cfg, drv, cfg: CompareConfig, seed: int, perception: str, csv_path: Path) -> dict:
    dt = sim_cfg.fixed_delta_seconds
    front = CaptureCamera()
    rng = random.Random(seed)
    sim_cfg.seed = seed
    collisions: list[str] = []
    with ActorPool() as pool:
        ego, spawn = spawn_ego(world, sim_cfg, rng)
        pool.add(ego)
        if world.get_weather().sun_altitude_angle < 0:
            ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position
                                                        | carla.VehicleLightState.LowBeam))
        col = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(),
                                attach_to=ego)
        pool.add(col)
        col.listen(lambda e: collisions.append(e.other_actor.type_id))
        cam_q = None
        if perception == "model":
            cam, cam_q = attach_rgb_camera(world, ego, CameraConfig(front.width, front.height, front.fov),
                                           carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(cam)

            def make(route, _gt):
                from driveloop.perception.model import ModelPerception
                return ModelPerception(world, ego, route, cam, front.width, front.height, front.fov,
                                       str(PROJECT_ROOT / cfg.weights), drv.tl_lookahead, conf=cfg.conf)
        else:
            make = None
        world.reset_all_traffic_lights()       # 모든 실행이 같은 신호 주기에서 출발
        world.tick()
        agent = DrivingAgent(world, ego, drv, rng, make)

        rows = []
        t0 = world.get_snapshot().timestamp.elapsed_seconds
        wall = time.perf_counter()
        while True:
            frame = world.tick()
            img = get_frame(cam_q, frame) if cam_q is not None else None
            t = world.get_snapshot().timestamp.elapsed_seconds - t0
            if t >= cfg.seconds:
                break
            r = agent.step(img, dt)
            dbg = getattr(agent.perception, "debug", None)
            rows.append([round(t, 2), round(r.transform.location.x, 2), round(r.transform.location.y, 2),
                         round(r.speed, 3), r.decision.state.value, round(r.target_speed, 2),
                         r.p.tl_state.value if r.p.tl_state else None,
                         dbg.raw_state if dbg else None,
                         r.gt.tl_state.value if r.gt.tl_state else None,
                         None if r.gt.stop_distance is None else round(r.gt.stop_distance, 2),
                         r.gt_tl_id, round(dbg.infer_ms, 1) if dbg else None])
        wall = time.perf_counter() - wall

    with open(csv_path.with_suffix(".csv.partial"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        w.writerows(rows)
    csv_path.with_suffix(".csv.partial").replace(csv_path)
    log = pd.read_csv(csv_path)
    summary = nan_to_none(summarize_run(log, dt, drv.stop_margin))
    summary.update({"spawn": spawn, "collisions": len(collisions), "collision_with": sorted(set(collisions)),
                    "wall_s": round(wall, 1),
                    "infer_ms_median": None if log.infer_ms.isna().all() else round(float(log.infer_ms.median()), 1)})
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval" / "compare_v1.yaml"))
    ap.add_argument("--only", default=None, help="조건 id 목록 (쉼표), 예: clear_noon,rain_night")
    ap.add_argument("--limit", type=int, default=None, help="이번에 돌릴 최대 실행 수")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(CompareConfig, args.config)
    out = PROJECT_ROOT / cfg.output_root / cfg.name / "runs"
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in plan(cfg, set(args.only.split(",")) if args.only else None)
            if not (out / f"{r[0]}_s{r[2]}_{r[3]}.json").exists()]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"[plan] {cfg.name}: 이번에 {len(todo)}회 주행 (각 {cfg.seconds:.0f}초, {cfg.map}) → {out}")
    for cid, _, s, p in todo:
        print(f"       - {cid}_s{s}_{p}")
    if args.dry_run or not todo:
        return

    sim_cfg = load_sim_config()
    sim_cfg.map = cfg.map
    drv = load_driving_config()
    presets = load_collection_config(PROJECT_ROOT / cfg.conditions_source)
    client, world = connect(sim_cfg)
    with synchronous_mode(world, sim_cfg.fixed_delta_seconds):
        current = None
        for i, (cid, (w, tod), seed, perc) in enumerate(todo, 1):
            if cid != current:
                apply_weather(world, {**presets.weathers[w], **presets.times[tod]})
                current = cid
            stem = f"{cid}_s{seed}_{perc}"
            s = run_one(client, world, sim_cfg, drv, cfg, seed, perc, out / f"{stem}.csv")
            s.update({"condition": cid, "seed": seed, "perception": perc})
            (out / f"{stem}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {stem:28} {s['wall_s']:5.0f}s  위반 {s['red_violations']}  "
                  f"정지 {s['proper_stops']}/{s['stops']}  오정지 {s['false_stops']}  "
                  f"출발지연 {s['start_delay_mean']}  일치 {s['decision_agree']}  충돌 {s['collisions']}")


if __name__ == "__main__":
    main()
