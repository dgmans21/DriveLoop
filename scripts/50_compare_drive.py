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
from contextlib import contextmanager
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
from driveloop.sim.traffic import spawn_npc_vehicles
from driveloop.sim.weather import apply_weather

# raw_state가 비었을 때 원인 구분용 (compare_v1에서 로그만으로 못 가렸던 것):
#   n_tl_det = 0          → 모델이 신호등을 아예 못 찾음
#   n_tl_det > 0, n_exp>0 → 찾았는데 지도 예상 위치와 짝이 안 맞음
#   n_exp = 0             → 예상 위치가 화면 밖 / 신호가 카메라를 향하지 않음
# 앞차(B단계): lead_dist / lead_speed = 정답값 앞차 (지표용), p_lead_dist / p_lead_speed = 판단에 쓴 인지의 앞차
LOG_COLS = ["t", "x", "y", "speed", "state", "target_speed", "tl_state", "raw_state", "gt_state", "gt_dist",
            "gt_tl", "infer_ms", "n_det", "n_tl_det", "n_exp", "assoc_conf",
            "lead_dist", "lead_speed", "lead_id", "p_lead_dist", "p_lead_speed", "acc_target", "acc_emergency"]


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
    traffic: int = 0                       # NPC 차량 수 (0 = 빈 도로, 신호등 비교 v1·v2)
    tm_port: int = 8000


@contextmanager
def _tm_sync_off(client, cfg: CompareConfig):
    """끝나면 Traffic Manager 동기 모드 해제 (남아 있으면 다음 스크립트의 tick이 멈출 수 있다)."""
    try:
        yield
    finally:
        if cfg.traffic > 0:
            client.get_trafficmanager(cfg.tm_port).set_synchronous_mode(False)


def _r(v: float | None, nd: int = 2) -> float | None:
    return None if v is None else round(v, nd)


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
    collisions: list[dict] = []
    npc_ids: list[int] = []
    with ActorPool() as pool:
        ego, spawn = spawn_ego(world, sim_cfg, rng)
        pool.add(ego)
        if cfg.traffic > 0:
            # 시드마다 같은 NPC 배치·같은 TM 난수 → 정답값/모델 실행이 같은 교통에서 출발
            # (단, NPC는 내 차의 움직임에 반응하므로 시간이 지나면 교통 흐름이 조금씩 갈라질 수 있다)
            tm = client.get_trafficmanager(cfg.tm_port)
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(seed)
            npc_ids = spawn_npc_vehicles(client, world, tm, cfg.traffic, random.Random(seed + 1000))
        if world.get_weather().sun_altitude_angle < 0:
            ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position
                                                        | carla.VehicleLightState.LowBeam))
        col = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(),
                                attach_to=ego)
        pool.add(col)
        def on_collision(e):
            # 상대 위치를 내 차 기준으로: 앞(x>0)이면 내가 들이받음, 뒤면 들이받힘 (내 판단과 무관할 수 있음)
            tf, o = ego.get_transform(), e.other_actor.get_location()
            fwd = tf.get_forward_vector()
            rel = (o.x - tf.location.x) * fwd.x + (o.y - tf.location.y) * fwd.y
            collisions.append({"id": e.other_actor.id, "type": e.other_actor.type_id, "front": rel > 0,
                               "frame": e.frame})
        col.listen(on_collision)
        cam_q = None
        if perception == "model":
            cam, cam_q = attach_rgb_camera(world, ego, CameraConfig(front.width, front.height, front.fov),
                                           carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(cam)

            def make(route, _gt):
                from driveloop.perception.model import ModelPerception
                return ModelPerception(world, ego, route, cam, front.width, front.height, front.fov,
                                       str(PROJECT_ROOT / cfg.weights), drv.tl_lookahead, conf=cfg.conf,
                                       lead_lookahead=drv.lead_lookahead, lane_half_width=drv.lane_half_width,
                                       cam_height=drv.mono_cam_height, dt=dt)
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
                         r.gt_tl_id, round(dbg.infer_ms, 1) if dbg else None,
                         len(dbg.detections) if dbg else None,
                         sum(d.cls.startswith("tl_") for d in dbg.detections) if dbg else None,
                         len(dbg.expected) if dbg else None,
                         round(dbg.association.conf, 2) if dbg and dbg.association else None,
                         _r(r.gt.lead_distance), _r(r.gt.lead_speed), r.gt.lead_id,
                         _r(r.p.lead_distance), _r(r.p.lead_speed),
                         _r(r.acc.target_speed), int(r.acc.emergency)])
        wall = time.perf_counter() - wall
        if npc_ids:
            client.apply_batch_sync([carla.command.DestroyActor(i) for i in npc_ids], True)

    with open(csv_path.with_suffix(".csv.partial"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        w.writerows(rows)
    csv_path.with_suffix(".csv.partial").replace(csv_path)
    log = pd.read_csv(csv_path)
    summary = nan_to_none(summarize_run(log, dt, drv.stop_margin, drv.comfort_decel, drv.cruise_speed_kmh / 3.6))
    # 같은 상대와 맞닿아 있는 동안 매 프레임 이벤트가 나온다 → 상대 1대 = 충돌 1건 (처음 닿은 순간 기준)
    first = {}
    for c in collisions:
        first.setdefault(c["id"], c)
    hits = list(first.values())
    summary.update({"spawn": spawn, "npc": len(npc_ids), "collisions": len(hits),
                    "collisions_front": sum(c["front"] for c in hits),
                    "collisions_rear": sum(not c["front"] for c in hits),
                    "collision_with": sorted({c["type"] for c in hits}),
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
    with synchronous_mode(world, sim_cfg.fixed_delta_seconds), _tm_sync_off(client, cfg):
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
                  f"주의감속 {s['caution_s']}s  "
                  f"정지 {s['proper_stops']}/{s['stops']}  오정지 {s['false_stops']}  "
                  f"출발지연 {s['start_delay_mean']}  일치 {s['decision_agree']}  "
                  f"충돌 {s['collisions']}(앞{s['collisions_front']}/뒤{s['collisions_rear']})  "
                  f"차간최소 {s['min_time_gap']}s  TTC최소 {s['min_ttc']}s  급제동 {s['hard_brakes']}")


if __name__ == "__main__":
    main()
