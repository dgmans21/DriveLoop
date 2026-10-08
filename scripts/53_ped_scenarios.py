"""C-2: 보행자 시나리오 — 양보 판단(planning/pedestrian.py)을 실제 시뮬레이터에서 확인. 실행마다 tick 로그 CSV + 요약 JSON.

    python scripts/53_ped_scenarios.py --dry-run
    python scripts/53_ped_scenarios.py --scenarios cross --seeds 1          # 파일럿
    python scripts/53_ped_scenarios.py                                      # 전체 (끝난 실행은 건너뜀)
    python scripts/53_ped_scenarios.py --record                             # 오버레이 영상 → <name>/clips/

시나리오 정의는 eval/scenarios.py (PED_SCENARIOS). 52_acc_scenarios.py와 같은 방식:
  - 같은 출발 지점·경로 (sim/scenario_start.py), 신호등 모두 초록 고정 (보행자 판단만 본다)
  - 보행자는 AI 컨트롤러가 아니라 이 스크립트가 시간표대로 직접 움직인다 (WalkerControl: 방향 + 속도)
  - 보행자 출발은 시간이 아니라 '내 앞 범퍼까지 남은 거리' 기준 → 실행마다 같은 거리에서 뛰어든다
인지: gt(정답값) / model(카메라 + v4 모델, C-7). 판단·제어는 같은 DrivingAgent.
  model 로그에는 모델이 본 보행자 중 실제 보행자와 가장 가까운 것의 경로 좌표(p_ped_*)를 정답값(ped_*)과 나란히 남긴다.
결과: <output_root>/<name>/runs/<날씨>_<시나리오>_s<시드>_<인지>.csv / .json
"""
import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import carla
import pandas as pd

from driveloop.agent import DrivingAgent
from driveloop.config import PROJECT_ROOT, CameraConfig, load_config, load_driving_config, load_sim_config
from driveloop.data.collection_config import CaptureCamera, load_collection_config
from driveloop.eval.driving import nan_to_none, ped_metrics, ped_perception_metrics
from driveloop.eval.scenarios import PED_SCENARIOS, PedScenario
from driveloop.perception.lead import Obstacle, _cumulative, project_on_path
from driveloop.planning.pedestrian import path_coords
from driveloop.sim.client import ActorPool, connect, synchronous_mode
from driveloop.sim.scenario_start import choose_start, sidewalk_offset
from driveloop.sim.sensors import attach_rgb_camera, get_frame
from driveloop.sim.weather import apply_weather

LOG_COLS = ["t", "x", "y", "speed", "state", "target_speed",
            "ped_x", "ped_y", "ped_speed", "ped_u", "ped_moving",
            "ped_gap", "ped_lat", "ped_toward",               # 정답값 경로 좌표 (판단과 같은 계산)
            "yield_id", "yield_reason", "yield_gap", "ped_acc", "ped_emergency",
            # C-7 모델 인지: 확정된 보행자 수, 실제 보행자와 짝지은 추정(2m 안) 경로 좌표, 이번 프레임 검출 수
            "n_p_peds", "p_ped_gap", "p_ped_lat", "p_ped_toward", "n_ped_det", "infer_ms",
            # 추적 전 이번 프레임 측정(발 점)의 경로 좌표 — 추적기 지연과 측정 지연을 구분하려고 (C-7 v2)
            "p_ped_raw_gap", "p_ped_raw_lat",
            "ped_caution"]                                    # 주의 서행 상한 (C-7 v3)
PAIR_RADIUS = 2.0             # m, 모델이 본 보행자를 실제 보행자와 짝짓는 거리 (로그·지표용, 판단에는 안 씀)
ROUTE_SEED = 0                # 경로 분기 시드 고정 → 시드는 '조건에 맞는 출발 지점 중 몇 번째'만 뜻한다


@dataclass
class PedScenarioConfig:
    name: str
    map: str
    conditions: list[list[str]]
    scenarios: list[str]
    seeds: list[int]
    seconds: float = 30.0
    perceptions: list[str] = field(default_factory=lambda: ["gt"])
    conditions_source: str = "configs/collection/v3.yaml"
    walker_blueprint: str = "walker.pedestrian.0001"
    weights: str = "runs/detect/v4_base/weights/best.pt"
    conf: float = 0.10
    straight_length: float = 100.0     # m, 보행자는 60m 안 + 지표는 지나간 뒤 3초까지 → 140m(앞차)보다 짧아도 됨
    sidewalk_margin: float = 10.0      # m, 보행자 출발점 앞뒤 이만큼 바로 옆에 보도가 있어야 함 (20m면 Town05 후보 0~3곳)
    max_heading_change: float = 15.0
    output_root: str = "outputs/compare"


def _r(v, nd=2):
    return None if v is None else round(v, nd)


class PedDriver:
    """보행자를 시나리오 시간표대로: 기준 waypoint의 오른쪽(연석) 벡터 u와 진행 방향으로 WalkerControl."""

    def __init__(self, walker, base_wp, sc: PedScenario, front_offset: float, u_start: float):
        self.w, self.sc, self.front, self.u_start = walker, sc, front_offset, u_start
        self.origin = base_wp.transform.location
        self.right = base_wp.transform.get_right_vector()
        self.fwd = base_wp.transform.get_forward_vector()
        self.t_start: float | None = None

    def u(self) -> float:
        loc = self.w.get_location()
        return (loc.x - self.origin.x) * self.right.x + (loc.y - self.origin.y) * self.right.y

    def step(self, t: float, gap: float | None) -> None:
        if self.t_start is None and (self.sc.trigger_gap is None or (gap is not None and gap <= self.sc.trigger_gap)):
            self.t_start = t
        since = None if self.t_start is None else t - self.t_start
        cmd = self.sc.command(since, self.u(), self.u_start)
        dx = self.right.x * cmd.u_speed + self.fwd.x * cmd.along_speed
        dy = self.right.y * cmd.u_speed + self.fwd.y * cmd.along_speed
        speed = math.hypot(dx, dy)
        if speed < 1e-3:
            self.w.apply_control(carla.WalkerControl(speed=0.0))
        else:
            self.w.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx / speed, dy / speed, 0.0), speed=speed))


def _clip_frame(front_img, chase_img, r, row, dbg=None):
    from driveloop.sim.sensors import image_to_rgb
    from driveloop.viz.overlay import draw_detections, draw_panel, paste_inset

    W, G, Y, R, C = (255, 255, 255), (180, 180, 180), (255, 210, 60), (255, 90, 90), (90, 210, 255)
    gap, lat, lim = row["ped_gap"], row["ped_lat"], row["ped_acc"]
    ped = "-" if gap is None else f"{gap:5.1f} m ahead, {abs(lat):4.1f} m from lane center"
    limit = "-" if lim is None else f"{lim * 3.6:5.1f} km/h"
    view = image_to_rgb(front_img)
    if dbg is not None:
        view = draw_detections(view, dbg.ped_dets)
        pg, pl = row.get("p_ped_gap"), row.get("p_ped_lat")
        seen = "-" if pg is None else f"{pg:5.1f} m ahead, {abs(pl):4.1f} m from lane center"
        head = [("Perception: MY MODEL (camera only)", C), (f"Ped model : {seen}", Y), (f"Ped truth : {ped}", G)]
    else:
        head = [("Perception: GROUND TRUTH", G), (f"Pedestrian: {ped}", W)]
    lines = head + [
             (f"Yield     : {r.ped.reason if r.ped else ('caution ' + format(r.ped_caution * 3.6, '.0f') + ' km/h' if r.ped_caution else '-')}",
              Y if r.ped or r.ped_caution else G),
             (f"Ped limit : {limit}" + ("  EMERGENCY" if row["ped_emergency"] else ""),
              R if row["ped_emergency"] else W),
             (f"Speed     : {r.speed * 3.6:5.1f} / {r.target_speed * 3.6:5.1f} km/h", W)]
    view = draw_panel(view, lines, width=520)
    return paste_inset(view, image_to_rgb(chase_img), 0.28)


def run_one(world, sim_cfg, drv, cfg: PedScenarioConfig, sc: PedScenario, seed: int, perception: str,
            csv_path: Path, record_dir: Path | None = None) -> dict:
    dt = sim_cfg.fixed_delta_seconds
    wmap = world.get_map()
    idx, path = choose_start(wmap, wmap.get_spawn_points(), seed, drv.route_spacing, cfg.straight_length,
                             cfg.max_heading_change, False, sc.name,
                             need_sidewalk=(sc.ahead - cfg.sidewalk_margin, sc.ahead + cfg.sidewalk_margin),
                             distinct=True, route_seed=ROUTE_SEED)
    sp = wmap.get_spawn_points()[idx]
    base = path[min(len(path) - 1, round(sc.ahead / drv.route_spacing))]
    sidewalk_u = sidewalk_offset(base)
    u0 = sc.start_u(base.lane_width, sidewalk_u)
    start = base.transform.location + base.transform.get_right_vector() * u0 + carla.Location(z=1.0)

    lights = list(world.get_actors().filter("traffic.traffic_light"))
    collisions: list[dict] = []
    bl = world.get_blueprint_library()
    with ActorPool() as pool:
        ego = world.try_spawn_actor(bl.find(sim_cfg.vehicle_blueprint), sp)
        wbp = bl.find(cfg.walker_blueprint)
        if wbp.has_attribute("is_invincible"):
            wbp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(wbp, carla.Transform(start, base.transform.rotation))
        if ego is None or walker is None:
            raise RuntimeError(f"스폰 실패 (spawn {idx}, 보행자 {start})")
        pool.add(ego)
        pool.add(walker)
        if world.get_weather().sun_altitude_angle < 0:
            ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position
                                                        | carla.VehicleLightState.LowBeam))
        col = pool.add(world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=ego))
        col.listen(lambda e: collisions.append({"id": e.other_actor.id, "type": e.other_actor.type_id}))

        recorder, cam, cam_q, chase_q = None, None, None, None
        front = CaptureCamera()
        if record_dir is not None or perception == "model":
            cam, cam_q = attach_rgb_camera(world, ego, CameraConfig(front.width, front.height, front.fov),
                                           carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(cam)
        if record_dir is not None:
            from driveloop.viz.overlay import Mp4Recorder
            chase, chase_q = attach_rgb_camera(world, ego, sim_cfg.camera)
            pool.add(chase)
            record_dir.mkdir(parents=True, exist_ok=True)
            recorder = Mp4Recorder(record_dir / f"{csv_path.stem}_raw.mp4", 1 / dt, (front.width, front.height))

        for tl in lights:                                  # 보행자 판단만 본다: 신호는 모두 초록 고정
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        try:
            world.tick()
            def make_model(route, _gt):
                from driveloop.perception.model import ModelPerception
                return ModelPerception(world, ego, route, cam, front.width, front.height, front.fov,
                                       str(PROJECT_ROOT / cfg.weights), drv.tl_lookahead, conf=cfg.conf,
                                       lead_lookahead=drv.lead_lookahead, lane_half_width=drv.lane_half_width,
                                       cam_height=drv.mono_cam_height, dt=dt)
            agent = DrivingAgent(world, ego, drv, random.Random(ROUTE_SEED),     # choose_start와 같은 분기 선택
                                 make_model if perception == "model" else None)
            front_offset = ego.bounding_box.extent.x
            driver = PedDriver(walker, base, sc, front_offset, u0)
            rows = []
            t0 = world.get_snapshot().timestamp.elapsed_seconds
            wall = time.perf_counter()
            gap = None
            while True:
                frame = world.tick()
                img = get_frame(cam_q, frame) if cam_q is not None else None
                t = world.get_snapshot().timestamp.elapsed_seconds - t0
                if t >= sc.seconds:
                    break
                driver.step(t, gap)
                r = agent.step(img if perception == "model" else None, dt)
                dbg = getattr(agent.perception, "debug", None)
                pts = agent.route.points_xy()
                wl, wv = walker.get_location(), walker.get_velocity()
                row = {"ped_gap": None, "ped_lat": None, "ped_toward": None,
                       "p_ped_gap": None, "p_ped_lat": None, "p_ped_toward": None,
                       "p_ped_raw_gap": None, "p_ped_raw_lat": None}
                if len(pts) >= 2:
                    cum = _cumulative(pts)
                    s_ego, _, _ = project_on_path(pts, cum, (r.transform.location.x, r.transform.location.y))
                    c = path_coords(pts, cum, s_ego, front_offset,
                                    Obstacle(walker.id, wl.x, wl.y, wv.x, wv.y, walker.bounding_box.extent.x))
                    row.update({"ped_gap": _r(c.gap), "ped_lat": _r(c.lateral), "ped_toward": _r(c.toward)})
                    near = [o for o in r.p.pedestrians if math.hypot(o.x - wl.x, o.y - wl.y) <= PAIR_RADIUS]
                    if perception == "model" and near:
                        o = min(near, key=lambda o: math.hypot(o.x - wl.x, o.y - wl.y))
                        pc = path_coords(pts, cum, s_ego, front_offset, o)
                        row.update({"p_ped_gap": _r(pc.gap), "p_ped_lat": _r(pc.lateral),
                                    "p_ped_toward": _r(pc.toward)})
                    raw = [q for q in (dbg.ped_points if dbg else []) if math.hypot(q[0] - wl.x, q[1] - wl.y) <= PAIR_RADIUS]
                    if raw:
                        q = min(raw, key=lambda q: math.hypot(q[0] - wl.x, q[1] - wl.y))
                        rc = path_coords(pts, cum, s_ego, front_offset, Obstacle(-1, q[0], q[1], half_length=0.3))
                        row.update({"p_ped_raw_gap": _r(rc.gap), "p_ped_raw_lat": _r(rc.lateral)})
                gap = row["ped_gap"]
                row.update({"ped_acc": _r(r.ped_acc.target_speed) if r.ped_acc else None,
                            "ped_emergency": int(bool(r.ped_acc and r.ped_acc.emergency))})
                rows.append([round(t, 2), round(r.transform.location.x, 2), round(r.transform.location.y, 2),
                             round(r.speed, 3), r.decision.state.value, round(r.target_speed, 2),
                             round(wl.x, 2), round(wl.y, 2), _r(math.hypot(wv.x, wv.y)), _r(driver.u()),
                             int(driver.t_start is not None),
                             row["ped_gap"], row["ped_lat"], row["ped_toward"],
                             r.ped.id if r.ped else None, r.ped.reason if r.ped else None,
                             _r(r.ped.distance) if r.ped else None, row["ped_acc"], row["ped_emergency"],
                             len(r.p.pedestrians) if perception == "model" else None,
                             row["p_ped_gap"], row["p_ped_lat"], row["p_ped_toward"],
                             len(dbg.ped_dets) if dbg else None, _r(dbg.infer_ms, 1) if dbg else None,
                             row["p_ped_raw_gap"], row["p_ped_raw_lat"], _r(r.ped_caution)])
                if recorder is not None:
                    recorder.write(_clip_frame(img if img is not None else get_frame(cam_q, frame),
                                               get_frame(chase_q, frame), r, row, dbg))
            wall = time.perf_counter() - wall
        finally:
            for tl in lights:
                tl.freeze(False)
            world.reset_all_traffic_lights()
            if recorder is not None:
                from driveloop.viz.overlay import to_web_mp4
                recorder.close()
                if to_web_mp4(recorder.path, record_dir / f"{csv_path.stem}.mp4"):
                    Path(recorder.path).unlink()

    with open(csv_path.with_suffix(".csv.partial"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLS)
        w.writerows(rows)
    csv_path.with_suffix(".csv.partial").replace(csv_path)
    log = pd.read_csv(csv_path)
    s = nan_to_none({**ped_metrics(log, dt, cruise_speed=drv.cruise_speed_kmh / 3.6),
                     **(ped_perception_metrics(log, dt) if perception == "model" else {})})
    hits = {c["id"]: c for c in collisions}
    s.update({"spawn": idx, "collisions": len(hits),
              "collision_with": sorted({c["type"] for c in hits.values()}),
              "ped_hit": any(c["type"].startswith("walker") for c in hits.values()),
              "ped_started": bool(log.ped_moving.any()), "yield_expected": sc.yield_expected,
              "lane_width": round(base.lane_width, 2), "sidewalk_u": round(sidewalk_u, 2), "start_u": round(u0, 2),
              "wall_s": round(wall, 1)})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval" / "ped_scen_v1.yaml"))
    ap.add_argument("--only", default=None, help="날씨 조건 id (쉼표), 예: clear_noon")
    ap.add_argument("--scenarios", default=None, help="시나리오 (쉼표), 예: cross,dartout")
    ap.add_argument("--seeds", default=None, help="시드 (쉼표), 예: 1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--record", action="store_true", help="오버레이 영상 → <output_root>/<name>/clips/")
    args = ap.parse_args()

    cfg = load_config(PedScenarioConfig, args.config)
    only = set(args.only.split(",")) if args.only else None
    scen = args.scenarios.split(",") if args.scenarios else cfg.scenarios
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg.seeds
    out = PROJECT_ROOT / cfg.output_root / cfg.name / "runs"
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for w, tod in cfg.conditions:
        cid = f"{w}_{tod}"
        if only and cid not in only:
            continue
        for sname in scen:
            for seed in seeds:
                for p in cfg.perceptions:
                    stem = f"{cid}_{sname}_s{seed}_{p}"
                    if not (out / f"{stem}.json").exists():
                        todo.append((cid, w, tod, sname, seed, p, stem))
    print(f"[plan] {cfg.name}: 이번에 {len(todo)}회 (시나리오 각 {cfg.seconds:.0f}초, {cfg.map}) → {out}")
    for t in todo:
        print(f"       - {t[-1]}")
    if args.dry_run or not todo:
        return

    sim_cfg = load_sim_config()
    sim_cfg.map = cfg.map
    drv = load_driving_config()
    presets = load_collection_config(PROJECT_ROOT / cfg.conditions_source)
    _, world = connect(sim_cfg)
    with synchronous_mode(world, sim_cfg.fixed_delta_seconds):
        current = None
        for i, (cid, w, tod, sname, seed, p, stem) in enumerate(todo, 1):
            if cid != current:
                apply_weather(world, {**presets.weathers[w], **presets.times[tod]})
                current = cid
            base = PED_SCENARIOS[sname]
            sc = PedScenario(base.name, base.ahead, base.anchor, base.offset, base.trigger_gap, base.yield_expected,
                             cfg.seconds)
            s = run_one(world, sim_cfg, drv, cfg, sc, seed, p, out / f"{stem}.csv",
                        out.parent / "clips" if args.record else None)
            s.update({"weather": cid, "scenario": sname, "seed": seed, "perception": p})
            (out / f"{stem}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {stem:30} {s['wall_s']:5.0f}s  충돌 {s['collisions']}{'(보행자)' if s['ped_hit'] else ''}  "
                  f"최소간격 {s['min_ped_gap']}m  양보 {s['yield_s']}s  최저속도 {s['min_speed_after_cruise']}  "
                  f"정지 {s['stop_s']}s  재출발 {s['restart_delay']}s  급제동 {s['hard_brakes']}  "
                  f"최대감속 {s['max_decel']}  출발 {s['ped_started']}"
                  + (f"  [모델] 놓침 {s['ped_miss_rate']}  첫 확정 {s['first_seen_gap']}m  거리오차 {s['ped_gap_err_med']}m  "
                     f"옆오차 {s['ped_lat_err_med']}m" if p == "model" else ""))


if __name__ == "__main__":
    main()
