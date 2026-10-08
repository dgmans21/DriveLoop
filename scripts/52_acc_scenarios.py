"""B-6: 앞차 시나리오 — 정답값 인지(gt) vs 내 모델 인지(model). 실행마다 tick 로그 CSV + 요약 JSON.

    python scripts/52_acc_scenarios.py --dry-run
    python scripts/52_acc_scenarios.py --only clear_noon --scenarios follow,brake     # 파일럿
    python scripts/52_acc_scenarios.py                                               # 전체 (끝난 실행은 건너뜀)
    python scripts/51_compare_report.py --config configs/eval/acc_scen_v1.yaml       # 분석 (같은 형식)

시나리오 정의는 eval/scenarios.py. 공정한 비교를 위해 실행마다
  - 같은 출발 지점: 시드 순서로 스폰 지점을 돌며 앞쪽 경로가 곧은 곳(방향 변화 < 15°)을 고른다
  - 같은 경로: 내 차(DrivingAgent)와 앞차 경로를 같은 시드의 RoutePlanner로 만든다 → 교차로 분기 선택이 같다
  - 같은 앞차 움직임: 앞차는 Traffic Manager가 아니라 이 스크립트가 시간표대로 직접 운전한다
  - 신호등은 모두 초록으로 고정 (앞차 판단만 비교), 실행이 끝나면 원래대로
결과: <output_root>/<name>/runs/<날씨>_<시나리오>_s<시드>_<인지>.csv / .json  (51_compare_report.py로 분석)
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
from driveloop.control.lateral import pick_lookahead_point, pure_pursuit_steer, rear_axle
from driveloop.control.pid import LongitudinalController
from driveloop.data.collection_config import CaptureCamera, load_collection_config
from driveloop.eval.driving import nan_to_none, summarize_run
from driveloop.eval.scenarios import SCENARIOS, Scenario, blend_paths, heading_change
from driveloop.planning.route import RoutePlanner
from driveloop.sim.client import ActorPool, connect, speed_mps, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame
from driveloop.sim.weather import apply_weather

LOG_COLS = ["t", "x", "y", "speed", "state", "target_speed", "tl_state", "raw_state", "gt_state", "gt_dist",
            "gt_tl", "infer_ms", "n_det", "n_tl_det", "n_exp", "assoc_conf",
            "lead_dist", "lead_speed", "lead_id", "p_lead_dist", "p_lead_speed", "acc_target", "acc_emergency",
            "lead_true_speed",
            # B-7b: 차체 기울기 = ego_pitch − road_pitch (IMU 필요 여부 판단), p_lead_raw = 추적 전 카메라 측정 간격
            "ego_pitch", "road_pitch", "p_lead_raw"]


@dataclass
class ScenarioConfig:
    name: str
    map: str
    conditions: list[list[str]]
    scenarios: list[str]
    seeds: list[int]
    seconds: float = 35.0
    perceptions: list[str] = field(default_factory=lambda: ["gt", "model"])
    conditions_source: str = "configs/collection/v3.yaml"
    weights: str = "runs/detect/v3_base/weights/best.pt"
    conf: float = 0.10
    straight_length: float = 140.0
    max_heading_change: float = 15.0
    output_root: str = "outputs/compare"


def _r(v, nd=2):
    return None if v is None else round(v, nd)


def _same_dir_neighbor(wp: carla.Waypoint) -> carla.Waypoint | None:
    """같은 방향 옆 차선 (끼어들기 출발 차선)."""
    for n in (wp.get_left_lane(), wp.get_right_lane()):
        if n is not None and n.lane_type == carla.LaneType.Driving and n.lane_id * wp.lane_id > 0:
            return n
    return None


def choose_start(world_map, points, seed: int, drv, cfg: ScenarioConfig, sc: Scenario):
    """시드 순서로 스폰 지점을 돌며 앞쪽이 곧고, (끼어들기면) 옆 차선이 있는 곳."""
    for idx in random.Random(seed).sample(range(len(points)), len(points)):
        sp = points[idx]
        path = RoutePlanner(world_map, sp.location, drv.route_spacing, cfg.straight_length + 60,
                            random.Random(seed)).waypoints
        if len(path) * drv.route_spacing < cfg.straight_length:
            continue
        xy = [(w.transform.location.x, w.transform.location.y) for w in path]
        if heading_change(xy, cfg.straight_length) > cfg.max_heading_change:
            continue
        if sc.adjacent:
            adj = [_same_dir_neighbor(w) for w in path[: int(cfg.straight_length / drv.route_spacing)]]
            if any(a is None for a in adj):
                continue
        return idx, path
    raise RuntimeError(f"{sc.name}: 조건에 맞는 출발 지점 없음")


class LeadDriver:
    """앞차를 시나리오 시간표대로 운전: 경로 추종(pure pursuit) + 속도 PID / 급정거."""

    def __init__(self, vehicle, ego_pts, adj_pts, drv, dt):
        self.v, self.ego_pts, self.adj_pts, self.drv, self.dt = vehicle, ego_pts, adj_pts, drv, dt
        self.lon = LongitudinalController(drv)
        self.max_steer = math.radians(vehicle.get_physics_control().wheels[0].max_steer_angle)
        self.i = 0

    def step(self, cmd) -> None:
        pts = blend_paths(self.adj_pts, self.ego_pts, cmd.lane_blend) if self.adj_pts else self.ego_pts
        tf = self.v.get_transform()
        xy = (tf.location.x, tf.location.y)
        while self.i + 1 < len(pts) and math.dist(pts[self.i + 1], xy) <= math.dist(pts[self.i], xy):
            self.i += 1                                   # 지나간 경로 점은 버린다 (뒤쪽 점을 목표로 잡지 않게)
        ahead = pts[self.i:]
        speed = speed_mps(self.v)
        steer = 0.0
        if len(ahead) >= 2:
            target = pick_lookahead_point(ahead, rear_axle(xy, tf.rotation.yaw, self.drv.wheelbase), 4.0 + 0.3 * speed)
            steer = pure_pursuit_steer(xy, tf.rotation.yaw, target, self.drv.wheelbase, self.max_steer)
        if cmd.full_brake:
            self.v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer, hand_brake=speed < 0.1))
        else:
            thr, brk = self.lon.step(cmd.target_speed, speed, self.dt)
            self.v.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))


def _clip_frame(front_img, chase_img, r, dbg):
    """녹화용 한 프레임: 검출 박스 + 고른 앞차(굵게, 모델 거리 vs 정답) + 판단 패널 + 3인칭 화면."""
    import cv2

    from driveloop.sim.sensors import image_to_rgb
    from driveloop.viz.overlay import draw_detections, draw_panel, paste_inset

    view = image_to_rgb(front_img)
    if dbg is not None:
        view = draw_detections(view, [d for d in dbg.detections if d.cls == "vehicle"])
        if dbg.lead_det is not None and r.p.lead_distance is not None:
            x1, y1, x2, y2 = (int(v) for v in dbg.lead_det.box)
            cv2.rectangle(view, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 210, 60), 3, cv2.LINE_AA)
            gt = f" (gt {r.gt.lead_distance:.1f})" if r.gt.lead_distance is not None else ""
            label = f"LEAD {r.p.lead_distance:.1f} m{gt}"
            cv2.putText(view, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 60), 2, cv2.LINE_AA)
    W, G, Y, C = (255, 255, 255), (180, 180, 180), (255, 210, 60), (90, 210, 255)

    def fmt(d, s):
        return "-" if d is None else f"{d:5.1f} m  {s * 3.6:5.1f} km/h"
    lines = [("Perception: MY MODEL (camera only)", C),
             (f"Lead model: {fmt(r.p.lead_distance, r.p.lead_speed or 0.0)}", Y),
             (f"Lead truth: {fmt(r.gt.lead_distance, r.gt.lead_speed or 0.0)}", G),
             (f"ACC limit : {'-' if r.acc.target_speed is None else f'{r.acc.target_speed * 3.6:5.1f} km/h'}"
              + ("  EMERGENCY" if r.acc.emergency else ""), (255, 90, 90) if r.acc.emergency else W),
             (f"Speed     : {r.speed * 3.6:5.1f} / {r.target_speed * 3.6:5.1f} km/h", W)]
    view = draw_panel(view, lines, width=470)
    return paste_inset(view, image_to_rgb(chase_img), 0.28)


def run_one(world, sim_cfg, drv, cfg: ScenarioConfig, sc: Scenario, seed: int, perception: str,
            csv_path: Path, record_dir: Path | None = None) -> dict:
    dt = sim_cfg.fixed_delta_seconds
    front = CaptureCamera()
    wmap = world.get_map()
    idx, path = choose_start(wmap, wmap.get_spawn_points(), seed, drv, cfg, sc)
    sp = wmap.get_spawn_points()[idx]
    k = min(len(path) - 1, round(sc.lead_offset / drv.route_spacing))
    lead_wp = _same_dir_neighbor(path[k]) if sc.adjacent else path[k]
    ego_pts = [(w.transform.location.x, w.transform.location.y) for w in path]
    adj_pts = None
    if sc.adjacent:
        adj = [_same_dir_neighbor(w) for w in path]
        adj_pts = [(a.transform.location.x, a.transform.location.y) if a else p for a, p in zip(adj, ego_pts)]

    lights = list(world.get_actors().filter("traffic.traffic_light"))
    collisions: list[dict] = []
    bl = world.get_blueprint_library()
    with ActorPool() as pool:
        ego = world.try_spawn_actor(bl.find(sim_cfg.vehicle_blueprint), sp)
        lead_tf = carla.Transform(lead_wp.transform.location + carla.Location(z=0.3), lead_wp.transform.rotation)
        lead = world.try_spawn_actor(bl.find("vehicle.audi.a2"), lead_tf)
        if ego is None or lead is None:
            raise RuntimeError(f"스폰 실패 (spawn {idx})")
        pool.add(ego)
        pool.add(lead)
        if world.get_weather().sun_altitude_angle < 0:
            lamps = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
            ego.set_light_state(lamps)
            lead.set_light_state(lamps)
        col = pool.add(world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=ego))

        def on_collision(e):
            tf, o = ego.get_transform(), e.other_actor.get_location()
            fwd = tf.get_forward_vector()
            rel = (o.x - tf.location.x) * fwd.x + (o.y - tf.location.y) * fwd.y
            collisions.append({"id": e.other_actor.id, "type": e.other_actor.type_id, "front": rel > 0})
        col.listen(on_collision)

        cam, cam_q = None, None
        if perception == "model":
            cam, cam_q = attach_rgb_camera(world, ego, CameraConfig(front.width, front.height, front.fov),
                                           carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(cam)

        recorder, chase_q = None, None
        if record_dir is not None and perception == "model":
            from driveloop.viz.overlay import Mp4Recorder
            chase, chase_q = attach_rgb_camera(world, ego, sim_cfg.camera)
            pool.add(chase)
            record_dir.mkdir(parents=True, exist_ok=True)
            recorder = Mp4Recorder(record_dir / f"{csv_path.stem}_raw.mp4", 1 / dt, (front.width, front.height))

        def make_model(route, _gt):
            from driveloop.perception.model import ModelPerception
            return ModelPerception(world, ego, route, cam, front.width, front.height, front.fov,
                                   str(PROJECT_ROOT / cfg.weights), drv.tl_lookahead, conf=cfg.conf,
                                   lead_lookahead=drv.lead_lookahead, lane_half_width=drv.lane_half_width,
                                   cam_height=drv.mono_cam_height, dt=dt)
        make = make_model if perception == "model" else None
        for tl in lights:                                  # 앞차 판단만 비교: 신호는 모두 초록 고정
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
        try:
            world.tick()
            agent = DrivingAgent(world, ego, drv, random.Random(seed), make)   # 앞차 경로와 같은 분기 선택
            driver = LeadDriver(lead, ego_pts, adj_pts, drv, dt)
            rows = []
            t0 = world.get_snapshot().timestamp.elapsed_seconds
            wall = time.perf_counter()
            while True:
                frame = world.tick()
                img = get_frame(cam_q, frame) if cam_q is not None else None
                t = world.get_snapshot().timestamp.elapsed_seconds - t0
                if t >= sc.seconds:
                    break
                driver.step(sc.command(t))
                r = agent.step(img, dt)
                dbg = getattr(agent.perception, "debug", None)
                rows.append([round(t, 2), round(r.transform.location.x, 2), round(r.transform.location.y, 2),
                             round(r.speed, 3), r.decision.state.value, round(r.target_speed, 2),
                             r.p.tl_state.value if r.p.tl_state else None, dbg.raw_state if dbg else None,
                             r.gt.tl_state.value if r.gt.tl_state else None, _r(r.gt.stop_distance), r.gt_tl_id,
                             _r(dbg.infer_ms, 1) if dbg else None,
                             len(dbg.detections) if dbg else None,
                             sum(d.cls.startswith("tl_") for d in dbg.detections) if dbg else None,
                             len(dbg.expected) if dbg else None,
                             _r(dbg.association.conf) if dbg and dbg.association else None,
                             _r(r.gt.lead_distance), _r(r.gt.lead_speed), r.gt.lead_id,
                             _r(r.p.lead_distance), _r(r.p.lead_speed), _r(r.acc.target_speed), int(r.acc.emergency),
                             _r(speed_mps(lead)),
                             _r(r.transform.rotation.pitch, 3),
                             _r(wmap.get_waypoint(r.transform.location).transform.rotation.pitch, 3),
                             _r(dbg.lead_measured) if dbg else None])
                if recorder is not None:
                    recorder.write(_clip_frame(img, get_frame(chase_q, frame), r, dbg))
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
    s = nan_to_none(summarize_run(log, dt, drv.stop_margin, drv.comfort_decel, drv.cruise_speed_kmh / 3.6))
    first = {}
    for c in collisions:
        first.setdefault(c["id"], c)
    hits = list(first.values())
    end_gap = log.lead_dist.dropna()
    s.update({"spawn": idx, "collisions": len(hits), "collisions_front": sum(c["front"] for c in hits),
              "collisions_rear": sum(not c["front"] for c in hits),
              "collision_with": sorted({c["type"] for c in hits}), "wall_s": round(wall, 1),
              "final_gap": _r(float(end_gap.iloc[-1])) if len(end_gap) else None,
              "emergency_ticks": int(log.acc_emergency.sum()),
              "infer_ms_median": None if log.infer_ms.isna().all() else round(float(log.infer_ms.median()), 1)})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval" / "acc_scen_v1.yaml"))
    ap.add_argument("--only", default=None, help="날씨 조건 id (쉼표), 예: clear_noon")
    ap.add_argument("--scenarios", default=None, help="시나리오 (쉼표), 예: follow,brake")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="모델 인지 실행을 오버레이 영상으로 녹화 → <output_root>/<name>/clips/ (40_build_site.py가 웹에 넣음)")
    args = ap.parse_args()

    cfg = load_config(ScenarioConfig, args.config)
    only = set(args.only.split(",")) if args.only else None
    scen = args.scenarios.split(",") if args.scenarios else cfg.scenarios
    out = PROJECT_ROOT / cfg.output_root / cfg.name / "runs"
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for w, tod in cfg.conditions:
        cid = f"{w}_{tod}"
        if only and cid not in only:
            continue
        for sname in scen:
            for seed in cfg.seeds:
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
            sc = SCENARIOS[sname]
            sc = Scenario(sc.name, sc.lead_offset, sc.adjacent, cfg.seconds)
            s = run_one(world, sim_cfg, drv, cfg, sc, seed, p, out / f"{stem}.csv",
                        out.parent / "clips" if args.record else None)
            s.update({"condition": f"{cid}_{sname}", "weather": cid, "scenario": sname, "seed": seed, "perception": p})
            (out / f"{stem}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {stem:32} {s['wall_s']:5.0f}s  충돌 {s['collisions']}(앞{s['collisions_front']})  "
                  f"최소간격 {s['min_lead_gap']}m  차간최소 {s['min_time_gap']}s  TTC최소 {s['min_ttc']}s  "
                  f"급제동 {s['hard_brakes']}  비상 {s['emergency_ticks']}  끝간격 {s['final_gap']}  "
                  f"거리오차 {s['lead_err_med']}  놓침 {s['lead_miss_rate']}  헛봄 {s['lead_phantom_rate']}")


if __name__ == "__main__":
    main()
