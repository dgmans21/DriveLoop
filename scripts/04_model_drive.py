"""2-4: 내 모델(YOLO)로 인지해서 운전 — 1-3과 판단·제어는 같고 인지만 교체.

    python scripts/04_model_drive.py --perception model --record --seconds 120
    python scripts/04_model_drive.py --perception gt --record --seconds 120     # 비교용 (정답값 인지)
    python scripts/04_model_drive.py --perception model --weights runs/detect/v2_base/weights/best.pt

주의: 모델은 Epic 화질 이미지로 학습했다 → 서버를 Epic으로 띄울 것
      (start_carla.ps1 -Quality Epic -OffScreen). Low 화질이면 처음 보는 이미지라 성능이 크게 떨어진다.

화면/녹화: 전방 카메라(모델 입력) + 검출 박스 + 지도로 예상한 내 신호 위치(흰 테두리) + 고른 검출(MY LIGHT)
          + 오른쪽 아래 3인칭 화면. 녹화는 매 tick(20 fps) = 실제 시간 속도.
로그(CSV): 매 tick 모델 판단과 실제 신호 상태(gt_state)를 나란히 기록 — gt_state는 판단에 쓰지 않고 비교 분석용.
"""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import carla

from driveloop.config import CameraConfig, PROJECT_ROOT, load_driving_config, load_sim_config
from driveloop.control.lateral import pick_lookahead_point, pure_pursuit_steer, rear_axle
from driveloop.control.pid import LongitudinalController
from driveloop.data.collection_config import CaptureCamera
from driveloop.perception.ground_truth import GroundTruthPerception
from driveloop.planning.behavior import TrafficLightBehavior
from driveloop.planning.route import RoutePlanner
from driveloop.planning.speed_limit import curve_speed_limit
from driveloop.sim.client import ActorPool, connect, spawn_ego, speed_mps, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame, image_to_rgb
from driveloop.viz.hud import Display
from driveloop.viz.overlay import STATE_RGB, Mp4Recorder, draw_detections, draw_panel, paste_inset

WHITE, GRAY, CYAN = (255, 255, 255), (180, 180, 180), (90, 210, 255)
BEHAVIOR_RGB = {"CRUISE": (80, 230, 110), "STOPPING": (255, 210, 60), "STOPPED": (255, 70, 70),
                "PROCEED_YELLOW": (90, 210, 255)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", choices=["model", "gt"], default="model")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "runs" / "detect" / "v2_base" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.10,
                        help="검출 신뢰도 임계값. Town05 test 스윕: 0.25→0.10에서 red recall 0.76→0.82, "
                             "늘어난 오검출은 지도 예상 위치 근처만 쓰므로 영향 작음")
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=None, help="시뮬레이션 시간 (없으면 ESC까지)")
    parser.add_argument("--record", action="store_true", help="outputs/videos/ 에 MP4 녹화")
    parser.add_argument("--no-display", action="store_true", help="창 없이 실행 (녹화·로그만)")
    args = parser.parse_args()

    cfg = load_sim_config()
    drv = load_driving_config()
    if args.map:
        cfg.map = args.map
    if args.seed is not None:
        cfg.seed = args.seed
    dt = cfg.fixed_delta_seconds
    front = CaptureCamera()  # 학습 데이터와 같은 위치·해상도·시야각

    client, world = connect(cfg)
    rng = random.Random(cfg.seed)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    tag = f"{args.perception}_{cfg.map}_s{cfg.seed}_{stamp}"
    out_dir = PROJECT_ROOT / "outputs"
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"drive_{tag}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["t", "x", "y", "speed", "perception", "tl_state", "gt_state", "raw_state", "assoc_conf",
                  "stop_dist", "state", "target_speed", "infer_ms", "n_det"])
    display = None if args.no_display else Display(front.width, front.height, f"DriveLoop 2-4: {args.perception}")
    recorder = Mp4Recorder(out_dir / "videos" / f"drive_{tag}.mp4", 1 / dt, (front.width, front.height)) \
        if args.record else None
    print(f"[log] {log_path}" + (f"\n[rec] {recorder.path}" if recorder else ""))

    try:
        with synchronous_mode(world, dt), ActorPool() as pool:
            ego, idx = spawn_ego(world, cfg, rng)
            pool.add(ego)
            front_cam, front_q = attach_rgb_camera(
                world, ego, CameraConfig(front.width, front.height, front.fov),
                carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(front_cam)
            chase_cam, chase_q = attach_rgb_camera(world, ego, cfg.camera)
            pool.add(chase_cam)
            world.tick()

            max_steer = math.radians(ego.get_physics_control().wheels[0].max_steer_angle)
            route = RoutePlanner(world.get_map(), ego.get_location(), drv.route_spacing, drv.route_horizon, rng)
            gt = GroundTruthPerception(world, ego, route, drv.tl_lookahead)   # model 모드에선 비교 기록용
            if args.perception == "model":
                from driveloop.perception.model import ModelPerception
                perception = ModelPerception(world, ego, route, front_cam, front.width, front.height, front.fov,
                                             args.weights, drv.tl_lookahead, conf=args.conf)
            else:
                perception = gt
            behavior = TrafficLightBehavior(drv)
            longitudinal = LongitudinalController(drv)
            print(f"[start] perception={args.perception} map={cfg.map} spawn={idx}"
                  + (f" weights={args.weights}" if args.perception == "model" else ""))

            prev_state = behavior.state
            t0 = world.get_snapshot().timestamp.elapsed_seconds
            while True:
                if display is not None and display.poll_quit():
                    break
                frame = world.tick()
                front_img = get_frame(front_q, frame)
                chase_img = get_frame(chase_q, frame)
                sim_t = world.get_snapshot().timestamp.elapsed_seconds - t0
                if args.seconds is not None and sim_t >= args.seconds:
                    break

                tf = ego.get_transform()
                speed = speed_mps(ego)
                route.update(tf.location)

                # 인지 → 판단 → 제어 (판단·제어는 1-3과 동일)
                p = perception.perceive(front_img)
                gt_p = gt.perceive() if args.perception == "model" else p
                decision = behavior.step(p, speed)
                points = route.points_xy()
                curve_limit = curve_speed_limit(points, drv.curve_lat_accel, drv.comfort_decel, behavior.cruise_speed)
                target_speed = min(decision.target_speed, curve_limit)
                throttle, brake = longitudinal.step(target_speed, speed, dt)
                if len(points) >= 2:
                    ego_xy = (tf.location.x, tf.location.y)
                    target = pick_lookahead_point(points, rear_axle(ego_xy, tf.rotation.yaw, drv.wheelbase),
                                                  drv.lookahead_min + drv.lookahead_gain * speed)
                    steer = pure_pursuit_steer(ego_xy, tf.rotation.yaw, target, drv.wheelbase, max_steer)
                else:
                    steer, throttle, brake = 0.0, 0.0, 1.0
                ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

                dbg = getattr(perception, "debug", None)
                tl_val = p.tl_state.value if p.tl_state else ""
                gt_val = gt_p.tl_state.value if gt_p.tl_state else ""
                log.writerow([f"{sim_t:.2f}", f"{tf.location.x:.2f}", f"{tf.location.y:.2f}", f"{speed:.2f}",
                              args.perception, tl_val, gt_val,
                              (dbg.raw_state or "") if dbg else "",
                              f"{dbg.association.conf:.2f}" if dbg and dbg.association else "",
                              "" if p.stop_distance is None else f"{p.stop_distance:.2f}",
                              decision.state.value, f"{target_speed:.2f}",
                              f"{dbg.infer_ms:.1f}" if dbg else "", len(dbg.detections) if dbg else ""])
                if decision.state is not prev_state:
                    print(f"[t={sim_t:7.2f}s] {prev_state.value:>14} -> {decision.state.value:<14} "
                          f"v={speed * 3.6:5.1f}km/h  light={tl_val or '-'} (gt {gt_val or '-'})  {decision.reason}")
                    prev_state = decision.state

                if display is None and recorder is None:
                    continue
                view = image_to_rgb(front_img)
                if dbg is not None:
                    view = draw_detections(view, dbg.detections, dbg.expected,
                                           dbg.association.det if dbg.association else None)
                light = "none" if p.tl_state is None else f"{tl_val} {p.stop_distance:4.1f} m"
                lines = [
                    (f"Perception: {'MY MODEL (YOLO)' if args.perception == 'model' else 'ground truth'}", CYAN),
                    (f"Light  : {light}", STATE_RGB.get(tl_val, GRAY)),
                ]
                if dbg is not None:
                    lines.append((f"  raw {dbg.raw_state or '-':7} | vote {dbg.voted_state:7} | gt {gt_val or '-'}",
                                  GRAY))
                lines += [
                    (f"State  : {decision.state.value}", BEHAVIOR_RGB.get(decision.state.value, WHITE)),
                    (f"Speed  : {speed * 3.6:5.1f} / {target_speed * 3.6:5.1f} km/h", WHITE),
                ]
                if dbg is not None:
                    lines.append((f"Infer  : {dbg.infer_ms:5.1f} ms  ({len(dbg.detections)} det)", GRAY))
                view = draw_panel(view, lines)
                view = paste_inset(view, image_to_rgb(chase_img), 0.28)
                if recorder:
                    recorder.write(view)
                if display is not None:
                    display.draw(view)
    finally:
        log_file.close()
        if recorder:
            recorder.close()
            print(f"[rec] {recorder.frames} frames → {recorder.path}")
        if display is not None:
            display.close()


if __name__ == "__main__":
    main()
