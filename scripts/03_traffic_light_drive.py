"""1-3: 자율 주행 + 신호등 대응 (정답값 인지).

    python scripts/03_traffic_light_drive.py
    python scripts/03_traffic_light_drive.py --map Town03 --seed 7
확인: 차가 차선을 따라 달리다 빨간불에서 정지선 앞에 서서히 서고, 초록불에 출발.
      HUD에 신호 상태/정지선 거리/판단 상태/속도가 표시되고, 상태 전이가 터미널에 기록된다.
"""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import carla

from driveloop.config import PROJECT_ROOT, load_driving_config, load_sim_config
from driveloop.control.lateral import pick_lookahead_point, pure_pursuit_steer
from driveloop.control.pid import LongitudinalController
from driveloop.perception.ground_truth import GroundTruthPerception
from driveloop.perception.types import TLState
from driveloop.planning.behavior import BehaviorState, TrafficLightBehavior
from driveloop.planning.route import RoutePlanner
from driveloop.planning.speed_limit import curve_speed_limit
from driveloop.sim.client import ActorPool, connect, spawn_ego, speed_mps, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame, image_to_rgb
from driveloop.viz.hud import CYAN, GRAY, GREEN, RED, WHITE, YELLOW, Display

TL_COLOR = {TLState.RED: RED, TLState.YELLOW: YELLOW, TLState.GREEN: GREEN, TLState.UNKNOWN: GRAY}
STATE_COLOR = {
    BehaviorState.CRUISE: GREEN,
    BehaviorState.STOPPING: YELLOW,
    BehaviorState.STOPPED: RED,
    BehaviorState.PROCEED_YELLOW: CYAN,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", default=None, help="프레임별 CSV 경로 (기본: outputs/drive_<시각>.csv)")
    args = parser.parse_args()

    cfg = load_sim_config()
    drv = load_driving_config()
    if args.map:
        cfg.map = args.map
    if args.seed is not None:
        cfg.seed = args.seed
    dt = cfg.fixed_delta_seconds

    client, world = connect(cfg)
    rng = random.Random(cfg.seed)
    display = Display(cfg.camera.width, cfg.camera.height, "DriveLoop 1-3: traffic lights (GT)")
    log_path = Path(args.log) if args.log else PROJECT_ROOT / "outputs" / f"drive_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["t", "x", "y", "yaw", "speed", "tl_state", "stop_dist", "state", "target_speed",
                  "curve_limit", "throttle", "brake", "steer", "tgt_x", "tgt_y", "road", "lane", "route_len"])
    print(f"[log] {log_path}")

    try:
        with synchronous_mode(world, dt), ActorPool() as pool:
            ego, idx = spawn_ego(world, cfg, rng)
            pool.add(ego)
            camera, cam_q = attach_rgb_camera(world, ego, cfg.camera)
            pool.add(camera)
            world.tick()  # 스폰 반영

            max_steer = math.radians(ego.get_physics_control().wheels[0].max_steer_angle)
            route = RoutePlanner(world.get_map(), ego.get_location(), drv.route_spacing,
                                 drv.route_horizon, rng)
            perception = GroundTruthPerception(world, ego, route, drv.tl_lookahead)
            behavior = TrafficLightBehavior(drv)
            longitudinal = LongitudinalController(drv)
            print(f"[start] map={cfg.map} spawn={idx} traffic_lights={perception.num_traffic_lights}")

            prev_state = behavior.state
            t0 = world.get_snapshot().timestamp.elapsed_seconds
            while not display.poll_quit():
                frame = world.tick()
                image = get_frame(cam_q, frame)
                sim_t = world.get_snapshot().timestamp.elapsed_seconds - t0

                tf = ego.get_transform()
                speed = speed_mps(ego)
                route.update(tf.location)

                # 인지 → 판단 → 제어
                p = perception.perceive(image)
                decision = behavior.step(p, speed)
                points = route.points_xy()
                curve_limit = curve_speed_limit(points, drv.curve_lat_accel, drv.comfort_decel,
                                                behavior.cruise_speed)
                target_speed = min(decision.target_speed, curve_limit)
                throttle, brake = longitudinal.step(target_speed, speed, dt)

                target = (math.nan, math.nan)
                if len(points) >= 2:
                    ego_xy = (tf.location.x, tf.location.y)
                    ld = drv.lookahead_min + drv.lookahead_gain * speed
                    target = pick_lookahead_point(points, ego_xy, ld)
                    steer = pure_pursuit_steer(ego_xy, tf.rotation.yaw, target, drv.wheelbase, max_steer)
                else:  # 경로 끝 (막다른 길)
                    steer, throttle, brake = 0.0, 0.0, 1.0
                ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

                head = route.waypoints[0]
                log.writerow([f"{sim_t:.2f}", f"{tf.location.x:.2f}", f"{tf.location.y:.2f}",
                              f"{tf.rotation.yaw:.1f}", f"{speed:.2f}",
                              p.tl_state.value if p.tl_state else "",
                              "" if p.stop_distance is None else f"{p.stop_distance:.2f}",
                              decision.state.value, f"{target_speed:.2f}", f"{curve_limit:.2f}",
                              f"{throttle:.2f}", f"{brake:.2f}", f"{steer:+.3f}",
                              f"{target[0]:.2f}", f"{target[1]:.2f}",
                              head.road_id, head.lane_id, len(points)])

                if decision.state is not prev_state:
                    print(f"[t={sim_t:7.2f}s] {prev_state.value:>14} -> {decision.state.value:<14} "
                          f"v={speed * 3.6:5.1f}km/h  ({decision.reason})")
                    prev_state = decision.state

                if p.tl_state is None:
                    tl_line = ("Light : none", GRAY)
                else:
                    tl_line = (f"Light : {p.tl_state.value:<7} {p.stop_distance:5.1f} m", TL_COLOR[p.tl_state])
                display.draw(image_to_rgb(image), [
                    tl_line,
                    (f"State : {decision.state.value}", STATE_COLOR[decision.state]),
                    (f"        {decision.reason}", GRAY),
                    (f"Speed : {speed * 3.6:5.1f} / {target_speed * 3.6:5.1f} km/h", WHITE),
                    (f"Curve : limit {curve_limit * 3.6:5.1f} km/h",
                     YELLOW if curve_limit < behavior.cruise_speed else GRAY),
                    (f"Ctrl  : thr {throttle:.2f} brk {brake:.2f} str {steer:+.2f}", GRAY),
                    (f"FPS   : {display.fps:4.1f}", CYAN),
                ])
    finally:
        log_file.close()
        display.close()


if __name__ == "__main__":
    main()
