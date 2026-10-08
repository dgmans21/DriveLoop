"""매 tick 인지 → 판단 → 제어 (주행 데모 04와 비교 실험 50이 같은 코드를 쓴다 → 공정한 비교).

인지만 바꿔 끼운다: make_perception(route) 가 GroundTruthPerception 또는 ModelPerception을 돌려준다.
정답값 인지(gt)는 모델 모드에서도 항상 함께 계산해 로그에 남긴다 (판단에는 쓰지 않음).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import carla

from driveloop.config import DrivingConfig
from driveloop.control.lateral import pick_lookahead_point, pure_pursuit_steer, rear_axle
from driveloop.control.pid import LongitudinalController
from driveloop.perception.ground_truth import GroundTruthPerception
from driveloop.perception.types import PerceptionOutput
from driveloop.planning.acc import AccOutput, acc_target, smooth_target
from driveloop.planning.behavior import Decision, TrafficLightBehavior
from driveloop.planning.route import RoutePlanner
from driveloop.planning.speed_limit import curve_speed_limit
from driveloop.sim.client import speed_mps


@dataclass
class StepResult:
    p: PerceptionOutput          # 판단에 쓴 인지
    gt: PerceptionOutput         # 정답값 인지 (비교 기록용)
    gt_tl_id: int | None         # 정답값 기준 내 신호등 actor id
    decision: Decision
    target_speed: float          # 최종: min(신호 판단, 커브 제한, 앞차 ACC)
    speed: float
    transform: carla.Transform
    acc: AccOutput               # 앞차 ACC 상한 (앞차 없으면 target_speed=None)


class DrivingAgent:
    def __init__(self, world: carla.World, ego: carla.Vehicle, drv: DrivingConfig, rng: random.Random,
                 make_perception: Callable[[RoutePlanner, GroundTruthPerception], object] | None = None) -> None:
        self.ego = ego
        self.drv = drv
        self.route = RoutePlanner(world.get_map(), ego.get_location(), drv.route_spacing, drv.route_horizon, rng)
        self.gt = GroundTruthPerception(world, ego, self.route, drv.tl_lookahead, drv.lead_lookahead,
                                        drv.lane_half_width)
        self.perception = make_perception(self.route, self.gt) if make_perception else self.gt
        self.behavior = TrafficLightBehavior(drv)
        self.longitudinal = LongitudinalController(drv)
        self.max_steer = math.radians(ego.get_physics_control().wheels[0].max_steer_angle)
        self._prev_target: float | None = None

    def step(self, image, dt: float) -> StepResult:
        drv = self.drv
        tf = self.ego.get_transform()
        speed = speed_mps(self.ego)
        self.route.update(tf.location)

        p = self.perception.perceive(image)
        nearest = self.gt.nearest_light()
        gt_p = self.gt.perceive() if self.perception is not self.gt else p
        decision = self.behavior.step(p, speed)

        points = self.route.points_xy()
        curve_limit = curve_speed_limit(points, drv.curve_lat_accel, drv.comfort_decel, self.behavior.cruise_speed)
        acc = acc_target(p.lead_distance, p.lead_speed, speed, time_gap=drv.acc_time_gap,
                         standstill_gap=drv.acc_standstill_gap, tau=drv.acc_tau,
                         comfort_decel=drv.comfort_decel, max_decel=drv.max_stop_decel)
        acc_limit = smooth_target(acc, self._prev_target, drv.comfort_decel, dt)   # 간격 항은 편안한 감속으로만
        target_speed = min(decision.target_speed, curve_limit,
                           acc_limit if acc_limit is not None else float("inf"))
        self._prev_target = target_speed
        throttle, brake = self.longitudinal.step(target_speed, speed, dt)
        if len(points) >= 2:
            ego_xy = (tf.location.x, tf.location.y)
            target = pick_lookahead_point(points, rear_axle(ego_xy, tf.rotation.yaw, drv.wheelbase),
                                          drv.lookahead_min + drv.lookahead_gain * speed)
            steer = pure_pursuit_steer(ego_xy, tf.rotation.yaw, target, drv.wheelbase, self.max_steer)
        else:  # 경로 끝 (막다른 길)
            steer, throttle, brake = 0.0, 0.0, 1.0
        self.ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        return StepResult(p, gt_p, nearest[0].id if nearest else None, decision, target_speed, speed, tf, acc)
