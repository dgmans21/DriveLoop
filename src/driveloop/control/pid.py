"""PID 제어기와 종방향(속도) 제어 (CARLA 비의존)."""
from __future__ import annotations

from driveloop.config import DrivingConfig

HOLD_SPEED = 0.5      # m/s, 목표 0이고 이보다 느리면 브레이크로 정차 유지
BRAKE_DEADBAND = 0.05  # 작은 음수 출력은 브레이크 대신 타력 주행


class PID:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 2.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class LongitudinalController:
    """목표 속도 추종 → (throttle, brake)."""

    def __init__(self, cfg: DrivingConfig) -> None:
        g = cfg.speed_pid
        self._pid = PID(g.kp, g.ki, g.kd)
        self._max_throttle = cfg.max_throttle
        self._max_brake = cfg.max_brake

    def step(self, target_speed: float, speed: float, dt: float) -> tuple[float, float]:
        if target_speed < 0.1 and speed < HOLD_SPEED:
            self._pid.reset()
            return 0.0, self._max_brake
        u = self._pid.step(target_speed - speed, dt)
        if u >= 0:
            return min(u, self._max_throttle), 0.0
        if -u < BRAKE_DEADBAND:
            return 0.0, 0.0
        return 0.0, min(-u, self._max_brake)
