"""YAML 설정 → dataclass 로딩.

설정 파일에 오타 키가 있으면 조용히 무시하지 않고 에러를 낸다.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

T = TypeVar("T")


@dataclass
class CameraConfig:
    width: int = 800
    height: int = 450
    fov: float = 90.0


@dataclass
class SimConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 20.0
    map: str = "Town01"
    fixed_delta_seconds: float = 0.05
    seed: int | None = 42
    vehicle_blueprint: str = "vehicle.tesla.model3"
    spawn_index: int | None = None
    camera: CameraConfig = field(default_factory=CameraConfig)


@dataclass
class PIDGains:
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.02


@dataclass
class DrivingConfig:
    cruise_speed_kmh: float = 30.0
    speed_pid: PIDGains = field(default_factory=PIDGains)
    max_throttle: float = 0.75
    max_brake: float = 1.0
    comfort_decel: float = 2.0
    max_stop_decel: float = 4.0
    stop_margin: float = 2.0
    tl_lookahead: float = 45.0
    curve_lat_accel: float = 2.5
    route_spacing: float = 2.0
    route_horizon: float = 60.0
    lookahead_min: float = 4.0
    lookahead_gain: float = 0.5
    wheelbase: float = 2.9


def _build(cls: type[T], data: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: 알 수 없는 설정 키 {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        # Optional[데이터클래스] (X | None) 도 dict면 데이터클래스로 만든다
        candidates = [hint, *typing.get_args(hint)]
        target = next((c for c in candidates if is_dataclass(c)), None)
        if target is not None and isinstance(value, dict):
            value = _build(target, value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(cls: type[T], path: str | Path) -> T:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _build(cls, data)


def load_sim_config(path: str | Path | None = None) -> SimConfig:
    return load_config(SimConfig, path or CONFIG_DIR / "sim.yaml")


def load_driving_config(path: str | Path | None = None) -> DrivingConfig:
    return load_config(DrivingConfig, path or CONFIG_DIR / "driving.yaml")
