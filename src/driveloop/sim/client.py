"""CARLA 서버 접속, 동기 모드, 액터 생성/정리 유틸리티."""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from typing import Iterator

import carla

from driveloop.config import SimConfig

MAP_LOAD_TIMEOUT = 120.0


def connect(cfg: SimConfig) -> tuple[carla.Client, carla.World]:
    """서버에 접속하고, 설정한 맵이 아니면 로드한다."""
    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(cfg.timeout)
    world = client.get_world()  # 서버가 안 떠 있으면 여기서 timeout 에러

    client_ver, server_ver = client.get_client_version(), client.get_server_version()
    if client_ver != server_ver:
        print(f"[warn] client {client_ver} != server {server_ver}. 버전을 맞추세요.")

    current = world.get_map().name.split("/")[-1]
    if current != cfg.map:
        print(f"[sim] 맵 로드: {current} -> {cfg.map} (수십 초 걸릴 수 있음)")
        client.set_timeout(MAP_LOAD_TIMEOUT)
        world = client.load_world(cfg.map)
        client.set_timeout(cfg.timeout)
    return client, world


@contextmanager
def synchronous_mode(world: carla.World, fixed_delta_seconds: float) -> Iterator[carla.World]:
    """동기 모드로 전환하고, 종료 시(예외 포함) 원래 설정으로 복구한다.

    복구하지 않고 클라이언트가 죽으면 서버가 tick을 기다리며 멈춘 것처럼 보인다.
    그럴 땐 `python scripts/00_check_env.py --reset`.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    try:
        yield world
    finally:
        world.apply_settings(original)


class ActorPool:
    """생성한 액터를 모아두고 with 블록 종료 시 역순으로 파괴한다."""

    def __init__(self) -> None:
        self._actors: list[carla.Actor] = []

    def add(self, actor: carla.Actor) -> carla.Actor:
        self._actors.append(actor)
        return actor

    def __enter__(self) -> "ActorPool":
        return self

    def __exit__(self, *exc) -> None:
        for actor in reversed(self._actors):
            if isinstance(actor, carla.Sensor) and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()
        self._actors.clear()


def spawn_ego(world: carla.World, cfg: SimConfig, rng: random.Random) -> tuple[carla.Vehicle, int]:
    """자차를 스폰한다. 스폰 지점이 막혀 있으면 다른 지점을 시도한다."""
    bp = world.get_blueprint_library().find(cfg.vehicle_blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    points = world.get_map().get_spawn_points()
    if cfg.spawn_index is not None:
        order = [cfg.spawn_index]
    else:
        order = rng.sample(range(len(points)), len(points))
    for idx in order:
        actor = world.try_spawn_actor(bp, points[idx])
        if actor is not None:
            return actor, idx
    raise RuntimeError("자차 스폰 실패: 모든 스폰 지점이 막혀 있음")


def speed_mps(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def follow_with_spectator(world: carla.World, actor: carla.Actor,
                          distance: float = 8.0, height: float = 4.0) -> None:
    """서버 창의 관전자 카메라를 차량 뒤쪽 위에 둔다."""
    tf = actor.get_transform()
    fwd = tf.get_forward_vector()
    loc = carla.Location(
        x=tf.location.x - fwd.x * distance,
        y=tf.location.y - fwd.y * distance,
        z=tf.location.z + height,
    )
    world.get_spectator().set_transform(
        carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw))
    )
